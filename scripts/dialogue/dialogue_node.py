#!/usr/bin/python3
"""Dialogue node: Vosk STT + espeak-ng TTS + intent matching + QR fallback.

Single ROS node that owns audio in/out for Task 2. The orchestrator
(``task2.py``) talks to this node only via topics:

  Sub  /dialogue/prompt    (std_msgs/String, JSON)
                            {"text": "...", "gender": "male"|"female"|null,
                             "face_id": int|null, "expects_intent": bool}
  Sub  /dialogue/say       (std_msgs/String) — one-off TTS, no listening
  Sub  /top_camera/rgb/preview/image_raw  (sensor_msgs/Image) — for QR
  Pub  /dialogue/intent    (std_msgs/String, JSON)
                            {"face_id": int|null,
                             "intent": "barrels"|"rings"|"anomaly_red"|
                                       "anomaly_green"|"nothing"|null,
                             "source": "voice"|"qr"|"timeout",
                             "raw": "..."}
  Pub  /arm_command        (std_msgs/String) — used only during QR fallback
                            to position the wrist camera.

Soft dependencies (degrades gracefully if any missing):
  - vosk          → STT
  - sounddevice   → mic capture
  - pyzbar        → QR decoding
  - espeak-ng     → TTS (subprocess on $PATH)
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

from ament_index_python.packages import get_package_share_directory

# Optional deps — import inside try blocks so a missing one only kills the
# matching path, not the whole node.
try:
    from vosk import KaldiRecognizer, Model  # type: ignore
    _VOSK_OK = True
except ImportError:
    KaldiRecognizer = Model = None  # type: ignore
    _VOSK_OK = False

try:
    import sounddevice as sd  # type: ignore
    _SD_OK = True
except (ImportError, OSError):
    sd = None  # type: ignore
    _SD_OK = False

try:
    from pyzbar import pyzbar  # type: ignore
    _ZBAR_OK = True
except ImportError:
    pyzbar = None  # type: ignore
    _ZBAR_OK = False

sys.path.insert(0, os.path.dirname(__file__))
from intent_matcher import (  # noqa: E402
    ALL_INTENTS, classify, classify_qr,
)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
AUDIO_BLOCK = 8000
MAX_LISTEN_S = 8.0           # hard timeout on a single utterance
MAX_REASK = 2                # cap the woman re-ask loop
QR_ARM_SETTLE_S = 2.5
QR_FRAME_WAIT_S = 5.0
ESPEAK_RATE = 140
ESPEAK_VOICE = 'en-us'

# Vosk model location.  Drop the unzipped folder here:
#     <pkg_share>/models/vosk-small-en
# (See README for download instructions.)
DEFAULT_VOSK_DIR = 'models/vosk-small-en'


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class DialogueNode(Node):
    def __init__(self) -> None:
        super().__init__('dialogue_node')

        # ---- ROS plumbing
        self.bridge = CvBridge()
        self.create_subscription(String, '/dialogue/prompt', self._prompt_cb, 10)
        self.create_subscription(String, '/dialogue/say',    self._say_cb,    10)
        # Subscribe to BOTH cameras. OAK-D is preferred for QR fallback
        # because it's already pointed at the worker after task2's approach
        # spin; only when it fails do we go through the slow arm-camera path.
        self.create_subscription(
            Image, '/oakd/rgb/preview/image_raw',
            lambda msg: self._img_cb(msg, 'oakd'),
            qos_profile_sensor_data)
        self.create_subscription(
            Image, '/top_camera/rgb/preview/image_raw',
            lambda msg: self._img_cb(msg, 'top'),
            qos_profile_sensor_data)
        self.intent_pub = self.create_publisher(String, '/dialogue/intent', 10)
        self.arm_cmd_pub = self.create_publisher(String, '/arm_command', 10)

        # ---- State
        self._latest_images: dict[str, Optional[np.ndarray]] = {'oakd': None, 'top': None}
        self._image_lock = threading.Lock()

        # Audio queue (raw int16 bytes from sounddevice callback).
        self._audio_q: queue.Queue[bytes] = queue.Queue(maxsize=64)
        self._is_speaking = threading.Event()  # set while TTS plays — drop audio then

        # Pending prompts, processed one at a time in a worker thread.
        self._prompt_q: queue.Queue[dict] = queue.Queue()
        self._worker = threading.Thread(target=self._dialogue_worker, daemon=True)

        # ---- STT setup
        self._model = self._load_vosk_model()
        self._audio_stream = self._start_audio_stream()

        self._worker.start()
        self.get_logger().info(
            f'Dialogue node ready  (Vosk={_VOSK_OK}, mic={_SD_OK and self._audio_stream is not None}, '
            f'QR={_ZBAR_OK}, espeak-ng={self._which("espeak-ng") is not None})')

    # ----------------------------------------------------------------- setup

    def _load_vosk_model(self):
        if not _VOSK_OK:
            return None
        # Search a few likely locations: $HOME/.ros, install share, source tree.
        candidates = []
        env_path = os.environ.get('VOSK_MODEL_PATH')
        if env_path:
            candidates.append(env_path)
        try:
            share = get_package_share_directory('dis_tutorial3')
            candidates.append(os.path.join(share, DEFAULT_VOSK_DIR))
        except Exception:
            pass
        # Source tree fallback (useful during development with --symlink-install).
        candidates.append(os.path.join(os.path.dirname(__file__), '..', '..',
                                       DEFAULT_VOSK_DIR))
        candidates.append(os.path.expanduser('~/' + DEFAULT_VOSK_DIR))

        for path in candidates:
            path = os.path.abspath(path)
            if os.path.isdir(path):
                self.get_logger().info(f'Loading Vosk model from {path}')
                try:
                    return Model(path)
                except Exception as e:
                    self.get_logger().error(f'Failed to load Vosk model: {e}')
                    return None
        self.get_logger().warn(
            'Vosk model directory not found. Speech recognition disabled — '
            'will rely on QR fallback. Searched:\n  ' + '\n  '.join(candidates))
        return None

    def _start_audio_stream(self):
        if not _SD_OK or self._model is None:
            return None

        def callback(indata, frames, time_info, status):
            if status:
                self.get_logger().debug(f'mic status: {status}')
            if self._is_speaking.is_set():
                return  # don't hear ourselves
            try:
                self._audio_q.put_nowait(bytes(indata))
            except queue.Full:
                # Drop the oldest sample to keep latency bounded.
                try:
                    self._audio_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._audio_q.put_nowait(bytes(indata))
                except queue.Full:
                    pass

        try:
            stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE, blocksize=AUDIO_BLOCK,
                channels=1, dtype='int16', callback=callback)
            stream.start()
            return stream
        except Exception as e:
            self.get_logger().error(f'Could not open microphone: {e}')
            return None

    @staticmethod
    def _which(prog: str) -> Optional[str]:
        for path_dir in os.environ.get('PATH', '').split(os.pathsep):
            cand = os.path.join(path_dir, prog)
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
        return None

    # ----------------------------------------------------------- subscribers

    def _prompt_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data) if msg.data.startswith('{') else {'text': msg.data}
        except (ValueError, TypeError):
            payload = {'text': msg.data}
        self._prompt_q.put(payload)

    def _say_cb(self, msg: String) -> None:
        # Simple one-off TTS — enqueue as a prompt with expects_intent=False.
        self._prompt_q.put({'text': msg.data, 'expects_intent': False})

    def _img_cb(self, msg: Image, cam: str) -> None:
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return
        with self._image_lock:
            self._latest_images[cam] = img

    # --------------------------------------------------------- dialogue flow

    def _dialogue_worker(self) -> None:
        while rclpy.ok():
            try:
                payload = self._prompt_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._handle_prompt(payload)
            except Exception as e:
                self.get_logger().error(f'Dialogue worker error: {e}')

    def _handle_prompt(self, payload: dict) -> None:
        text = str(payload.get('text', '')).strip()
        gender = payload.get('gender')
        face_id = payload.get('face_id')
        expects_intent = payload.get('expects_intent', True)

        if text:
            self._tts(text)
        if not expects_intent:
            return

        intent, raw, source = self._collect_intent(gender)

        result = {
            'face_id': face_id,
            'intent': intent,
            'source': source,
            'raw': raw,
        }
        self.intent_pub.publish(String(data=json.dumps(result)))
        self.get_logger().info(f'/dialogue/intent → {result}')

    def _collect_intent(self, gender: Optional[str]) -> tuple[Optional[str], str, str]:
        """Run the dialogue FSM and return (intent, raw_text, source)."""
        first = self._listen_classify()
        if first is None or first.intent is None:
            return self._qr_fallback()

        if gender == 'male':
            return first.intent, first.raw, 'voice'

        # Gender female (or unknown) → confirm.
        prev_intent = first.intent
        prev_raw = first.raw
        for _ in range(MAX_REASK):
            self._tts('Are you sure?')
            reply = self._listen_classify()
            if reply is None:
                # Silent confirmation timeout → accept what we have.
                return prev_intent, prev_raw, 'voice'

            # If the new utterance carries an intent...
            if reply.intent is not None:
                if reply.intent == prev_intent:
                    # Repeated == confirmed per spec p.9.
                    return prev_intent, reply.raw, 'voice'
                # Different intent: per spec, the new one is the chosen one
                # (women "reconsider"). Continue confirming once more.
                prev_intent = reply.intent
                prev_raw = reply.raw
                continue

            # Intent-less reply: read confirm/reconsider cues.
            if reply.confirm:
                return prev_intent, reply.raw, 'voice'
            if reply.reconsider:
                # Asked to change but didn't say to what — keep asking.
                continue
            # Ambiguous — keep going up to MAX_REASK.

        return prev_intent, prev_raw, 'voice'

    def _listen_classify(self):
        """Listen for one utterance and classify it. Returns MatchResult or None on timeout."""
        if self._model is None or self._audio_stream is None:
            return None
        # Drain any audio that arrived during the just-played TTS.
        self._drain_audio_q()

        rec = KaldiRecognizer(self._model, SAMPLE_RATE)
        rec.SetWords(False)
        deadline = time.monotonic() + MAX_LISTEN_S
        accumulated = ''
        while time.monotonic() < deadline and rclpy.ok():
            try:
                chunk = self._audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if rec.AcceptWaveform(chunk):
                text = json.loads(rec.Result()).get('text', '').strip()
                if text:
                    accumulated = (accumulated + ' ' + text).strip()
                    # We have an end-of-utterance; stop here unless it's empty.
                    break
        if not accumulated:
            # Vosk may still have a partial waveform pending.
            tail = json.loads(rec.FinalResult()).get('text', '').strip()
            accumulated = tail
        if not accumulated:
            return None
        self.get_logger().info(f'STT heard: {accumulated!r}')
        return classify(accumulated)

    def _drain_audio_q(self) -> None:
        try:
            while True:
                self._audio_q.get_nowait()
        except queue.Empty:
            pass

    # --------------------------------------------------------- QR fallback

    def _qr_fallback(self) -> tuple[Optional[str], str, str]:
        """QR fallback via the forward OAK-D only.

        The OAK-D is already pointed at the worker after task2's approach
        spin, so the QR card "next to the person" should be in frame. No
        arm motion needed — saves the ~8 s arm settle + scan + park cycle.
        """
        if not _ZBAR_OK:
            return None, '', 'timeout'

        self.get_logger().info('Voice failed — scanning OAK-D for QR.')
        self._tts('Let me check the code.')

        intent, payload = self._scan_camera('oakd', window_s=QR_FRAME_WAIT_S)
        if intent is None:
            self.get_logger().warn(
                f'QR fallback: no recognised code on OAK-D '
                f'(last raw payload seen: {payload!r}).')
            return None, payload, 'timeout'
        self.get_logger().info(f'QR via OAK-D: {payload!r} → {intent}')
        return intent, payload, 'qr'

    def _scan_camera(self, cam: str, window_s: float) -> tuple[Optional[str], str]:
        """Poll one camera's latest frame for a known QR payload.

        QR text is run through the same sentence parser as voice STT —
        spec shows QRs encoding things like "Detect anomalies in the
        green cell." — and falls back to the short-key dictionary
        (`qr_barrels` etc.) for backwards compatibility.
        """
        deadline = time.monotonic() + window_s
        payload: str = ''
        while time.monotonic() < deadline and rclpy.ok():
            with self._image_lock:
                img = self._latest_images.get(cam)
                img = img.copy() if img is not None else None
            if img is not None:
                for code in pyzbar.decode(img):
                    text = code.data.decode('utf-8', errors='ignore').strip()
                    if not text:
                        continue
                    # 1. Sentence parser (handles full prose).
                    parsed = classify(text)
                    candidate = parsed.intent
                    # 2. Dict-keyed fallback (legacy short QR payloads).
                    if candidate is None:
                        candidate = classify_qr(text)
                    if candidate is not None:
                        return candidate, text
                    if not payload:
                        payload = text  # keep first unrecognised payload for logging
            time.sleep(0.15)
        return None, payload

    # ------------------------------------------------------------------- TTS

    def _tts(self, text: str) -> None:
        if not text:
            return
        self.get_logger().info(f'TTS: {text!r}')
        self._is_speaking.set()
        try:
            proc = subprocess.Popen(
                ['espeak-ng', '-s', str(ESPEAK_RATE), '-v', ESPEAK_VOICE, text],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            proc.wait()
        except FileNotFoundError:
            self.get_logger().warn('espeak-ng not on PATH; TTS skipped.')
        finally:
            # Small grace period for the mic to flush our own audio tail.
            time.sleep(0.2)
            self._is_speaking.clear()
            self._drain_audio_q()

    # --------------------------------------------------------------- cleanup

    def destroy_node(self) -> bool:  # type: ignore[override]
        if self._audio_stream is not None:
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            except Exception:
                pass
        return super().destroy_node()


# ---------------------------------------------------------------------------

def main() -> None:
    rclpy.init(args=None)
    node = DialogueNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

# Silence unused-import warnings from ALL_INTENTS (kept exported for symmetry).
_ = ALL_INTENTS
