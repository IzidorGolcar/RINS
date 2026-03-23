#!/usr/bin/env python3

import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    from playsound import playsound
except Exception:
    playsound = None


class SpeechPlayer(Node):

    def __init__(self):
        super().__init__('speech_player')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('audio_dir', os.path.expanduser('~/.ros/competition_audio')),
            ],
        )

        self.audio_dir = os.path.expanduser(str(self.get_parameter('audio_dir').value))
        self.audio_files = {
            'greet': 'greet.wav',
            'red': 'red.wav',
            'green': 'green.wav',
            'blue': 'blue.wav',
            'yellow': 'yellow.wav',
            'unknown': 'unknown.wav',
            'mission_done': 'mission_done.wav',
        }

        self.create_subscription(String, '/speech_event', self._speech_event_callback, 10)
        self.get_logger().info('Speech player initialized and listening on /speech_event.')

    def _speech_event_callback(self, msg: String):
        key = msg.data.strip().lower()
        if not key:
            return

        audio_name = self.audio_files.get(key)
        if audio_name is None:
            self.get_logger().warn(f'Unknown speech key: {key}')
            return

        audio_path = os.path.join(self.audio_dir, audio_name)
        if not os.path.exists(audio_path):
            self.get_logger().warn(f'Audio file not found: {audio_path}')
            return

        if playsound is None:
            self.get_logger().warn('playsound module is not available in this environment.')
            return

        try:
            playsound(audio_path)
        except Exception as e:
            self.get_logger().warn(f'Failed to play {audio_path}: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = SpeechPlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
