from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Anomaly tile screenshots are dumped here by anomaly_detector.py
# (filename pattern: `tile_<id>.png` and `tile_<id>_anomaly.png`).
ANOMALY_TILE_IMG_DIR = '/tmp/anomaly_tiles'


@dataclass
class BarrelEntry:
    id: int
    colour: str
    orientation: str          # 'vertical' | 'horizontal'
    leaking: bool


@dataclass
class TileEntry:
    id: int
    anomalous: bool


@dataclass
class RingsSummary:
    total: int
    per_colour: dict[str, int]


@dataclass
class TaskExecution:
    requested_by: str
    task_type: str            # 'barrels' | 'rings' | 'anomaly_red' | 'anomaly_green' | 'nothing'
    results: Any              # list[BarrelEntry] | list[TileEntry] | RingsSummary | None


@dataclass
class InspectionReport:
    robot_name: str = 'R2D2'
    date_str: str = field(default_factory=lambda: time.strftime('%Y-%m-%d'))
    executions: list[TaskExecution] = field(default_factory=list)

    # ----- recording (called by task2.py during the run) -------------

    def add_execution(self, requested_by: str, task_type: str, results: Any) -> None:
        self.executions.append(TaskExecution(
            requested_by=requested_by,
            task_type=task_type,
            results=results,
        ))

    # ----- finalisation ----------------------------------------------

    def finalize(self, out_dir: str) -> Path:
        """Write the report and return its absolute path.

        Tries PDF first via ``reportlab``; if that import fails, writes
        Markdown so the run output is never lost.
        """
        out_dir_path = Path(out_dir).expanduser()
        out_dir_path.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')

        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import (Image as RLImage,
                                            Paragraph, SimpleDocTemplate,
                                            Spacer, Table, TableStyle)
        except ImportError:
            md_path = out_dir_path / f'inspection_report_{stamp}.md'
            md_path.write_text(self._as_markdown(), encoding='utf-8')
            return md_path

        pdf_path = out_dir_path / f'inspection_report_{stamp}.pdf'
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story: list[Any] = []

        story.append(Paragraph('<b>Inspection report</b>', styles['Title']))
        story.append(Paragraph(f'Date: {self.date_str}', styles['Normal']))
        story.append(Paragraph(f'Robot: {self.robot_name}', styles['Normal']))
        story.append(Spacer(1, 12))

        if not self.executions:
            story.append(Paragraph('<i>No tasks executed.</i>', styles['Normal']))

        for ex in self.executions:
            story.append(Paragraph(
                f'<b>Task: {self._task_title(ex.task_type)}</b>', styles['Heading2']))
            story.append(Paragraph(f'Requested by: {ex.requested_by}', styles['Normal']))
            story.append(Spacer(1, 6))

            data, totals = self._rows_for(ex)
            if data:
                tbl = Table(data, hAlign='LEFT')
                tbl.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#21568A')),
                    ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
                    ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID',       (0, 0), (-1, -1), 0.5, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                     [colors.whitesmoke, colors.lightgrey]),
                ]))
                story.append(tbl)
            if totals:
                story.append(Spacer(1, 6))
                for line in totals:
                    story.append(Paragraph(line, styles['Normal']))

            # For anomaly tasks, embed the per-tile screenshots saved
            # by anomaly_detector.py (raw + overlay side-by-side).
            if ex.task_type.startswith('anomaly') and isinstance(ex.results, list):
                img_rows = self._anomaly_tile_image_rows(ex.results, RLImage, inch)
                if img_rows:
                    story.append(Spacer(1, 8))
                    story.append(Paragraph('<b>Tile screenshots</b>',
                                            styles['Heading3']))
                    img_tbl = Table(img_rows, hAlign='LEFT')
                    img_tbl.setStyle(TableStyle([
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('BOX',    (0, 0), (-1, -1), 0.25, colors.grey),
                        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ]))
                    story.append(img_tbl)
                else:
                    story.append(Paragraph(
                        '<i>(no tile screenshots found in '
                        f'{ANOMALY_TILE_IMG_DIR})</i>',
                        styles['Italic']))

            story.append(Spacer(1, 16))

        doc.build(story)
        return pdf_path

    @staticmethod
    def _anomaly_tile_image_rows(tile_results, RLImage, inch):
        """Build a table of (id, raw, overlay) rows for every tile that
        has on-disk screenshots in ANOMALY_TILE_IMG_DIR."""
        rows = [['ID', 'Tile', 'Anomaly overlay']]
        any_found = False
        for t in tile_results:
            raw = os.path.join(ANOMALY_TILE_IMG_DIR, f'tile_{t.id}.png')
            ovl = os.path.join(ANOMALY_TILE_IMG_DIR, f'tile_{t.id}_anomaly.png')
            raw_cell = (RLImage(raw, width=1.6 * inch, height=1.6 * inch)
                        if os.path.exists(raw) else '—')
            ovl_cell = (RLImage(ovl, width=1.6 * inch, height=1.6 * inch)
                        if os.path.exists(ovl) else '—')
            if os.path.exists(raw) or os.path.exists(ovl):
                any_found = True
            label = f'{t.id} ({"NOK" if t.anomalous else "OK"})'
            rows.append([label, raw_cell, ovl_cell])
        return rows if any_found else []

    # ----- helpers ----------------------------------------------------

    @staticmethod
    def _task_title(task_type: str) -> str:
        return {
            'barrels':       'Barrel inspection',
            'rings':         'Ring counting',
            'anomaly_red':   'Anomaly detection — red cell',
            'anomaly_green': 'Anomaly detection — green cell',
            'nothing':       'Nothing',
        }.get(task_type, task_type)

    def _rows_for(self, ex: TaskExecution) -> tuple[list[list[str]], list[str]]:
        """Build (table rows, summary lines) for one execution."""
        results = ex.results
        if ex.task_type == 'barrels' and isinstance(results, list):
            rows: list[list[str]] = [['Barrel ID', 'Colour', 'Orientation', 'Leak detected']]
            for b in results:
                rows.append([
                    str(b.id),
                    b.colour.capitalize(),
                    b.orientation.capitalize(),
                    'Yes' if b.leaking else 'No',
                ])
            return rows, [f'Total number of barrels inspected: {len(results)}']

        if ex.task_type == 'rings' and isinstance(results, RingsSummary):
            rows = [['Colour', 'Count']]
            for colour, n in sorted(results.per_colour.items()):
                rows.append([colour.capitalize(), str(n)])
            totals = [
                f'Total number of rings detected: {results.total}',
            ]
            return rows, totals

        if ex.task_type.startswith('anomaly') and isinstance(results, list):
            rows = [['Tile ID', 'Status']]
            ok = nok = 0
            for t in results:
                rows.append([str(t.id), 'NOK' if t.anomalous else 'OK'])
                if t.anomalous:
                    nok += 1
                else:
                    ok += 1
            totals = [
                f'Total number of tiles inspected: {len(results)}',
                f'OK: {ok}',
                f'NOK: {nok}',
            ]
            return rows, totals

        return [], ['(no results)']

    def _as_markdown(self) -> str:
        """Plain Markdown fallback when reportlab is unavailable."""
        lines = [
            '# Inspection report',
            '',
            f'- Date: {self.date_str}',
            f'- Robot: {self.robot_name}',
            '',
        ]
        if not self.executions:
            lines.append('_No tasks executed._')
            return '\n'.join(lines)

        for ex in self.executions:
            lines.append(f'## Task: {self._task_title(ex.task_type)}')
            lines.append(f'- Requested by: {ex.requested_by}')
            data, totals = self._rows_for(ex)
            if data:
                lines.append('')
                lines.append('| ' + ' | '.join(data[0]) + ' |')
                lines.append('|' + '|'.join(['---'] * len(data[0])) + '|')
                for row in data[1:]:
                    lines.append('| ' + ' | '.join(row) + ' |')
            for t in totals:
                lines.append(f'- {t}')
            lines.append('')
        return '\n'.join(lines)


# Quick self-test.
def _self_test() -> None:
    rep = InspectionReport(robot_name='R2D2')
    rep.add_execution('Tom', 'rings', RingsSummary(total=5, per_colour={'red': 2, 'blue': 1, 'green': 2}))
    rep.add_execution('Ana', 'barrels', [
        BarrelEntry(id=1, colour='red',   orientation='vertical',   leaking=False),
        BarrelEntry(id=2, colour='green', orientation='horizontal', leaking=False),
        BarrelEntry(id=3, colour='blue',  orientation='horizontal', leaking=True),
    ])
    rep.add_execution('Janez', 'anomaly_red', [
        TileEntry(id=1, anomalous=False),
        TileEntry(id=2, anomalous=False),
        TileEntry(id=3, anomalous=True),
    ])
    out = rep.finalize(out_dir=os.path.expanduser('~/.ros'))
    print(f'Wrote {out}')


if __name__ == '__main__':
    _self_test()
