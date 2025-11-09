#!/usr/bin/env python3
"""
Generate an interactive HTML demo for Sound-of-Pixels results.

Given a directory produced by the test pipeline (e.g.,
`ckpt/test_upsample_224/test`), this script builds a single-page site
containing:
  * An embedded YouTube video for each sample (video ID inferred from the
    directory name suffix).
  * A pixel-accurate audio overlay on the video — enabling the grid lets you
    click any spatial location and immediately audition the corresponding
    prediction (`pred{row}x{col}.wav`).
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote


PIXEL_AUDIO_PATTERN = re.compile(r"^pred(\d+)x(\d+)\.wav$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Path to the test results folder (e.g. ckpt/test_upsample_224/test).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demo.html"),
        help="Output HTML filename. Defaults to <root>/demo.html when a relative path is given.",
    )
    return parser.parse_args()


def detect_grid(audio_dir: Path) -> Optional[Tuple[int, int]]:
    """Infer grid dimensions from the available `pred{row}x{col}.wav` files."""
    max_row = 0
    max_col = 0
    for wav_path in audio_dir.iterdir():
        if not wav_path.is_file():
            continue
        match = PIXEL_AUDIO_PATTERN.match(wav_path.name)
        if not match:
            continue
        row = int(match.group(1))
        col = int(match.group(2))
        if row > max_row:
            max_row = row
        if col > max_col:
            max_col = col
    if max_row == 0 or max_col == 0:
        return None
    return max_row, max_col


def to_url_path(path: Path) -> str:
    """Convert a filesystem path (relative) into a browser-friendly URL fragment."""
    return quote(path.as_posix(), safe="/")


def gather_samples(root: Path) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    for sample_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        audio_dir = sample_dir / "pred audio"
        if not audio_dir.is_dir():
            continue

        grid = detect_grid(audio_dir)
        if grid is None:
            continue

        rel_sample = sample_dir.relative_to(root)
        name = sample_dir.name
        youtube_id = name.rsplit("-", 1)[-1] if "-" in name else ""

        samples.append(
            {
                "name": name,
                "youtube_id": youtube_id,
                "audio_dir": to_url_path((rel_sample / "pred audio")),
                "grid_h": grid[0],
                "grid_w": grid[1],
            }
        )
    return samples


def build_sample_section(sample: Dict[str, object], index: int) -> str:
    escaped_name = html.escape(str(sample["name"]))
    youtube_id = html.escape(str(sample["youtube_id"]))
    audio_dir = html.escape(str(sample["audio_dir"]))
    grid_h = int(sample["grid_h"])
    grid_w = int(sample["grid_w"])

    iframe_html = (
        f'<iframe '
        f'src="https://www.youtube.com/embed/{youtube_id}?rel=0" '
        f'title="YouTube video player" frameborder="0" '
        f'allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" '
        f'allowfullscreen></iframe>'
        if youtube_id
        else '<div class="video-placeholder warning">YouTube ID not found in directory name.</div>'
    )

    overlay_html = (
        f'<div class="pixel-overlay" data-grid-h="{grid_h}" data-grid-w="{grid_w}">'
        f'<div class="click-indicator" aria-hidden="true"></div>'
        f'<div class="overlay-hint">Click pixel to play audio</div>'
        f'</div>'
    )

    return f"""
    <section class="sample" data-audio-dir="{audio_dir}">
        <h2>{escaped_name}</h2>
        <div class="video-wrapper">
            {iframe_html}
            {overlay_html}
        </div>
        <div class="controls">
            <button type="button" class="toggle-overlay">Enable audio grid</button>
        </div>
        <audio class="audio-player" controls preload="none"></audio>
        <p class="status">Ready.</p>
    </section>
    """


def render_html(samples: List[Dict[str, object]]) -> str:
    sections = "\n".join(
        build_sample_section(sample, idx) for idx, sample in enumerate(samples)
    )

    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Sound-of-Pixels Demo</title>
    <style>
        :root {{
            color-scheme: dark;
        }}
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 2rem;
            background: #101014;
            color: #f0f0f0;
        }}
        h1 {{
            margin-top: 0;
            text-align: center;
            letter-spacing: 0.05em;
        }}
        a {{
            color: #7ecfff;
        }}
        .sample {{
            border: 1px solid #1f2230;
            background: #161922;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 0 auto 2rem;
            max-width: 960px;
            box-shadow: 0 6px 24px rgba(0,0,0,0.25);
        }}
        .sample h2 {{
            margin-top: 0;
            font-size: 1.4rem;
        }}
        .video-wrapper {{
            position: relative;
            width: 100%;
            margin: 0 auto;
            border-radius: 12px;
            overflow: hidden;
            background: #000;
        }}
        .video-wrapper iframe,
        .video-wrapper .video-placeholder {{
            display: block;
            width: 100%;
            aspect-ratio: 16 / 9;
            min-height: 220px;
        }}
        .video-wrapper iframe {{
            border: none;
        }}
        .video-wrapper.overlay-active iframe {{
            filter: brightness(0.7);
            pointer-events: none;
        }}
        .pixel-overlay {{
            position: absolute;
            inset: 0;
            display: none;
            cursor: crosshair;
            pointer-events: none;
        }}
        .video-wrapper.overlay-active .pixel-overlay {{
            display: block;
            pointer-events: auto;
        }}
        .click-indicator {{
            position: absolute;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            border: 2px solid rgba(126, 207, 255, 0.95);
            background: rgba(126, 207, 255, 0.35);
            transform: translate(-50%, -50%);
            pointer-events: none;
            display: none;
            box-shadow: 0 0 12px rgba(126, 207, 255, 0.9);
        }}
        .overlay-hint {{
            position: absolute;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            padding: 0.35rem 0.7rem;
            background: rgba(0, 0, 0, 0.6);
            border-radius: 999px;
            font-size: 0.8rem;
            pointer-events: none;
            display: none;
        }}
        .video-wrapper.overlay-active .overlay-hint {{
            display: block;
        }}
        .controls {{
            margin-top: 1rem;
            display: flex;
            gap: 0.75rem;
            align-items: center;
        }}
        .toggle-overlay {{
            background: #2b7fff;
            border: none;
            color: #fff;
            padding: 0.5rem 1rem;
            border-radius: 999px;
            font-size: 0.95rem;
            cursor: pointer;
            transition: background 0.15s ease;
        }}
        .toggle-overlay:hover {{
            background: #4a94ff;
        }}
        audio.audio-player {{
            display: block;
            margin-top: 1rem;
            width: 100%;
        }}
        .status {{
            margin-top: 0.5rem;
            color: #a8b3d1;
            font-size: 0.95rem;
        }}
        .warning {{
            color: #ff8c8c;
        }}
        .note {{
            font-size: 0.85rem;
        }}
        .instrument-map-details summary {{
            cursor: pointer;
        }}
        .instrument-map {{
            max-width: 100%;
            height: auto;
            margin-top: 0.75rem;
            border-radius: 8px;
            border: 1px solid #2a2f44;
        }}
    </style>
</head>
<body>
    <h1>Sound-of-Pixels Demo</h1>
    <p style="text-align:center;">
        Enable the audio grid for a video, then click any highlighted pixel to audition the separated audio for that location.
    </p>
    {sections}
    <script>
        document.querySelectorAll('.sample').forEach(sample => {{
            const audioDir = sample.dataset.audioDir;
            const audioEl = sample.querySelector('.audio-player');
            const statusEl = sample.querySelector('.status');
            const videoWrapper = sample.querySelector('.video-wrapper');
            const toggleBtn = sample.querySelector('.toggle-overlay');
            const iframe = videoWrapper.querySelector('iframe');
            const overlay = sample.querySelector('.pixel-overlay');
            const indicator = overlay ? overlay.querySelector('.click-indicator') : null;
            const gridH = overlay ? parseInt(overlay.dataset.gridH, 10) : null;
            const gridW = overlay ? parseInt(overlay.dataset.gridW, 10) : null;

            const resetState = () => {{
                if (audioEl) {{
                    audioEl.pause();
                }}
                if (indicator) {{
                    indicator.style.display = 'none';
                }}
                if (statusEl) {{
                    statusEl.textContent = 'Ready.';
                }}
            }};

            if (overlay && gridH && gridW) {{
                overlay.addEventListener('click', evt => {{
                    const rect = overlay.getBoundingClientRect();
                    const relX = Math.min(Math.max(evt.clientX - rect.left, 0), rect.width);
                    const relY = Math.min(Math.max(evt.clientY - rect.top, 0), rect.height);
                    const col = Math.min(gridW, Math.max(1, Math.floor(relX / rect.width * gridW) + 1));
                    const row = Math.min(gridH, Math.max(1, Math.floor(relY / rect.height * gridH) + 1));
                    const wavPath = audioDir + '/pred' + row + 'x' + col + '.wav';

                    if (indicator) {{
                        indicator.style.display = 'block';
                        indicator.style.left = ((col - 0.5) / gridW * 100).toFixed(2) + '%';
                        indicator.style.top = ((row - 0.5) / gridH * 100).toFixed(2) + '%';
                    }}

                    if (audioEl) {{
                        audioEl.pause();
                        audioEl.src = wavPath;
                        audioEl.currentTime = 0;
                        const playPromise = audioEl.play();
                        if (playPromise !== undefined) {{
                            playPromise.catch(err => {{
                                console.error('Audio playback failed:', err);
                            }});
                        }}
                    }}

                    if (statusEl) {{
                        statusEl.textContent = 'Playing pixel (' + row + ', ' + col + ')';
                    }}
                }});
            }}

            if (toggleBtn) {{
                toggleBtn.addEventListener('click', () => {{
                    const active = videoWrapper.classList.toggle('overlay-active');
                    toggleBtn.textContent = active ? 'Disable audio grid' : 'Enable audio grid';
                    if (iframe) {{
                        iframe.style.pointerEvents = active ? 'none' : 'auto';
                    }}
                    if (!active) {{
                        resetState();
                    }} else if (statusEl) {{
                        statusEl.textContent = 'Audio grid enabled. Click a pixel to play its audio.';
                    }}
                }});
            }}
        }});
    </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] Root directory does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    output = args.output
    if not output.is_absolute():
        output = root / output
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    samples = gather_samples(root)
    if not samples:
        print(f"[ERROR] No valid samples found inside {root}", file=sys.stderr)
        sys.exit(1)

    html_content = render_html(samples)
    output.write_text(html_content, encoding="utf-8")
    print(f"[OK] Demo page written to {output}")


if __name__ == "__main__":
    main()
