"""
Paso 1: Extrae el audio del video con ffmpeg.
Guarda: output/{run_id}/audio.mp3
No carga GPU.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from config import Config, get_run_dir, setup_logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()

    cfg = Config()
    out_dir = get_run_dir(cfg.output_dir, args.run_id)
    log = setup_logging(out_dir)
    audio_path = out_dir / "audio.mp3"

    log.info(f"[Paso 1] Extrayendo audio de: {cfg.video_path}")

    cmd = [
        "ffmpeg", "-y",
        "-i", cfg.video_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-q:a", "2",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error(f"ffmpeg falló:\n{result.stderr[-500:]}")
        sys.exit(1)

    size_kb = audio_path.stat().st_size // 1024
    log.info(f"[Paso 1] Audio guardado → {audio_path} ({size_kb} KB)")
    sys.exit(0)  # Salida explícita exitosa

if __name__ == "__main__":
    main()
