"""
step0_split_video.py — Divide el video en chunks para procesar en paralelo.

Uso:
    python step0_split_video.py --chunk_duration 5        # 5 minutos por chunk
    python step0_split_video.py --chunk_duration 10 --run_id myrun
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from config import Config, get_run_dir, setup_logging


def get_video_duration(video_path: str) -> float:
    """Obtiene la duración del video en segundos usando ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1:noesc=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error obteniendo duración: {e}")
        return 0


def split_video(
    video_path: str,
    output_dir: Path,
    chunk_duration_sec: int,
    log: logging.Logger,
) -> list[str]:
    """
    Divide el video en chunks usando ffmpeg.

    Retorna lista de rutas a los videos divididos.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        log.error(f"Video no encontrado: {video_path}")
        sys.exit(1)

    total_duration = get_video_duration(str(video_path))
    log.info(f"Duración total del video: {total_duration:.1f} segundos ({total_duration/60:.1f} minutos)")

    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunk_files = []
    chunk_num = 0
    current_time = 0

    while current_time < total_duration:
        chunk_num += 1
        chunk_path = chunks_dir / f"chunk_{chunk_num:03d}.mp4"

        # Calcular tiempo de inicio y duración
        start_time = current_time
        duration = min(chunk_duration_sec, total_duration - current_time)

        log.info(
            f"Extrayendo chunk {chunk_num}: "
            f"{start_time:.2f}s - {start_time + duration:.2f}s "
            f"({duration:.2f}s)"
        )

        # Reencodear para cortes frame-accurate (evita drift A/V con -c copy)
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-ss", str(start_time),    # float: no perder sub-segundos
            "-t", str(duration),       # float: ídem
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-avoid_negative_ts", "make_zero",
            str(chunk_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f"Error al extraer chunk {chunk_num}:")
            log.error(result.stderr)
            sys.exit(1)

        chunk_files.append(str(chunk_path))
        current_time += duration

    log.info(f"✓ Video dividido en {chunk_num} chunks")
    return chunk_files


def main():
    parser = argparse.ArgumentParser(description="Dividir video en chunks")
    parser.add_argument(
        "--chunk_duration",
        type=int,
        default=5,
        help="Duración de cada chunk en minutos (default: 5)",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="ID del run (se genera si no se especifica)",
    )
    args = parser.parse_args()

    cfg = Config()
    run_id = args.run_id or "split_test"
    run_dir = get_run_dir(cfg.output_dir, run_id)
    log = setup_logging(run_dir)

    log.info(f"Dividiendo video en chunks de {args.chunk_duration} minutos...")

    chunk_files = split_video(
        cfg.video_path,
        run_dir,
        args.chunk_duration * 60,  # convertir a segundos
        log,
    )

    log.info(f"Chunks creados: {len(chunk_files)}")
    for i, chunk in enumerate(chunk_files, 1):
        log.info(f"  {i}. {chunk}")

    # Guardar lista de chunks para que el pipeline pueda usarla
    chunks_list_file = run_dir / "chunks.txt"
    chunks_list_file.write_text("\n".join(chunk_files))
    log.info(f"Lista de chunks guardada en: {chunks_list_file}")


if __name__ == "__main__":
    main()