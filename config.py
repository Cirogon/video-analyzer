"""
config.py — Configuración compartida entre todos los pasos.
"""

import logging
import os
import sys
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def get_media_duration(media_path: str) -> float:
    """Obtiene la duración de un archivo multimedia usando ffprobe."""
    path = Path(media_path)
    if not path.exists():
        return 0.0

    if shutil.which("ffprobe") is None:
        return 0.0

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


@dataclass
class Config:
    video_path: str = r"C:\Users\Mischard\Downloads\video.mp4"

    # Whisper
    whisper_model:   str           = "small"    # base | small | medium | large-v2
    whisper_compute: str           = "int8"   # int8 | float16 | float32
    whisper_device:  str           = "cuda"    # cpu | cuda
    whisper_language: Optional[str] = "es"     # None = autodetect
    whisper_auto_mode: bool         = True
    whisper_short_threshold_sec: int = 300
    whisper_medium_threshold_sec:int = 900
    whisper_long_threshold_sec: int  = 1800
    whisper_long_model: str         = "small"
    whisper_long_compute: str       = "int8"

    # Detección de escenas
    scene_interval_sec:   int   = 2
    scene_diff_threshold: float = 25.0

    # Ollama — modelo de texto
    ollama_url:     str = "http://localhost:11434/api/generate"
    ollama_model:   str = "llama3.1:latest"    # modelo de texto (mejor que phi3 para análisis)
    ollama_timeout: int = 240  # aumentado de 180s para llama3.1
    ollama_retries: int = 5

    # Ollama — modelo de visión (LLaVA)
    # Cambiá por "moondream" si tu GPU es pequeña; es más rápido y ligero.
    ollama_vision_model: str = "qwen3.5:latest"  # "llava-llama3" | "llava:7b" | "llava:latest" | "moondream"

    # Embeddings — nomic-embed-text para búsqueda semántica en paso 5
    embed_model:    str = "nomic-embed-text:latest"
    semantic_top_k: int = 12  # aumentado de 8 a 12 para más contexto

    # Output
    output_dir:      str = "output"
    extract_clips:   bool = True
    clip_padding_sec: int = 4  # aumentado de 1 a 4 para clips cortos
    frame_sample_interval_sec: int = 10  # muestreo adicional de frames cada N segundos
    frame_max_samples: int = 100         # número máximo de frames a describir en el paso 4

    def __post_init__(self) -> None:
        override = os.environ.get("VIDEO_PATH")
        if override:
            self.video_path = override


    def adapt_whisper_for_duration(self, duration_sec: float, log: logging.Logger | None = None) -> None:
        """Ajusta el modelo Whisper sobre la base de la duración del audio/video."""
        if not self.whisper_auto_mode or duration_sec <= 0:
            return

        if duration_sec <= self.whisper_short_threshold_sec:
            return

        if duration_sec <= self.whisper_medium_threshold_sec:
            new_model = "small"
            new_compute = "float16"
        elif duration_sec <= self.whisper_long_threshold_sec:
            new_model = "small"
            new_compute = "int8"
        else:
            new_model = self.whisper_long_model
            new_compute = self.whisper_long_compute

        if self.whisper_model != new_model or self.whisper_compute != new_compute:
            if log:
                log.info(
                    f"[Config] Audio largo detectado ({duration_sec/60:.1f} min); "
                    f"usando Whisper {new_model} / {new_compute}"
                )
            self.whisper_model = new_model
            self.whisper_compute = new_compute


def get_run_dir(output_dir: str, run_id: str) -> Path:
    p = Path(output_dir) / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run.log"

    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # Evitar handlers duplicados si se llama varias veces
    logger = logging.getLogger("VideoAnalyzer")
    if logger.handlers:
        return logger

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return logger
