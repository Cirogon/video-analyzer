"""
Paso 2b: Analiza las características de energía del audio.
Lee:    output/{run_id}/audio.mp3
Guarda: output/{run_id}/audio_features.json
"""

import argparse
import json
import sys
from pathlib import Path

from config import Config, get_run_dir, setup_logging


def extract_audio_features(audio_path: Path, log) -> dict:
    try:
        import librosa
        import numpy as np
    except ImportError:
        log.warning(
            "Librosa o numpy no están instalados. Se omite el análisis de energía de audio. "
            "Instala librosa para habilitar step2b_audio_features."
        )
        return None

    log.info(f"[Paso 2b] Cargando audio {audio_path}...")
    y, sr = librosa.load(str(audio_path), sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

    mean_rms = float(np.mean(rms))
    std_rms = float(np.std(rms))
    threshold = mean_rms + 1.5 * std_rms
    silence_threshold = max(mean_rms * 0.3, 1e-8)

    energy_peaks = [
        {"time": round(float(t), 2), "energy": round(float(e), 4)}
        for t, e in zip(times, rms)
        if e > threshold
    ]

    silent_frames = [
        (idx, float(t), float(e))
        for idx, (t, e) in enumerate(zip(times, rms))
        if e < silence_threshold
    ]

    silent_segments = []
    if silent_frames:
        start = silent_frames[0][1]
        last = silent_frames[0][1]
        for _, t, _ in silent_frames[1:]:
            if t - last > 1.0:
                if last - start >= 0.2:
                    silent_segments.append({
                        "start": round(start, 2),
                        "end": round(last, 2),
                    })
                start = t
            last = t
        if last - start >= 0.2:
            silent_segments.append({
                "start": round(start, 2),
                "end": round(last, 2)})

    analysis = {
        "duration_sec": round(duration, 2),
        "sr": sr,
        "rms_mean": round(mean_rms, 6),
        "rms_std": round(std_rms, 6),
        "energy_threshold": round(threshold, 6),
        "energy_peaks": energy_peaks,
        "silent_segments": silent_segments,
    }

    log.info(f"[Paso 2b] {len(energy_peaks)} picos de energía detectados, {len(silent_segments)} segmentos de silencio.")
    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()

    cfg = Config()
    out_dir = get_run_dir(cfg.output_dir, args.run_id)
    log = setup_logging(out_dir)

    audio_path = out_dir / "audio.mp3"
    features_path = out_dir / "audio_features.json"

    if not audio_path.exists():
        log.error(f"No se encontró {audio_path}. ¿Corriste el paso 1?")
        sys.exit(1)

    try:
        features = extract_audio_features(audio_path, log)
        if features is None:
            log.info("[Paso 2b] Análisis de audio omitido por dependencias faltantes.")
            sys.exit(0)

        features_path.write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(f"[Paso 2b] Audio features guardados → {features_path}")
    except Exception as e:
        log.error(f"[Paso 2b] Error: {e}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
