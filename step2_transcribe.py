"""
Paso 2: Transcribe el audio con faster-whisper.
Lee:    output/{run_id}/audio.mp3
Guarda: output/{run_id}/transcript.json
Carga GPU solo durante este paso; libera al terminar.
"""

import argparse
import json
import sys
from pathlib import Path
from config import Config, get_media_duration, get_run_dir, setup_logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()

    cfg = Config()
    out_dir = get_run_dir(cfg.output_dir, args.run_id)
    log = setup_logging(out_dir)
    audio_path = out_dir / "audio.mp3"
    transcript_path = out_dir / "transcript.json"

    if not audio_path.exists():
        log.error(f"No se encontró {audio_path}. ¿Corriste el paso 1?")
        sys.exit(1)

    audio_duration = get_media_duration(str(audio_path))
    if audio_duration > 0:
        log.info(f"[Paso 2] Duración de audio: {audio_duration / 60:.1f} minutos")
        cfg.adapt_whisper_for_duration(audio_duration, log)

    log.info(f"[Paso 2] Transcribiendo con Whisper ({cfg.whisper_model}) en {cfg.whisper_device}...")

    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            cfg.whisper_model,
            compute_type=cfg.whisper_compute,
            device=cfg.whisper_device,
        )
        segments, info = model.transcribe(
            str(audio_path),
            language=cfg.whisper_language,
            beam_size=5,
            vad_filter=True,           # 🔥 filtra silencios, reduce alucinaciones
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
            word_timestamps=True,      # 🔥 timestamps por palabra, no solo por segmento
            condition_on_previous_text=True,  # 🔥 coherencia entre segmentos
            temperature=0.0,           # 🔥 determinístico, menos alucinaciones
            no_speech_threshold=0.6,   # 🔥 descarta segmentos con poca confianza
            compression_ratio_threshold=2.4,
        )

        transcript = []
        for seg in segments:
            transcript.append({
                "start": round(seg.start, 2),
                "end":   round(seg.end, 2),
                "text":  seg.text.strip(),
                "words": [{"word": w.word, "start": round(w.start, 2), "end": round(w.end, 2)}
                          for w in (seg.words or [])],
                "confidence": round(getattr(seg, 'avg_logprob', 0), 3),
            })

        log.info(
            f"[Paso 2] {len(transcript)} segmentos | "
            f"idioma: {info.language} ({info.language_probability:.0%})"
        )

        transcript_path.write_text(
            json.dumps(transcript, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info(f"[Paso 2] Transcripción guardada → {transcript_path}")

    except Exception as e:
        log.error(f"[Paso 2] Error: {e}")
        sys.exit(1)
    finally:
        # Liberar modelo y VRAM antes de salir
        try:
            del model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    sys.exit(0)  # Salida explícita exitosa

if __name__ == "__main__":
    main()