"""
Paso 3: Detecta cambios de escena con OpenCV.
Lee:    video_path de Config
Guarda: output/{run_id}/scenes.json

─── Algoritmo ────────────────────────────────────────────────────────────────
Combina dos señales para cada par de frames consecutivos (muestreados
cada scene_interval_sec):

  1. Diff de píxeles (absdiff sobre escala de grises): detecta cortes bruscos.
  2. Distancia de Bhattacharyya entre histogramas: confirma cambio real de
     contenido, reduce falsos positivos por movimiento de cámara o zoom.

Un cambio de escena se registra cuando AMBAS señales superan sus umbrales.

─── Umbrales por tipo de video ───────────────────────────────────────────────
El paso 3 detecta automáticamente si el video es un "talking head" o tiene
variedad visual, y ajusta los umbrales en consecuencia:

  - Talking head (persona fija en cámara, fondo estático):
      diff_threshold = 8.0   hist_threshold = 0.12
  - Video con cortes y movimiento:
      diff_threshold = 18.0  hist_threshold = 0.25

La detección es heurística: si los primeros 30 frames tienen diff promedio
< 5.0, se clasifica como talking head.

─── Output ───────────────────────────────────────────────────────────────────
scenes.json guarda un dict con:
  - timestamps: lista de floats (segundos de cada corte detectado)
  - video_meta: fps, duracion, tamaño, modo detectado

Retrocompatibilidad: si scenes.json es una lista plana (formato viejo),
el paso 4 lo lee igual.
"""

import argparse
import json
import sys
from pathlib import Path

from config import Config, get_run_dir, setup_logging


# ─── Umbrales por modo ────────────────────────────────────────────────────────

THRESHOLDS = {
    "talking_head": {"diff": 8.0,  "hist": 0.12},
    "dynamic":      {"diff": 18.0, "hist": 0.25},
}

TALKING_HEAD_DIFF_LIMIT = 5.0  # diff promedio maximo para clasificar como talking head
PROBE_FRAMES            = 30   # frames a samplear para clasificar el video


def probe_video_mode(cap, fps: float, log) -> str:
    """
    Clasifica el video como 'talking_head' o 'dynamic' mirando los primeros
    PROBE_FRAMES frames distribuidos en los primeros 60 segundos.
    """
    import cv2
    diffs   = []
    n       = PROBE_FRAMES
    max_sec = min(60.0, cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 60.0)
    step    = max_sec / n

    prev_gray = None
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, i * step * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray).mean()
            diffs.append(diff)
        prev_gray = gray

    avg_diff = sum(diffs) / len(diffs) if diffs else 0.0
    mode = "talking_head" if avg_diff < TALKING_HEAD_DIFF_LIMIT else "dynamic"
    log.info(
        f"[Paso 3] Modo detectado: {mode.upper()} "
        f"(diff promedio en muestra = {avg_diff:.2f})"
    )
    return mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument(
        "--mode",
        choices=["auto", "talking_head", "dynamic"],
        default="auto",
        help="Forzar modo de deteccion (default: auto)",
    )
    args = parser.parse_args()

    cfg         = Config()
    out_dir     = get_run_dir(cfg.output_dir, args.run_id)
    log         = setup_logging(out_dir)
    scenes_path = out_dir / "scenes.json"

    log.info("[Paso 3] Detectando cambios de escena con OpenCV...")

    try:
        import cv2

        cap = cv2.VideoCapture(cfg.video_path)
        if not cap.isOpened():
            log.error(f"[Paso 3] No se pudo abrir el video: {cfg.video_path}")
            sys.exit(1)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration     = total_frames / fps if fps > 0 else 0.0
        log.info(f"[Paso 3] Video: {duration:.1f}s a {fps:.1f} fps ({width}x{height})")

        # Clasificar tipo de video
        if args.mode == "auto":
            mode = probe_video_mode(cap, fps, log)
        else:
            mode = args.mode
            log.info(f"[Paso 3] Modo forzado: {mode.upper()}")

        th_diff = THRESHOLDS[mode]["diff"]
        th_hist = THRESHOLDS[mode]["hist"]
        log.info(
            f"[Paso 3] Umbrales: diff>{th_diff:.1f}  hist_bhattacharyya>{th_hist:.2f}  "
            f"intervalo={cfg.scene_interval_sec}s"
        )

        scenes    = []
        prev_gray = None
        prev_hist = None
        sec       = 0.0

        while sec < duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])

            if prev_gray is not None:
                diff      = cv2.absdiff(prev_gray, gray).mean()
                hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)

                if diff > th_diff and hist_diff > th_hist:
                    scenes.append(round(sec, 2))

            prev_gray = gray
            prev_hist = hist
            sec      += cfg.scene_interval_sec

        cap.release()

        output = {
            "timestamps": scenes,
            "video_meta": {
                "duration_sec": round(duration, 2),
                "fps":          round(fps, 2),
                "total_frames": total_frames,
                "width":        width,
                "height":       height,
                "mode":         mode,
            },
        }

        scenes_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info(f"[Paso 3] {len(scenes)} escenas guardadas -> {scenes_path}")
        if len(scenes) == 0:
            log.info(
                "[Paso 3] 0 cortes detectados. El paso 4 usara muestreo temporal uniforme. "
                "Para forzar modo con mas sensibilidad: "
                f"python step3_detect_scenes.py --run_id {args.run_id} --mode dynamic"
            )

    except ImportError:
        log.error("[Paso 3] OpenCV no instalado. Instala con: pip install opencv-python")
        sys.exit(1)
    except Exception as e:
        log.error(f"[Paso 3] Error inesperado: {e}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
