"""
Paso 6b: Detecta el protagonista en los clips extraídos y actualiza el manifest.

Lee:
    output/{run_id}/clips/manifest.json
    output/{run_id}/clips/clip_N.mp4

Escribe:
    output/{run_id}/clips/manifest.json

El manifest se enriquece con:
    protagonist_bbox:  {cx, cy, w, h}  — valores relativos 0..1
                       o null si no se detecta cara dominante
    tracking_mode:     "single"         — un solo actor
                       "dual"           — dos actores, encuadre conjunto
                       "static"         — sin detección, crop centrado
    actor_count:       número promedio de actores detectados por frame

─── Cambios v2 ─────────────────────────────────────────────────────────────
- Usa YOLOv8n cuando está disponible (detecta personas a cualquier distancia,
  de perfil, parcialmente ocluidas — no solo caras frontales).
- Fallback a Haar cascade si ultralytics no está instalado.
- Detecta escenas de dos actores y marca tracking_mode = "dual".
- Resolución de protagonista con consistencia temporal (IoU entre frames).
- Haar cascade también detecta múltiples caras (no solo la más grande).
"""

import argparse
import json
import math
import sys
from pathlib import Path

from config import Config, get_run_dir, setup_logging

# Umbrales — alineados con smart_tracking.py
DUAL_ACTOR_THRESHOLD = 0.50
DUAL_SIZE_RATIO      = 3.0
IOU_THRESH           = 0.25


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _bbox_area(box: tuple) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, (x2 - x1) * (y2 - y1))


def _detect_all_persons_yolo(
    frames_data: list[dict],
    log,
) -> tuple[list[list[tuple]], str]:
    """
    Detecta TODAS las personas por frame usando YOLOv8n.
    Retorna (all_detections, detector_name).
    Cada detección es un bbox normalizado (x1, y1, x2, y2) en 0..1.
    """
    from ultralytics import YOLO
    import cv2

    model = YOLO("yolov8n.pt")
    all_detections: list[list[tuple]] = []

    for item in frames_data:
        frame   = item["frame"]
        h_f, w_f = frame.shape[:2]
        results = model(frame, classes=[0], verbose=False)

        frame_boxes: list[tuple] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                frame_boxes.append((x1/w_f, y1/h_f, x2/w_f, y2/h_f))
        all_detections.append(frame_boxes)

    del model
    return all_detections, "yolov8n"


def _detect_all_faces_haar(
    frames_data: list[dict],
) -> tuple[list[list[tuple]], str]:
    """
    Detecta TODAS las caras frontales por frame usando Haar cascade.
    Retorna (all_detections, detector_name).
    """
    import cv2

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        return [[] for _ in frames_data], "none"

    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        return [[] for _ in frames_data], "none"

    all_detections: list[list[tuple]] = []

    for item in frames_data:
        frame   = item["frame"]
        h_src, w_src = frame.shape[:2]
        scale   = min(1.0, 640 / max(w_src, h_src))
        small   = cv2.resize(frame, (int(w_src * scale), int(h_src * scale)))
        gray    = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gs_h, gs_w = gray.shape[:2]

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor  = 1.05,
            minNeighbors = 4,
            minSize      = (20, 20),
            flags        = cv2.CASCADE_SCALE_IMAGE,
        )

        frame_boxes: list[tuple] = []
        for x, y, fw, fh in faces if len(faces) > 0 else []:
            frame_boxes.append((
                x / gs_w,
                y / gs_h,
                (x + fw) / gs_w,
                (y + fh) / gs_h,
            ))
        all_detections.append(frame_boxes)

    return all_detections, "haar"


def _is_dual_actor_scene(all_detections: list[list[tuple]]) -> bool:
    """Igual lógica que en smart_tracking — dos actores de tamaño comparable."""
    two_person_frames = 0
    comparable_frames = 0

    for boxes in all_detections:
        if len(boxes) == 2:
            two_person_frames += 1
            area_a = _bbox_area(boxes[0])
            area_b = _bbox_area(boxes[1])
            if area_a > 0 and area_b > 0:
                ratio = max(area_a, area_b) / min(area_a, area_b)
                if ratio < DUAL_SIZE_RATIO:
                    comparable_frames += 1

    total = len(all_detections)
    if total == 0:
        return False
    return (
        two_person_frames / total >= DUAL_ACTOR_THRESHOLD and
        comparable_frames / total >= DUAL_ACTOR_THRESHOLD * 0.8
    )


def _resolve_protagonist_bbox(all_detections: list[list[tuple]]) -> list[tuple | None]:
    """
    Resuelve el protagonista frame a frame con consistencia temporal (IoU).
    """
    result: list[tuple | None] = []
    prev: tuple | None = None

    for boxes in all_detections:
        if not boxes:
            result.append(None)
            continue

        if prev is None:
            best = max(boxes, key=_bbox_area)
            result.append(best)
            prev = best
            continue

        best_box = max(boxes, key=lambda b: _iou(prev, b))
        best_iou = _iou(prev, best_box)

        if best_iou >= IOU_THRESH:
            result.append(best_box)
            prev = best_box
        else:
            # Ruptura de continuidad: reiniciar con la persona más grande
            best_by_size = max(boxes, key=_bbox_area)
            result.append(best_by_size)
            prev = best_by_size

    return result


def _compute_protagonist_avg(protagonist_boxes: list[tuple | None]) -> dict | None:
    """Promedia los bboxes del protagonista para obtener una posición representativa."""
    valid = [b for b in protagonist_boxes if b is not None]
    if not valid:
        return None

    avg_cx = sum((b[0] + b[2]) / 2 for b in valid) / len(valid)
    avg_cy = sum((b[1] + b[3]) / 2 for b in valid) / len(valid)
    avg_w  = sum(b[2] - b[0] for b in valid) / len(valid)
    avg_h  = sum(b[3] - b[1] for b in valid) / len(valid)

    return {
        "cx": round(avg_cx, 4),
        "cy": round(avg_cy, 4),
        "w":  round(avg_w,  4),
        "h":  round(avg_h,  4),
    }


def _compute_joint_bbox(all_detections: list[list[tuple]]) -> dict | None:
    """
    Para escena dual: bounding box que contiene a ambos actores en promedio.
    """
    frames_with_two = [boxes for boxes in all_detections if len(boxes) >= 2]
    if not frames_with_two:
        return None

    cx_list, cy_list, w_list, h_list = [], [], [], []

    for boxes in frames_with_two:
        b0, b1 = boxes[0], boxes[1]
        x1 = min(b0[0], b1[0])
        y1 = min(b0[1], b1[1])
        x2 = max(b0[2], b1[2])
        y2 = max(b0[3], b1[3])
        cx_list.append((x1 + x2) / 2)
        cy_list.append((y1 + y2) / 2)
        w_list.append(x2 - x1)
        h_list.append(y2 - y1)

    return {
        "cx": round(sum(cx_list) / len(cx_list), 4),
        "cy": round(sum(cy_list) / len(cy_list), 4),
        "w":  round(sum(w_list)  / len(w_list),  4),
        "h":  round(sum(h_list)  / len(h_list),  4),
    }


def analyze_clip(clip_path: Path, log) -> dict:
    """
    Analiza el clip y retorna:
      {
        protagonist_bbox:  {...} | null,
        tracking_mode:     "single" | "dual" | "static",
        actor_count:       float,
        detector:          "yolov8n" | "haar" | "none",
      }
    """
    try:
        import cv2
    except ImportError:
        log.error("OpenCV no disponible.")
        return {"protagonist_bbox": None, "tracking_mode": "static", "actor_count": 0, "detector": "none"}

    if not clip_path.exists():
        log.warning(f"Clip no encontrado: {clip_path}")
        return {"protagonist_bbox": None, "tracking_mode": "static", "actor_count": 0, "detector": "none"}

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return {"protagonist_bbox": None, "tracking_mode": "static", "actor_count": 0, "detector": "none"}

    fps           = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames  = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration      = total_frames / fps
    sample_times  = [t for t in [i * 1.0 for i in range(int(duration) + 1)] if t < duration]
    if not sample_times:
        sample_times = [0.0]

    frames_data = []
    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        scale = min(1.0, 640 / max(w, h))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        frames_data.append({"t": t, "frame": frame})
    cap.release()

    if not frames_data:
        return {"protagonist_bbox": None, "tracking_mode": "static", "actor_count": 0, "detector": "none"}

    # ── Detectar personas/caras ───────────────────────────────────────────────
    detector = "none"
    all_detections: list[list[tuple]] = []

    try:
        import ultralytics  # noqa
        all_detections, detector = _detect_all_persons_yolo(frames_data, log)
        log.info(f"  → YOLOv8n")
    except Exception as e:
        log.info(f"  → Haar cascade ({type(e).__name__})")
        all_detections, detector = _detect_all_faces_haar(frames_data)

    # Liberar frames de RAM
    for item in frames_data:
        del item["frame"]

    # ── Estadísticas ──────────────────────────────────────────────────────────
    actor_counts   = [len(d) for d in all_detections]
    avg_actors     = sum(actor_counts) / max(len(actor_counts), 1)
    any_detection  = any(d for d in all_detections)

    if not any_detection:
        return {
            "protagonist_bbox": None,
            "tracking_mode":    "static",
            "actor_count":      round(avg_actors, 2),
            "detector":         detector,
        }

    # ── Escena de dos actores ─────────────────────────────────────────────────
    if _is_dual_actor_scene(all_detections):
        joint_bbox = _compute_joint_bbox(all_detections)
        dual_frames = sum(1 for d in all_detections if len(d) == 2)
        log.info(
            f"  → Escena dual ({dual_frames}/{len(all_detections)} frames) "
            f"— encuadre conjunto"
        )
        return {
            "protagonist_bbox": joint_bbox,
            "tracking_mode":    "dual",
            "actor_count":      round(avg_actors, 2),
            "detector":         detector,
        }

    # ── Un solo protagonista ──────────────────────────────────────────────────
    protagonist_boxes = _resolve_protagonist_bbox(all_detections)
    valid_count       = sum(1 for b in protagonist_boxes if b is not None)
    log.info(f"  → Protagonista único ({valid_count}/{len(protagonist_boxes)} frames)")

    avg_bbox = _compute_protagonist_avg(protagonist_boxes)
    return {
        "protagonist_bbox": avg_bbox,
        "tracking_mode":    "single" if avg_bbox else "static",
        "actor_count":      round(avg_actors, 2),
        "detector":         detector,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Detecta el protagonista (o plano conjunto) en los clips y actualiza el manifest"
    )
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()

    cfg     = Config()
    out_dir = get_run_dir(cfg.output_dir, args.run_id)
    log     = setup_logging(out_dir)

    manifest_path = out_dir / "clips" / "manifest.json"
    if not manifest_path.exists():
        log.error(f"No se encontró {manifest_path}. Ejecutá primero step6_extract_clips.py")
        sys.exit(1)

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.error(f"Error leyendo JSON de {manifest_path}")
        sys.exit(1)

    clips = manifest.get("clips", [])
    if not clips:
        log.error("Manifest no contiene clips.")
        sys.exit(1)

    stats = {"single": 0, "dual": 0, "static": 0}

    for clip in clips:
        filename = clip.get("filename")
        if not filename:
            log.warning("Clip sin filename en manifest, se omite")
            continue

        clip_path = out_dir / "clips" / filename
        log.info(f"[Paso 6b] {filename}...")

        result = analyze_clip(clip_path, log)

        clip["protagonist_bbox"] = result["protagonist_bbox"]
        clip["tracking_mode"]    = result["tracking_mode"]
        clip["actor_count"]      = result["actor_count"]
        clip["detector"]         = result["detector"]

        mode = result["tracking_mode"]
        stats[mode] = stats.get(mode, 0) + 1

        if result["protagonist_bbox"]:
            log.info(
                f"  ✓ mode={mode}  bbox={result['protagonist_bbox']}  "
                f"actores_promedio={result['actor_count']:.1f}  det={result['detector']}"
            )
        else:
            log.info(f"  ✗ sin detección — crop centrado  det={result['detector']}")

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log.info(
        f"\n[Paso 6b] Listo → {manifest_path}\n"
        f"  Single:   {stats.get('single', 0)} clips\n"
        f"  Dual:     {stats.get('dual', 0)} clips\n"
        f"  Static:   {stats.get('static', 0)} clips"
    )


if __name__ == "__main__":
    main()