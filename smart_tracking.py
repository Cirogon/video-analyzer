"""
smart_tracking.py — Tracking de sujeto multi-persona + Ken Burns suavizado.

Detecta personas en paneles, entrevistas y vlogs usando una cadena de detectores:

  1. YOLOv8n (ultralytics) — detección de personas completas (clase 0).
     → Detecta personas a cualquier distancia, de perfil, parcialmente ocluidas.
     → Ideal para TED talks, vlogs, entrevistas — no solo caras frontales.
     → Sin descarga extra si yolov8n.pt ya está en el sistema (step6b lo usa).
  2. Haar cascade doble (frontal + perfil) — 100% offline, fallback automático.
     → Se activa si ultralytics no está instalado o YOLO falla.

Resolución de protagonista (para paneles con 2+ personas):
  - Consistencia temporal (IoU entre frames) — evita saltos de cámara
  - Tamaño relativo — la persona más grande suele ser el orador activo
  - Energía de audio — picos de audio_features.json guían al protagonista
  - Posición central — en paneles, el moderador suele estar centrado

Pipeline render:
  1. Samplear frames del clip cada SAMPLE_INTERVAL segundos
  2. Detectar personas (YOLOv8n → Haar doble fallback)
  3. Resolver protagonista con scoring combinado
  4. Si hay exactamente 2 actores estables del mismo tamaño → encuadre conjunto
  5. Suavizar trayectoria con filtro gaussiano
  6. Render frame a frame con crop dinámico + zoom del shot_plan del Director

─── Cambios v4 ─────────────────────────────────────────────────────────────
- Detector primario: YOLOv8n (personas completas) reemplaza MediaPipe (caras)
  → Mismo modelo que ya usa step6b — sin dependencias nuevas
  → Detecta persona entera: mejor encuadre busto/cuerpo para Ken Burns
  → Haar doble sigue como fallback offline
- Protagonista por scoring: IoU + tamaño + centralidad + energía de audio
- Soporte de paneles: hasta 6 personas simultáneas
- shot_plan del Director respetado: zoom_peak / push_in / pull_out en render
"""

import math
import subprocess
from pathlib import Path
from typing import Optional

# ─── Constantes ──────────────────────────────────────────────────────────────

SAMPLE_INTERVAL       = 0.5    # segundos entre frames analizados
SMOOTH_WINDOW_SEC     = 2.0    # ventana gaussiana — más larga = movimientos más lentos
ZOOM_MIN              = 1.05   # zoom mínimo (siempre un poco de zoom para el 9:16)
ZOOM_MAX              = 1.40   # zoom máximo
PERSON_REFERENCE_FRAC = 0.55   # fracción del frame que debe ocupar la persona a zoom base
VERTICAL_FOCUS        = 0.28   # qué fracción del bbox de persona usar como centro vertical
HEADROOM              = 0.12   # headroom sobre la cabeza (fracción del bbox)

# Umbral para considerar que hay dos actores en plano conjunto
DUAL_ACTOR_THRESHOLD  = 0.50   # >50% de frames con 2 personas = escena dual
DUAL_SIZE_RATIO       = 3.0    # max ratio de tamaño entre personas para plano conjunto

# Para consistencia temporal de protagonista
IOU_CONTINUITY_THRESH = 0.25

# Peso de energía de audio en el scoring de protagonista (0..1)
ENERGY_WEIGHT = 0.35

# YOLOv8n — mismo modelo que usa step6b, sin descarga extra
_YOLO_MODEL_NAME = "yolov8n.pt"


# ─── Utilidades de bbox ───────────────────────────────────────────────────────

def _iou(a: tuple, b: tuple) -> float:
    """Calcula Intersection over Union entre dos bboxes (x1,y1,x2,y2)."""
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
    return max(0, (x2 - x1) * (y2 - y1))


def _bbox_cx(box: tuple) -> float:
    return (box[0] + box[2]) / 2


# ─── Detección con YOLOv8n (detector primario — personas completas) ───────────

def _detect_with_yolo(frames: list[dict], log) -> list[list[tuple]]:
    """
    Detecta TODAS las personas por frame usando YOLOv8n (clase 0 = person).

    Ventajas sobre MediaPipe FaceDetector:
      - Detecta persona COMPLETA (no solo cara): mejor encuadre para Ken Burns
      - Funciona de perfil, a cualquier distancia, parcialmente ocluida
      - Sin descarga extra: usa el mismo yolov8n.pt que step6b
      - Retorna bboxes de CUERPO → zoom natural sin cortar cabeza ni pies

    Output: all_detections — lista por frame de bboxes (x1,y1,x2,y2) en [0,1].

    Nota: bboxes de persona ya incluyen cabeza + cuerpo completo, no hace falta
    expandir como en MediaPipe donde solo se detectaba la cara.
    """
    from ultralytics import YOLO

    model = YOLO(_YOLO_MODEL_NAME)
    all_detections: list[list[tuple]] = []

    for item in frames:
        frame   = item["frame"]
        h_f, w_f = frame.shape[:2]

        results = model(frame, classes=[0], verbose=False)
        frame_boxes: list[tuple] = []

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < 0.35:          # filtrar detecciones poco confiables
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                frame_boxes.append((
                    max(0.0, x1 / w_f),
                    max(0.0, y1 / h_f),
                    min(1.0, x2 / w_f),
                    min(1.0, y2 / h_f),
                ))

        all_detections.append(frame_boxes)

    del model
    return all_detections


# ─── Detección con Haar cascade doble (fallback 100% offline) ─────────────────

def _detect_with_haar(frames: list[dict]) -> list[list[tuple]]:
    """
    Fallback offline: Haar cascade frontal + perfil (bundled en opencv-python).
    Detecta hasta N caras por frame en paneles.
    Desespeja las caras de perfil de la imagen espejada para capturar ambos lados.
    Deduplica por IoU para evitar doble-detección.
    """
    import cv2

    cascade_front   = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )
    cascade_profile = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_profileface.xml")
    )
    all_detections: list[list[tuple]] = []

    for item in frames:
        frame    = item["frame"]
        h_src, w_src = frame.shape[:2]
        scale    = min(1.0, 640 / max(w_src, h_src))
        small    = cv2.resize(frame, (int(w_src * scale), int(h_src * scale)))
        gray     = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gs_h, gs_w = gray.shape[:2]

        # Tres pasadas: frontal, perfil, perfil espejado
        faces_front = cascade_front.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20),
        )
        faces_profile = cascade_profile.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20),
        )
        gray_flip = cv2.flip(gray, 1)
        faces_profile_flip = cascade_profile.detectMultiScale(
            gray_flip, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20),
        )

        raw_boxes = []
        for x, y, fw, fh in (list(faces_front)   if len(faces_front)   else []):
            raw_boxes.append((x, y, fw, fh, False))
        for x, y, fw, fh in (list(faces_profile) if len(faces_profile) else []):
            raw_boxes.append((x, y, fw, fh, False))
        for x, y, fw, fh in (list(faces_profile_flip) if len(faces_profile_flip) else []):
            # Des-espejear: x → gs_w - x - fw
            raw_boxes.append((gs_w - x - fw, y, fw, fh, True))

        # Normalizar, expandir a busto, deduplicar por IoU
        seen: list[tuple]        = []
        frame_boxes: list[tuple] = []

        for x, y, fw, fh, _ in raw_boxes:
            x1 = x / gs_w
            y1 = y / gs_h
            x2 = (x + fw) / gs_w
            y2 = (y + fh) / gs_h
            face_h = y2 - y1
            # Expandir para incluir cabeza + cuello
            y1 = max(0.0, y1 - face_h * 0.25)
            y2 = min(1.0, y2 + face_h * 0.55)
            bbox = (x1, y1, x2, y2)
            if any(_iou(bbox, s) > 0.4 for s in seen):
                continue
            seen.append(bbox)
            frame_boxes.append(bbox)

        all_detections.append(frame_boxes)

    return all_detections


# ─── Scoring de protagonista ──────────────────────────────────────────────────

def _score_protagonist(
    boxes: list[tuple],
    prev_bbox: Optional[tuple],
    energy_active: bool,
) -> Optional[tuple]:
    """
    Elige el mejor candidato a protagonista en un frame usando scoring ponderado:
      - IoU con protagonista del frame anterior (continuidad temporal) — 50%
      - Tamaño relativo del bbox                                        — 25%
      - Centralidad horizontal (moderador suele estar centrado)         — 25% × (1-ENERGY_WEIGHT)
      - Bonus de energía de audio (sube tamaño + centralidad)          — ENERGY_WEIGHT

    En el primer frame (prev_bbox=None), gana el más grande.
    """
    if not boxes:
        return None
    if len(boxes) == 1:
        return boxes[0]

    scores = []
    for box in boxes:
        area = _bbox_area(box)
        cx   = _bbox_cx(box)

        iou_score    = _iou(prev_bbox, box) if prev_bbox else 0.0
        size_score   = min(area / 0.20, 1.0)
        center_score = 1.0 - abs(cx - 0.5) * 2.0
        energy_bonus = (size_score * 0.5 + center_score * 0.5) if energy_active else 0.0

        total = (
            iou_score    * 0.50 +
            size_score   * 0.25 +
            center_score * (1.0 - ENERGY_WEIGHT) * 0.25 +
            energy_bonus * ENERGY_WEIGHT
        )
        scores.append((total, box))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[0][1]


def _resolve_protagonist(
    all_detections: list[list[tuple]],
    sample_times: list[float],
    energy_peaks: list[float] | None = None,
    clip_start: float = 0.0,
) -> list[Optional[tuple]]:
    """
    Resuelve el protagonista frame a frame con scoring combinado.

    energy_peaks: timestamps absolutos de picos de energía del audio_features.json.
                  Se consideran activos si están a ±1.5s del frame actual.
    clip_start:   tiempo absoluto del inicio del clip en el video original.
    """
    protagonist_track: list[Optional[tuple]] = []
    prev_bbox: Optional[tuple] = None

    for i, boxes in enumerate(all_detections):
        if not boxes:
            protagonist_track.append(None)
            continue

        t_abs = clip_start + (sample_times[i] if i < len(sample_times) else 0.0)
        energy_active = bool(
            energy_peaks and any(abs(p - t_abs) < 1.5 for p in energy_peaks)
        )

        best = _score_protagonist(boxes, prev_bbox, energy_active)
        protagonist_track.append(best)
        if best is not None:
            prev_bbox = best

    return protagonist_track


# ─── Detección de escena de dos actores ──────────────────────────────────────

def _detect_dual_actor_scene(all_detections: list[list[tuple]]) -> bool:
    """
    Detecta si el clip es una escena de dos actores estables y de tamaño comparable.
    Solo aplica encuadre conjunto cuando hay EXACTAMENTE 2 personas — en paneles
    de 3+ se sigue al protagonista individual.
    """
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


def _compute_joint_framing(
    all_detections: list[list[tuple]],
    sample_times: list[float],
) -> list[dict]:
    """Encuadre conjunto para escenas de dos actores."""
    track = []
    for boxes, t in zip(all_detections, sample_times):
        two_boxes = [b for b in boxes if _bbox_area(b) > 0][:2]
        if len(two_boxes) < 2:
            track.append({"t": t, "cx": 0.5, "cy": 0.30, "zoom": ZOOM_MIN, "mode": "joint_fallback"})
            continue

        cx_a = (two_boxes[0][0] + two_boxes[0][2]) / 2
        cx_b = (two_boxes[1][0] + two_boxes[1][2]) / 2
        cy_a = two_boxes[0][1] + (two_boxes[0][3] - two_boxes[0][1]) * VERTICAL_FOCUS
        cy_b = two_boxes[1][1] + (two_boxes[1][3] - two_boxes[1][1]) * VERTICAL_FOCUS

        cx_joint   = (cx_a + cx_b) / 2
        cy_joint   = (cy_a + cy_b) / 2
        x1_joint   = min(two_boxes[0][0], two_boxes[1][0])
        x2_joint   = max(two_boxes[0][2], two_boxes[1][2])
        y1_joint   = min(two_boxes[0][1], two_boxes[1][1])
        y2_joint   = max(two_boxes[0][3], two_boxes[1][3])
        joint_frac = math.sqrt((x2_joint - x1_joint) * (y2_joint - y1_joint))
        zoom       = max(ZOOM_MIN, min(ZOOM_MAX, PERSON_REFERENCE_FRAC / max(joint_frac, 0.05) * 0.8))

        track.append({
            "t":    t,
            "cx":   max(0.0, min(1.0, cx_joint)),
            "cy":   max(0.05, min(0.9, cy_joint)),
            "zoom": zoom,
            "mode": "joint",
        })

    return track


# ─── Conversión de bboxes a track cx/cy/person_frac ──────────────────────────

def _protagonist_bboxes_to_track(
    protagonist_boxes: list[Optional[tuple]],
    sample_times: list[float],
) -> list[dict]:
    """Convierte bboxes protagonistas a formato {t, cx, cy, person_frac}."""
    track = []
    for bbox, t in zip(protagonist_boxes, sample_times):
        if bbox is None:
            track.append({"t": t, "cx": None, "cy": None, "person_frac": None})
            continue

        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        cx = x1 + bw / 2
        cy = y1 + bh * VERTICAL_FOCUS - bh * HEADROOM

        track.append({
            "t":           t,
            "cx":          max(0.0, min(1.0, cx)),
            "cy":          max(0.05, min(0.9, cy)),
            "person_frac": math.sqrt(bw * bh),
        })

    return track


# ─── Suavizado gaussiano ──────────────────────────────────────────────────────

def _gaussian_weights(n: int, sigma: float) -> list[float]:
    half = n // 2
    w = [math.exp(-0.5 * ((i - half) / sigma) ** 2) for i in range(n)]
    s = sum(w)
    return [x / s for x in w]


def _smooth_track(raw: list[dict], sample_fps: float, window_sec: float) -> list[dict]:
    if not raw:
        return raw

    if raw[0].get("mode") in ("joint", "joint_fallback"):
        return _smooth_joint_track(raw, sample_fps, window_sec)

    filled = [dict(r) for r in raw]

    for field in ("cx", "cy", "person_frac"):
        valid = [i for i, r in enumerate(filled) if r.get(field) is not None]
        if not valid:
            default = 0.5 if field == "cx" else (0.3 if field == "cy" else PERSON_REFERENCE_FRAC)
            for r in filled:
                r[field] = default
            continue
        for i in range(valid[0]):
            filled[i][field] = filled[valid[0]][field]
        for i in range(valid[-1] + 1, len(filled)):
            filled[i][field] = filled[valid[-1]][field]
        for a, b in zip(valid, valid[1:]):
            if b - a <= 1:
                continue
            va, vb = filled[a][field], filled[b][field]
            for k in range(a + 1, b):
                alpha = (k - a) / (b - a)
                filled[k][field] = va + alpha * (vb - va)

    for r in filled:
        pf   = max(r.get("person_frac") or PERSON_REFERENCE_FRAC, 0.05)
        zoom = PERSON_REFERENCE_FRAC / pf
        r["zoom"] = max(ZOOM_MIN, min(ZOOM_MAX, zoom))

    win = max(5, int(window_sec * sample_fps))
    if win % 2 == 0:
        win += 1
    weights = _gaussian_weights(win, win / 5)
    half    = win // 2

    smoothed = []
    for i, r in enumerate(filled):
        cx = cy = zo = 0.0
        for j, wt in enumerate(weights):
            idx = max(0, min(len(filled) - 1, i - half + j))
            cx += filled[idx]["cx"]   * wt
            cy += filled[idx]["cy"]   * wt
            zo += filled[idx]["zoom"] * wt
        smoothed.append({
            "t":    r["t"],
            "cx":   round(cx, 4),
            "cy":   round(cy, 4),
            "zoom": round(zo, 4),
        })

    return smoothed


def _smooth_joint_track(raw: list[dict], sample_fps: float, window_sec: float) -> list[dict]:
    if not raw:
        return raw

    filled  = [dict(r) for r in raw]
    win     = max(5, int(window_sec * sample_fps))
    if win % 2 == 0:
        win += 1
    weights = _gaussian_weights(win, win / 5)
    half    = win // 2

    smoothed = []
    for i, r in enumerate(filled):
        cx = cy = zo = 0.0
        for j, wt in enumerate(weights):
            idx = max(0, min(len(filled) - 1, i - half + j))
            cx += filled[idx].get("cx",   0.5)      * wt
            cy += filled[idx].get("cy",   0.30)     * wt
            zo += filled[idx].get("zoom", ZOOM_MIN) * wt
        smoothed.append({
            "t":    r["t"],
            "cx":   round(cx, 4),
            "cy":   round(cy, 4),
            "zoom": round(zo, 4),
            "mode": "joint",
        })

    return smoothed


# ─── Interpolación con shot_plan del Director ─────────────────────────────────

def _ease_inout(p: float) -> float:
    """Suavizado cúbico ease-in-out para transiciones entre shots."""
    return p * p * (3 - 2 * p)


def _interp(track: list[dict], t: float, shot_plan: list[dict] | None = None) -> dict:
    """
    Interpola posición de encuadre para el tiempo t.

    Con shot_plan (del Director AI — paso 7): aplica instrucciones cinematográficas
    con transiciones suaves de 0.5s entre shots (ease-in-out).

    Movimientos del shot_plan:
      - push_in:   zoom progresivo hacia el sujeto (+0.12 sobre base)
      - pull_out:  zoom progresivo hacia afuera  (-0.12 sobre base)
      - zoom_peak: zoom in-out en V (+0.15 en el pico) — ideal para énfasis en discurso
      - static:    zoom fijo

    Sin shot_plan: interpola la trayectoria suavizada del tracker.
    """
    TRANSITION_SEC = 0.5

    if shot_plan:
        active_shot = None
        prev_shot   = None
        for i, shot in enumerate(shot_plan):
            if shot.get("t_start", 0) <= t <= shot.get("t_end", float("inf")):
                active_shot = shot
                prev_shot   = shot_plan[i - 1] if i > 0 else None
                break

        if active_shot is not None:
            t_start    = active_shot.get("t_start", 0)
            t_end      = active_shot.get("t_end", t + 1)
            duration_s = max(0.001, t_end - t_start)
            progress   = (t - t_start) / duration_s
            base_zoom  = float(active_shot.get("zoom", 1.1))
            movement   = active_shot.get("movement", "static")

            if movement == "push_in":
                zoom = base_zoom + progress * 0.12
            elif movement == "pull_out":
                zoom = base_zoom - progress * 0.12
            elif movement == "zoom_peak":
                # Forma de V: sube en primera mitad, baja en segunda
                zoom = base_zoom + 0.15 * (1 - abs(2 * progress - 1))
            else:
                zoom = base_zoom

            zoom      = max(ZOOM_MIN, min(ZOOM_MAX, zoom))
            target_cx = float(active_shot.get("cx", 0.5))
            target_cy = float(active_shot.get("cy", 0.30))

            if prev_shot and (t - t_start) < TRANSITION_SEC:
                alpha    = _ease_inout((t - t_start) / TRANSITION_SEC)
                prev_cx  = float(prev_shot.get("cx", 0.5))
                prev_cy  = float(prev_shot.get("cy", 0.30))
                final_cx = prev_cx + alpha * (target_cx - prev_cx)
                final_cy = prev_cy + alpha * (target_cy - prev_cy)
            else:
                final_cx = target_cx
                final_cy = target_cy

            return {"t": t, "cx": final_cx, "cy": final_cy, "zoom": zoom}

    # Sin shot_plan: interpolar track suavizado
    if not track:
        return {"t": t, "cx": 0.5, "cy": 0.30, "zoom": ZOOM_MIN}
    if t <= track[0]["t"]:
        return track[0]
    if t >= track[-1]["t"]:
        return track[-1]
    for a, b in zip(track, track[1:]):
        if a["t"] <= t <= b["t"]:
            dt    = b["t"] - a["t"]
            alpha = (t - a["t"]) / dt if dt > 0 else 0.0
            return {
                "t":    t,
                "cx":   a["cx"]   + alpha * (b["cx"]   - a["cx"]),
                "cy":   a["cy"]   + alpha * (b["cy"]   - a["cy"]),
                "zoom": a["zoom"] + alpha * (b["zoom"] - a["zoom"]),
            }
    return track[-1]


# ─── Render Ken Burns (OpenCV → pipe ffmpeg) ──────────────────────────────────

def render_ken_burns(
    clip_path: Path,
    track: list[dict],
    w_out: int,
    h_out: int,
    output_path: Path,
    log,
    use_gpu: bool = False,
    shot_plan: list[dict] | None = None,
    clip_start: float = 0.0,
    clip_duration: float | None = None,
) -> bool:
    """
    Renderiza el clip con reencuadre dinámico frame a frame (OpenCV).
    Lee cada frame, aplica crop centrado en el sujeto + zoom,
    escala a resolución de salida y pipe a ffmpeg para encodear.
    Sin audio — el paso siguiente mezcla el audio original.

    Si shot_plan está disponible (Director AI, paso 7), los movimientos
    push_in / pull_out / zoom_peak se aplican en cada frame.
    """
    import cv2

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        log.warning(f"  render_ken_burns: no se pudo abrir {clip_path.name}")
        return False

    fps_src      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w_src        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames == 0 or w_src == 0 or h_src == 0:
        cap.release()
        log.warning("  render_ken_burns: dimensiones inválidas")
        return False

    target_ratio = w_out / h_out
    if w_src / h_src > target_ratio:
        base_w = int(h_src * target_ratio)
        base_h = h_src
    else:
        base_w = w_src
        base_h = int(w_src / target_ratio)

    vcodec = (
        ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20", "-b:v", "0"]
        if use_gpu else
        ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]
    )

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w_out}x{h_out}",
        "-pix_fmt", "bgr24",
        "-r", "30",
        "-i", "pipe:0",
        "-an",
        *vcodec,
        "-pix_fmt", "yuv420p",
        "-video_track_timescale", "90000",
        str(output_path),
    ]

    try:
        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except Exception as e:
        cap.release()
        log.error(f"  render_ken_burns: error lanzando ffmpeg: {e}")
        return False

    # Seek al inicio del clip dentro del source
    if clip_start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, clip_start * 1000)

    # Límite de frames a procesar
    max_frames = int((clip_duration or (total_frames / fps_src)) * fps_src) + 1

    frame_idx = 0

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        t  = frame_idx / fps_src
        pt = _interp(track, t, shot_plan)

        zoom = pt["zoom"]
        cx   = pt["cx"]
        cy   = pt["cy"]

        cw = max(1, min(int(base_w / zoom), w_src))
        ch = max(1, min(int(base_h / zoom), h_src))

        cx_px = int(cx * w_src)
        cy_px = int(cy * h_src)
        x1    = max(0, min(cx_px - cw // 2, w_src - cw))
        y1    = max(0, min(cy_px - ch // 2, h_src - ch))

        cropped = frame[y1:y1+ch, x1:x1+cw]
        if cropped.size == 0:
            cropped = frame

        resized = cv2.resize(cropped, (w_out, h_out), interpolation=cv2.INTER_LANCZOS4)

        try:
            proc.stdin.write(resized.tobytes())
        except BrokenPipeError:
            break

        frame_idx += 1

    cap.release()
    try:
        proc.stdin.close()
    except Exception:
        pass

    _, stderr = proc.communicate(timeout=60)
    if proc.returncode != 0:
        log.warning(f"  render_ken_burns error:\n{stderr[-300:].decode('utf-8', errors='replace')}")
        return False

    size_mb = output_path.stat().st_size / (1024*1024) if output_path.exists() else 0
    log.info(f"    Ken Burns: {frame_idx} frames → {size_mb:.1f} MB")
    return True


# ─── API pública ──────────────────────────────────────────────────────────────

def track_subject_path(
    clip_path: Path,
    log,
    sample_interval: float = SAMPLE_INTERVAL,
    smooth_window:   float = SMOOTH_WINDOW_SEC,
    dense: bool = False,
    clip_data: dict | None = None,
) -> Optional[list[dict]]:
    """
    Analiza el clip y devuelve la trayectoria suavizada del protagonista.

    Cadena de detección:
      1. YOLOv8n (ultralytics) — personas completas, mismo modelo que step6b
      2. Haar cascade doble (frontal + perfil) — 100% offline, fallback automático

    Resolución de protagonista con scoring combinado:
      - IoU temporal (continuidad) + tamaño + centralidad + energía de audio
      - En paneles 3+: sigue al protagonista individualmente
      - En diálogos de 2 personas de igual tamaño: encuadre conjunto automático

    clip_data: dict del manifest (para leer audio_features y start del clip).
               Si None, se corre sin energía de audio.

    Returns: lista de { t, cx, cy, zoom } o None si no detecta nada.
    """
    try:
        import cv2
    except ImportError:
        log.warning("  OpenCV no disponible")
        return None

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None

    fps_src      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration     = total_frames / fps_src

    sample_times = []
    t = 0.0
    while t < duration:
        sample_times.append(round(t, 3))
        t += sample_interval
    if not sample_times:
        sample_times = [0.0]

    frames_data = []
    for st in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, st * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            scale = min(1.0, 854 / max(w, h))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            frames_data.append({"t": st, "frame": frame})

    cap.release()

    if not frames_data:
        return None

    # ── Leer energía de audio del clip ────────────────────────────────────────
    energy_peaks: list[float] = []
    clip_start = 0.0
    if clip_data:
        clip_start = clip_data.get("start", 0.0)
        raw_peaks  = clip_data.get("audio_features", {}).get("energy_peaks", [])
        for p in raw_peaks:
            if isinstance(p, dict):
                energy_peaks.append(float(p.get("time", 0)))
            elif isinstance(p, (int, float)):
                energy_peaks.append(float(p))

    # ── Detectar personas por frame ──────────────────────────────────────────
    # Cadena: YOLOv8n (personas completas) → Haar doble (fallback offline)
    all_detections: list[list[tuple]] = []
    detector_used = "Haar"

    try:
        all_detections = _detect_with_yolo(frames_data, log)
        detector_used  = "YOLOv8n"
        n_with_person  = sum(1 for d in all_detections if d)
        log.info(
            f"    YOLOv8n: {n_with_person}/{len(all_detections)} frames con personas "
            f"({n_with_person/len(all_detections)*100:.0f}%)"
        )
    except Exception as e:
        log.info(f"    YOLOv8n no disponible ({type(e).__name__}) → Haar cascade doble")
        all_detections = _detect_with_haar(frames_data)
        n_with_person  = sum(1 for d in all_detections if d)
        log.info(
            f"    Haar doble: {n_with_person}/{len(all_detections)} frames con personas "
            f"({n_with_person/len(all_detections)*100:.0f}%)"
        )

    # Liberar frames de RAM
    for item in frames_data:
        del item["frame"]

    # ── Sin detecciones → crop estático centrado ──────────────────────────────
    if not any(d for d in all_detections):
        log.info("    Sin detección → crop estático centrado")
        return [{"t": t, "cx": 0.5, "cy": 0.30, "zoom": ZOOM_MIN} for t in sample_times]

    # ── Escena de dos actores → encuadre conjunto ─────────────────────────────
    if _detect_dual_actor_scene(all_detections):
        dual_count = sum(1 for d in all_detections if len(d) == 2)
        log.info(
            f"    Escena de dos actores ({dual_count}/{len(all_detections)} frames) "
            f"→ encuadre conjunto"
        )
        joint_track = _compute_joint_framing(all_detections, sample_times)
        return _smooth_track(joint_track, 1.0 / sample_interval, smooth_window)

    # ── Protagonista con scoring combinado ────────────────────────────────────
    protagonist_boxes = _resolve_protagonist(
        all_detections,
        sample_times=sample_times,
        energy_peaks=energy_peaks or None,
        clip_start=clip_start,
    )
    valid_count  = sum(1 for b in protagonist_boxes if b is not None)
    energy_info  = f" | {len(energy_peaks)} picos de audio" if energy_peaks else ""
    log.info(
        f"    Protagonista ({detector_used}): {valid_count}/{len(protagonist_boxes)} frames"
        f"{energy_info}"
    )

    raw_track = _protagonist_bboxes_to_track(protagonist_boxes, sample_times)
    smoothed  = _smooth_track(raw_track, 1.0 / sample_interval, smooth_window)

    if smoothed:
        mid = smoothed[len(smoothed) // 2]
        log.info(
            f"    Trayectoria: {len(smoothed)} pts | "
            f"cx={mid['cx']:.2f} cy={mid['cy']:.2f} zoom={mid['zoom']:.2f}"
        )

    return smoothed


def track_to_static_bbox(track: list[dict]) -> dict:
    if not track:
        return {"cx": 0.5, "cy": 0.30}
    cx = sum(p["cx"] for p in track) / len(track)
    cy = sum(p["cy"] for p in track) / len(track)
    return {"cx": round(cx, 4), "cy": round(cy, 4)}


def build_ken_burns_filter(track, w_src, h_src, w_out, h_out) -> str:
    """DEPRECATED — se mantiene por compatibilidad."""
    bbox         = track_to_static_bbox(track)
    target_ratio = w_out / h_out
    if w_src / h_src > target_ratio:
        crop_h = h_src
        crop_w = int(h_src * target_ratio)
        crop_x = max(0, min(int(w_src * bbox["cx"] - crop_w / 2), w_src - crop_w))
        crop_y = 0
    else:
        crop_w = w_src
        crop_h = int(w_src / target_ratio)
        crop_x = 0
        crop_y = max(0, min(int(h_src * bbox["cy"] - crop_h / 2), h_src - crop_h))
    return f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={w_out}:{h_out}"


def cache_track(clip_data: dict, track: list[dict]) -> None:
    clip_data["subject_track"] = [
        {"t": p["t"], "cx": p["cx"], "cy": p["cy"], "zoom": p["zoom"]}
        for p in track
    ]


def load_cached_track(clip_data: dict) -> Optional[list[dict]]:
    return clip_data.get("subject_track")

def load_visual_modes(frames_path) -> dict:
    """Carga visual modes desde frames.json para encuadre inteligente."""
    try:
        import json
        from pathlib import Path
        p = Path(frames_path)
        if not p.exists():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "frames" in data:
            return {str(k): v for k, v in data["frames"].items() if isinstance(v, dict)}
        return {}
    except Exception:
        return {}