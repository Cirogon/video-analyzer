"""
step6c_smart_reframe.py — Smart Reframe: segmentación y encuadre inteligente por plano.

Corre DESPUÉS de step6b y ANTES de step7.

Para cada clip del manifest:
  1. Detecta cortes de cámara internos con ffmpeg (scene detection)
  2. Extrae 1 frame representativo por segmento
  3. Manda el frame a Moondream via Ollama → clasifica en: speaker | audience | slide | dark | other
  4. Según el tipo, calcula crop_cx, crop_cy, zoom, y movement óptimos
  5. Escribe "segments" en el manifest — step7_filmmaker los aplica automáticamente

Fallback sin Ollama: clasificación heurística por luminosidad + detección de personas (YOLO/Haar).

Manifest enriquecido por clip:
  segments: [
    {
      "start":    0.0,       # relativo al clip
      "end":      4.2,
      "type":     "speaker", # speaker | audience | slide | dark | other
      "crop_cx":  0.38,      # centro horizontal del crop (0..1)
      "crop_cy":  0.32,      # centro vertical — cara en tercio superior
      "zoom":     1.15,
      "movement": "push_in", # static | push_in | pull_out | pan_follow
      "confidence": 0.9,     # confianza de la clasificación
    },
    ...
  ]

Uso:
  python step6c_smart_reframe.py --run_id 20260419_211940
  python step6c_smart_reframe.py --run_id 20260419_211940 --force
  python step6c_smart_reframe.py --run_id 20260419_211940 --no_ollama  # solo heurística
"""

import argparse
import base64
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from config import Config, get_run_dir, setup_logging, get_media_duration


# ─── Constantes de encuadre por tipo de plano ────────────────────────────────

SEGMENT_PROFILES = {
    "speaker": {
        "crop_cx":  0.50,   # se sobrescribe si hay protagonist_bbox
        "crop_cy":  0.32,   # cara en tercio superior
        "zoom":     1.15,
        "movement": "push_in",
    },
    "audience": {
        "crop_cx":  0.50,
        "crop_cy":  0.50,
        "zoom":     1.05,
        "movement": "static",
    },
    "slide": {
        "crop_cx":  0.50,
        "crop_cy":  0.50,
        "zoom":     1.00,
        "movement": "static",
    },
    "dark": {
        "crop_cx":  0.50,
        "crop_cy":  0.40,
        "zoom":     1.00,
        "movement": "static",
    },
    "other": {
        "crop_cx":  0.50,
        "crop_cy":  0.40,
        "zoom":     1.05,
        "movement": "static",
    },
}

# Duración mínima de un segmento para ser incluido (segundos)
MIN_SEGMENT_DURATION = 0.8

# Umbral de scene detection — más bajo = más sensible a cortes
SCENE_THRESHOLD = 0.35

# Prompt para Moondream — muy corto, respuesta de 1 palabra
CLASSIFY_PROMPT = (
    "Look at this image. "
    "Is the main subject a speaker on stage, an audience or crowd, "
    "a presentation slide or screen, a dark or black frame, or something else? "
    "Reply with one word only: speaker, audience, slide, dark, or other."
)


# ─── Scene detection con ffmpeg ──────────────────────────────────────────────

def detect_scene_cuts(clip_path: Path, log) -> list[float]:
    """
    Usa ffmpeg blackdetect + select filter para encontrar cortes de escena.
    Retorna lista de timestamps (segundos) donde hay un corte, incluyendo 0.0.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", str(clip_path),
                "-vf", f"select='gt(scene,{SCENE_THRESHOLD})',showinfo",
                "-vsync", "vfr",
                "-f", "null", "-",
            ],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=120,
        )
        # showinfo escribe en stderr: "pts_time:X.XXX"
        cuts = [0.0]
        for line in result.stderr.splitlines():
            m = re.search(r"pts_time:([\d.]+)", line)
            if m:
                t = float(m.group(1))
                # Ignorar cortes muy cercanos al anterior
                if t - cuts[-1] >= MIN_SEGMENT_DURATION:
                    cuts.append(round(t, 3))
        return cuts
    except Exception as e:
        log.warning(f"  scene detection error: {e} — usando clip completo como un segmento")
        return [0.0]


def cuts_to_segments(cuts: list[float], duration: float) -> list[dict]:
    """Convierte lista de cortes en segmentos {start, end}."""
    segments = []
    for i, start in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else duration
        if end - start >= MIN_SEGMENT_DURATION:
            segments.append({"start": round(start, 3), "end": round(end, 3)})
    return segments


# ─── Extracción de frame representativo ──────────────────────────────────────

def extract_frame_b64(clip_path: Path, t_sec: float) -> Optional[str]:
    """
    Extrae un frame en t_sec del clip y lo devuelve como base64 JPEG.
    Escala a 384px de ancho para que Moondream sea rápido.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(str(clip_path))
        # Ir al 30% del segmento para evitar frames de transición
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        if w > 384:
            frame = cv2.resize(frame, (384, int(h * 384 / w)))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception:
        return None


def get_frame_brightness(clip_path: Path, t_sec: float) -> float:
    """Retorna luminosidad media del frame (0..255). Fallback heurístico."""
    try:
        import cv2, numpy as np
        cap = cv2.VideoCapture(str(clip_path))
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return 128.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
    except Exception:
        return 128.0


def count_persons_in_frame(clip_path: Path, t_sec: float) -> int:
    """Cuenta personas en el frame (YOLO si disponible, Haar fallback)."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(clip_path))
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return 0

        # Intentar YOLO primero
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")
            h, w = frame.shape[:2]
            scale = min(1.0, 640 / max(w, h))
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            results = model(small, classes=[0], verbose=False)
            del model
            return sum(len(r.boxes) for r in results)
        except Exception:
            pass

        # Fallback Haar
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(cascade_path))
        h, w = frame.shape[:2]
        scale = min(1.0, 480 / max(w, h))
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(20, 20))
        return len(faces) if len(faces) > 0 else 0
    except Exception:
        return 0


# ─── Clasificación heurística (sin Ollama) ───────────────────────────────────

def classify_heuristic(clip_path: Path, t_sec: float, log) -> tuple[str, float]:
    """
    Clasifica el segmento sin Ollama usando reglas simples:
    - Muy oscuro → dark
    - 1-2 personas → speaker
    - 3+ personas → audience
    - 0 personas + no oscuro → slide o other
    """
    brightness = get_frame_brightness(clip_path, t_sec)

    if brightness < 25:
        return "dark", 0.95

    n_persons = count_persons_in_frame(clip_path, t_sec)

    if n_persons == 0:
        # Podría ser slide o escenario vacío
        return "slide" if brightness > 60 else "other", 0.6

    if n_persons <= 2:
        return "speaker", 0.75

    # 3+ personas → audiencia
    return "audience", 0.80


# ─── Clasificación con Moondream via Ollama ───────────────────────────────────

def classify_with_moondream(
    frame_b64: str,
    cfg: Config,
    log,
) -> tuple[str, float]:
    """
    Manda el frame a Moondream y obtiene la clasificación.
    Retorna (tipo, confianza).

    Moondream en Ollama requiere que la imagen vaya embebida en el campo
    "images" como lista, y el prompt en "prompt" sin marcador especial.
    Funciona tanto con moondream como moondream2.
    """
    import requests

    VALID_TYPES = {"speaker", "audience", "slide", "dark", "other"}

    # Moondream acepta el formato estándar de Ollama vision
    payload = {
        "model":  cfg.smart_model,
        "prompt": CLASSIFY_PROMPT,
        "images": [frame_b64],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 20,  # un poco más que 10 por si responde con artículo
        },
    }

    for attempt in range(3):
        try:
            res = requests.post(
                cfg.ollama_url,
                json=payload,
                timeout=45,
            )
            if res.status_code != 200:
                log.warning(f"    Moondream HTTP {res.status_code} (intento {attempt+1}): {res.text[:100]}")
                time.sleep(2)
                continue

            body = res.json()
            raw  = body.get("response", "").strip().lower()

            # Log de diagnóstico en primer intento
            if attempt == 0 and not raw:
                log.warning(f"    Moondream respuesta vacía. Body keys: {list(body.keys())}")

            if not raw:
                time.sleep(2)
                continue

            # Extraer la primera palabra válida de la respuesta
            for word in re.split(r"[\s.,!?\-:]+", raw):
                word = word.strip()
                if word in VALID_TYPES:
                    return word, 0.90

            # Búsqueda parcial por si responde "this is a speaker" o similar
            for vtype in VALID_TYPES:
                if vtype in raw:
                    return vtype, 0.80

            log.warning(f"    Moondream sin tipo válido en: '{raw[:80]}'")
            return None, 0.0

        except Exception as e:
            log.warning(f"    Moondream error (intento {attempt+1}): {e}")
            time.sleep(2)

    return None, 0.0


# ─── Cálculo de crop por segmento ─────────────────────────────────────────────

def compute_crop_params(
    seg_type: str,
    clip_data: dict,
    segment_idx: int,
    total_segments: int,
) -> dict:
    """
    Calcula los parámetros de crop para un segmento según su tipo.
    Para 'speaker', usa el protagonist_bbox del manifest si está disponible.
    Aplica variación de zoom progresiva para dar dinamismo al clip.
    """
    profile = SEGMENT_PROFILES.get(seg_type, SEGMENT_PROFILES["other"]).copy()

    # Para el orador: usar protagonist_bbox si existe
    if seg_type == "speaker":
        bbox = clip_data.get("protagonist_bbox")
        if bbox:
            profile["crop_cx"] = round(bbox.get("cx", 0.50), 4)
            # cy ajustado al tercio superior de la cabeza
            raw_cy = bbox.get("cy", 0.40)
            profile["crop_cy"] = round(max(0.20, raw_cy - 0.08), 4)

        # Zoom progresivo: empieza más abierto, termina más cerrado
        progress = segment_idx / max(total_segments - 1, 1)
        base_zoom = 1.10
        max_zoom  = 1.25
        profile["zoom"] = round(base_zoom + (max_zoom - base_zoom) * progress * 0.6, 3)

        # Movement según posición en el clip
        if segment_idx == 0:
            profile["movement"] = "push_in"
        elif segment_idx == total_segments - 1:
            profile["movement"] = "pull_out"
        else:
            profile["movement"] = "push_in" if segment_idx % 2 == 0 else "static"

    return profile


# ─── Corrección de color por segmento ────────────────────────────────────────

def compute_color_correction(brightness: float, seg_type: str) -> dict:
    """
    Calcula parámetros de corrección de color para ffmpeg eq filter.
    Retorna dict con brightness, contrast, saturation para el filtro eq.
    """
    corrections = {"brightness": 0.0, "contrast": 1.0, "saturation": 1.0}

    if seg_type == "dark" or brightness < 40:
        # Levantar exposición en frames oscuros
        lift = min(0.25, (60 - brightness) / 150)
        corrections["brightness"] = round(lift, 3)
        corrections["contrast"]   = round(1.0 + lift * 0.5, 3)

    elif seg_type == "speaker":
        # Leve boost de contraste y saturación para el orador
        corrections["contrast"]   = 1.05
        corrections["saturation"] = 1.10

    elif seg_type == "audience":
        # Audience: leve desaturación para que no distraiga
        corrections["saturation"] = 0.90

    return corrections


# ─── Procesamiento de un clip ─────────────────────────────────────────────────

def process_clip(
    clip_data: dict,
    clips_dir: Path,
    cfg: Config,
    log,
    use_ollama: bool,
) -> list[dict]:
    """
    Procesa un clip completo y retorna la lista de segmentos enriquecidos.
    """
    filename = clip_data.get("filename", "")
    clip_path = clips_dir / filename

    if not clip_path.exists():
        log.warning(f"  Clip no encontrado: {clip_path}")
        return []

    duration = get_media_duration(str(clip_path))
    if duration <= 0:
        log.warning(f"  Duración inválida: {clip_path.name}")
        return []

    # ── 1. Detectar cortes de escena ─────────────────────────────────────────
    cuts      = detect_scene_cuts(clip_path, log)
    raw_segs  = cuts_to_segments(cuts, duration)
    log.info(f"  {len(raw_segs)} segmentos detectados en {clip_path.name} ({duration:.1f}s)")

    # ── 2. Clasificar cada segmento ──────────────────────────────────────────
    enriched = []
    speaker_segs = 0

    for i, seg in enumerate(raw_segs):
        # Muestra el frame al 30% del segmento (evita transición inicial)
        t_sample = seg["start"] + (seg["end"] - seg["start"]) * 0.30

        seg_type   = "other"
        confidence = 0.5
        brightness = get_frame_brightness(clip_path, t_sample)

        if use_ollama:
            frame_b64 = extract_frame_b64(clip_path, t_sample)
            if frame_b64:
                ollama_type, ollama_conf = classify_with_moondream(frame_b64, cfg, log)
                if ollama_type:
                    seg_type   = ollama_type
                    confidence = ollama_conf
                else:
                    # Fallback heurístico si Ollama falla
                    seg_type, confidence = classify_heuristic(clip_path, t_sample, log)
            else:
                seg_type, confidence = classify_heuristic(clip_path, t_sample, log)
        else:
            seg_type, confidence = classify_heuristic(clip_path, t_sample, log)

        # ── 3. Calcular crop params ───────────────────────────────────────────
        crop = compute_crop_params(seg_type, clip_data, i, len(raw_segs))
        color = compute_color_correction(brightness, seg_type)

        if seg_type == "speaker":
            speaker_segs += 1

        segment = {
            "start":      seg["start"],
            "end":        seg["end"],
            "duration":   round(seg["end"] - seg["start"], 3),
            "type":       seg_type,
            "confidence": round(confidence, 2),
            "crop_cx":    crop["crop_cx"],
            "crop_cy":    crop["crop_cy"],
            "zoom":       crop["zoom"],
            "movement":   crop["movement"],
            "color":      color,
            "brightness": round(brightness, 1),
        }
        enriched.append(segment)

        log.info(
            f"    [{seg['start']:.1f}s-{seg['end']:.1f}s] "
            f"{seg_type:8s} conf={confidence:.2f} "
            f"zoom={crop['zoom']:.2f} {crop['movement']:10s} "
            f"brightness={brightness:.0f}"
        )

    log.info(
        f"  → {speaker_segs}/{len(enriched)} segmentos de orador "
        f"{'(Moondream)' if use_ollama else '(heurística)'}"
    )
    return enriched


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Smart Reframe — segmentación y encuadre inteligente por plano"
    )
    parser.add_argument("--run_id", required=True)
    parser.add_argument(
        "--force", action="store_true",
        help="Reprocesar clips que ya tienen segments en el manifest",
    )
    parser.add_argument(
        "--no_ollama", action="store_true",
        help="Usar solo clasificación heurística (sin Moondream)",
    )
    args = parser.parse_args()

    cfg     = Config()
    out_dir = get_run_dir(cfg.output_dir, args.run_id)
    log     = setup_logging(out_dir)

    manifest_path = out_dir / "clips" / "manifest.json"
    if not manifest_path.exists():
        log.error(f"Manifest no encontrado. Corré step6 primero.")
        sys.exit(1)

    manifest   = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips_data = manifest.get("clips", [])

    if not clips_data:
        log.error("No hay clips en el manifest.")
        sys.exit(1)

    # Verificar disponibilidad de Ollama
    use_ollama = not args.no_ollama
    if use_ollama:
        try:
            import requests
            r = requests.get(
                cfg.ollama_url.replace("/api/generate", "/api/tags"),
                timeout=5,
            )
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(cfg.smart_model.split(":")[0] in m for m in models):
                log.warning(
                    f"Modelo '{cfg.smart_model}' no encontrado en Ollama "
                    f"(disponibles: {models[:5]}). Usando heurística."
                )
                use_ollama = False
            else:
                log.info(f"[Paso 6c] Moondream disponible: {cfg.smart_model}")
        except Exception as e:
            log.warning(f"Ollama no disponible ({e}). Usando clasificación heurística.")
            use_ollama = False

    log.info(
        f"[Paso 6c] Smart Reframe — {len(clips_data)} clips | "
        f"clasificación: {'Moondream' if use_ollama else 'heurística'}"
    )

    updated = False
    stats   = {"speaker": 0, "audience": 0, "slide": 0, "dark": 0, "other": 0}

    for i, clip_data in enumerate(clips_data, 1):
        filename = clip_data.get("filename", f"clip_{i}.mp4")

        # Saltear si ya tiene segments (a menos que --force)
        if clip_data.get("segments") and not args.force:
            log.info(
                f"  Clip {i}: ya tiene {len(clip_data['segments'])} segments — "
                f"saltando (usá --force para regenerar)"
            )
            continue

        log.info(f"\n[Paso 6c] Clip {i}/{len(clips_data)}: {filename}")

        segments = process_clip(
            clip_data  = clip_data,
            clips_dir  = out_dir / "clips",
            cfg        = cfg,
            log        = log,
            use_ollama = use_ollama,
        )

        if segments:
            clip_data["segments"] = segments
            updated = True

            for seg in segments:
                stats[seg["type"]] = stats.get(seg["type"], 0) + 1

    if updated:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info(f"\n[Paso 6c] Manifest actualizado → {manifest_path}")

    total = sum(stats.values())
    log.info(
        f"\n[Paso 6c] Resumen de {total} segmentos:\n"
        + "\n".join(
            f"  {k:10s}: {v:3d}  ({v/max(total,1)*100:.0f}%)"
            for k, v in stats.items() if v > 0
        )
    )
    log.info(
        "\n[Paso 6c] Siguiente paso:\n"
        "  python step7_director.py  --run_id " + args.run_id + "\n"
        "  python step7_filmmaker.py --run_id " + args.run_id
    )

    sys.exit(0)


if __name__ == "__main__":
    main()
