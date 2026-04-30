"""
Paso 4: Describe visualmente los frames clave con un modelo de vision via Ollama.
Lee:    output/{run_id}/scenes.json       -> timestamps de escenas
        output/{run_id}/audio_features.json -> energy_peaks para priorizar frames (opcional)
        video_path de Config              -> extrae frames en esos timestamps
Guarda: output/{run_id}/frames.json       -> lista de {time, description}
        output/{run_id}/frames/           -> frames extraidos como JPEG en disco

─── Cambios en esta version ────────────────────────────────────────────────
1. Frames en disco: en vez de tener N imagenes en RAM simultaneamente,
   cada frame se extrae y guarda como JPEG en output/{run_id}/frames/,
   y se levanta de a uno al momento de describir. RAM minima.

2. Recuperacion ante HTTP 500: si el runner crashea durante el procesamiento,
   se recarga el modelo via keep_alive y se reintenta. Antes el pipeline
   fallaba en el primer 500.

3. Warmup via /api/show: verifica que el modelo exista sin enviar imagen.
   llava-llama3 crashea con imagenes sinteticas pequenas pero funciona
   con frames JPEG reales de video.

4. Frames en disco son reusados entre corridas: si ya existe el JPEG de
   un timestamp, no se re-extrae del video.

5. No-think mode: se desactiva el razonamiento interno para modelos Qwen3
   y compatibles, via parametro think=False y prefijo /no_think en el prompt.
   Ademas se filtra cualquier bloque <think>...</think> residual en la respuesta.
"""

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

from config import Config, get_run_dir, setup_logging


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_scenes(scenes_path: Path, log) -> tuple[list[float], dict]:
    raw = json.loads(scenes_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        log.info("[Paso 4] scenes.json en formato legacy (lista plana)")
        return [float(t) for t in raw], {}
    timestamps = [float(t) for t in raw.get("timestamps", [])]
    meta       = raw.get("video_meta", {})
    return timestamps, meta


def get_video_info(video_path: str, log) -> tuple[float | None, float | None]:
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.error(f"[Paso 4] No se pudo abrir el video: {video_path}")
            return None, None
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps   = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps <= 0:
            return None, None
        return total / fps, fps
    except Exception as e:
        log.error(f"[Paso 4] Error leyendo video: {e}")
        return None, None


def build_frame_times(
    scenes:         list[float],
    duration:       float,
    audio_features: dict,
    interval:       int,
    max_samples:    int,
    log,
) -> list[float]:
    selected = set(round(t, 2) for t in scenes if 0 <= t <= duration)

    # Picos de energia de audio
    peaks = audio_features.get("energy_peaks", [])
    if peaks:
        buckets: dict[int, float] = {}
        for p in peaks:
            b    = int(p["time"] // 10)
            prev = buckets.get(b)
            if prev is None or p["energy"] > prev:
                buckets[b] = p["time"]
        audio_times = sorted(buckets.values())
        audio_quota = max(0, (max_samples - len(selected)) // 2)
        for t in audio_times[:audio_quota]:
            selected.add(round(t, 2))
        log.info(f"[Paso 4] Frames por energia de audio: {min(len(audio_times), audio_quota)}")

    # Muestreo uniforme
    ts = 0.0
    while ts < duration:
        selected.add(round(ts, 2))
        ts += interval
    selected.add(round(max(0.0, duration - 0.1), 2))

    frame_times = sorted(selected)

    if len(frame_times) > max_samples:
        log.warning(f"[Paso 4] {len(frame_times)} frames → submuestreando a {max_samples}")
        step        = len(frame_times) / max_samples
        frame_times = [frame_times[int(i * step)] for i in range(max_samples)]

    return frame_times


def extract_frames_to_disk(
    video_path: str,
    timestamps: list[float],
    frames_dir: Path,
    log,
) -> dict[float, Path]:
    """
    Extrae frames del video y los guarda como JPEG en frames_dir.
    Abre el video UNA sola vez. Reutiliza JPEGs ya existentes en disco.
    Retorna {timestamp -> path_al_jpeg}.
    """
    import cv2

    frames_dir.mkdir(parents=True, exist_ok=True)
    result: dict[float, Path] = {}

    # Detectar cuales ya existen
    to_extract = []
    for ts in timestamps:
        out_path = frames_dir / f"frame_{ts:.2f}.jpg"
        if out_path.exists():
            result[ts] = out_path
        else:
            to_extract.append(ts)

    if not to_extract:
        log.info(f"[Paso 4] Todos los frames ya existen en disco ({len(result)} archivos)")
        return result

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("[Paso 4] No se pudo abrir el video.")
        return result

    for ts in to_extract:
        out_path = frames_dir / f"frame_{ts:.2f}.jpg"
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if not ret:
            log.warning(f"[Paso 4] Frame no disponible en t={ts:.1f}s")
            continue

        h, w = frame.shape[:2]
        if w > 512:
            scale = 512 / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        result[ts] = out_path

    cap.release()
    log.info(f"[Paso 4] {len(result)}/{len(timestamps)} frames disponibles en {frames_dir}")
    return result


def frame_path_to_b64(path: Path) -> str | None:
    try:
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return None


def unload_all_models(cfg: "Config", log) -> None:
    import requests
    base = cfg.ollama_url.replace("/api/generate", "")
    try:
        ps_res = requests.get(f"{base}/api/ps", timeout=10)
        loaded = [m["name"] for m in ps_res.json().get("models", [])] if ps_res.status_code == 200 else [cfg.ollama_model]
    except Exception as e:
        log.warning(f"[Paso 4] No se pudo consultar /api/ps: {e}")
        loaded = [cfg.ollama_model]

    if not loaded:
        log.info("[Paso 4] Ningun modelo en VRAM.")
        return

    log.info(f"[Paso 4] Descargando de VRAM: {loaded}")
    for model_name in loaded:
        try:
            res = requests.post(
                f"{base}/api/generate",
                json={"model": model_name, "prompt": "", "keep_alive": 0},
                timeout=20,
            )
            status = "OK" if res.status_code == 200 else f"HTTP {res.status_code}"
            log.info(f"[Paso 4]   {model_name}: {status}")
        except Exception as e:
            log.warning(f"[Paso 4]   Error descargando {model_name}: {e}")


def verify_vision_model(cfg: "Config", log) -> bool:
    """
    Verifica que el modelo exista via /api/show sin enviar imagen.
    llava-llama3 crashea con imagenes sinteticas pequenas pero funciona
    con frames JPEG reales — el warmup con imagen era un falso negativo.
    """
    import requests
    base = cfg.ollama_url.replace("/api/generate", "")
    log.info(f"[Paso 4] Verificando modelo '{cfg.ollama_vision_model}'...")
    try:
        res = requests.post(f"{base}/api/show", json={"name": cfg.ollama_vision_model}, timeout=15)
        if res.status_code == 200:
            log.info(f"[Paso 4] Modelo '{cfg.ollama_vision_model}' confirmado y listo.")
            return True
        elif res.status_code == 404:
            log.error(f"[Paso 4] Modelo no encontrado. Instalalalo: ollama pull {cfg.ollama_vision_model}")
            return False
        else:
            log.error(f"[Paso 4] /api/show HTTP {res.status_code}: {res.text[:200]}")
            return False
    except Exception as e:
        log.error(f"[Paso 4] No se pudo conectar con Ollama: {e}")
        return False


def reload_vision_model(cfg: "Config", log) -> None:
    """Recarga el modelo de vision tras un crash HTTP 500."""
    import requests
    log.info(f"[Paso 4] Recargando '{cfg.ollama_vision_model}'...")
    try:
        requests.post(
            cfg.ollama_url,
            json={"model": cfg.ollama_vision_model, "prompt": "", "keep_alive": "5m"},
            timeout=30,
        )
    except Exception as e:
        log.warning(f"[Paso 4] Error recargando modelo: {e}")


def strip_think_blocks(text: str) -> str:
    """
    Elimina bloques <think>...</think> residuales que algunos modelos
    incluyen en la respuesta aunque se les pida no razonar.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# Prefijo /no_think para modelos Qwen3 y compatibles.
# El parametro think=False en el payload es el mecanismo principal;
# el prefijo es fallback para versiones antiguas de Ollama que no lo soporten.
FRAME_PROMPT = (
    "/no_think\n"
    "Describe this video frame in 3 short sentences: "
    "1) What is happening (main action or event). "
    "2) Who appears (people, expressions, reactions). "
    "3) Visual context (indoor/outdoor, lighting, camera angle). "
    "Be specific and factual."
)


def describe_frame(b64_img: str, timestamp: float, cfg: "Config", log) -> str | None:
    import requests

    payload = {
        "model":  cfg.ollama_vision_model,
        "prompt": FRAME_PROMPT,
        "images": [b64_img],
        "stream": False,
        "think":  False,   # desactiva razonamiento interno (Qwen3 y compatibles)
    }

    for attempt in range(1, cfg.ollama_retries + 1):
        try:
            t0      = time.time()
            res     = requests.post(cfg.ollama_url, json=payload, timeout=cfg.ollama_timeout)
            elapsed = time.time() - t0

            if res.status_code == 500:
                wait = 20 * attempt
                log.warning(
                    f"  Intento {attempt} — HTTP 500 en t={timestamp:.1f}s. "
                    f"Recargando modelo, esperando {wait}s..."
                )
                reload_vision_model(cfg, log)
                if attempt < cfg.ollama_retries:
                    time.sleep(wait)
                continue

            if res.status_code != 200:
                raise ValueError(f"HTTP {res.status_code}: {res.text[:200]}")

            raw_description = res.json().get("response", "").strip()
            description     = strip_think_blocks(raw_description)
            log.info(f"  [{timestamp:.1f}s] {description[:80]}... ({elapsed:.1f}s)")
            return description

        except requests.exceptions.Timeout:
            log.warning(f"  Intento {attempt} timeout en t={timestamp:.1f}s")
            if attempt < cfg.ollama_retries:
                time.sleep(5)
        except Exception as e:
            log.warning(f"  Intento {attempt} error en t={timestamp:.1f}s: {e}")
            if attempt < cfg.ollama_retries:
                time.sleep(5)

    return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()

    cfg     = Config()
    out_dir = get_run_dir(cfg.output_dir, args.run_id)
    log     = setup_logging(out_dir)

    scenes_path         = out_dir / "scenes.json"
    frames_path         = out_dir / "frames.json"
    cache_path          = out_dir / "frames_cache.json"
    audio_features_path = out_dir / "audio_features.json"
    frames_dir          = out_dir / "frames"

    if not scenes_path.exists():
        log.error(f"[Paso 4] No se encontro {scenes_path}. Corri el paso 3 primero.")
        sys.exit(1)

    scenes, video_meta = load_scenes(scenes_path, log)

    if len(scenes) == 0:
        mode = video_meta.get("mode", "unknown")
        log.info(f"[Paso 4] 0 escenas detectadas (modo: {mode}). Usando muestreo uniforme.")

    duration = video_meta.get("duration_sec")
    if not duration:
        duration, _ = get_video_info(cfg.video_path, log)
    if not duration:
        log.error("[Paso 4] No se pudo obtener la duracion del video.")
        sys.exit(1)

    log.info(f"[Paso 4] Duracion: {duration:.0f}s ({duration/60:.1f} min)")

    audio_features: dict = {}
    if audio_features_path.exists():
        try:
            audio_features = json.loads(audio_features_path.read_text(encoding="utf-8"))
        except Exception:
            log.warning("[Paso 4] No se pudo leer audio_features.json.")

    interval    = max(1, cfg.frame_sample_interval_sec)
    frame_times = build_frame_times(
        scenes         = scenes,
        duration       = duration,
        audio_features = audio_features,
        interval       = interval,
        max_samples    = cfg.frame_max_samples,
        log            = log,
    )

    log.info(
        f"[Paso 4] {len(frame_times)} frames seleccionados "
        f"(intervalo: {interval}s, max: {cfg.frame_max_samples})"
    )

    # ── Cache ──────────────────────────────────────────────────────────────────
    cached: dict[float, str] = {}
    if cache_path.exists():
        try:
            cached_items = json.loads(cache_path.read_text(encoding="utf-8"))
            cached = {
                item["time"]: item["description"]
                for item in cached_items
                if "time" in item and "description" in item
            }
            log.info(f"[Paso 4] Cache: {len(cached)} frames ya descritos")
        except Exception:
            log.warning("[Paso 4] No se pudo leer el cache.")

    pending = [t for t in frame_times if t not in cached]
    log.info(f"[Paso 4] Pendientes: {len(pending)} frames")

    # ── Verificar modelo y descargar residuales de VRAM ───────────────────────
    if pending:
        unload_all_models(cfg, log)
        if not verify_vision_model(cfg, log):
            sys.exit(1)

    # ── Extraer frames a disco ─────────────────────────────────────────────────
    if pending:
        frame_paths = extract_frames_to_disk(cfg.video_path, pending, frames_dir, log)
    else:
        frame_paths = {}

    # ── Describir frames uno por uno ──────────────────────────────────────────
    log.info(
        f"[Paso 4] Describiendo con {cfg.ollama_vision_model} "
        f"({len(cached)} en cache, {len(pending)} pendientes)..."
    )

    results = []
    for ts in frame_times:

        if ts in cached:
            results.append({"time": ts, "description": cached[ts]})
            continue

        fpath = frame_paths.get(ts)
        if fpath is None or not fpath.exists():
            log.warning(f"  Frame no disponible en t={ts:.1f}s, saltando")
            continue

        # Leer JPEG de disco → base64 → describir → liberar RAM
        b64 = frame_path_to_b64(fpath)
        if b64 is None:
            log.warning(f"  No se pudo leer {fpath}")
            continue

        desc        = describe_frame(b64, ts, cfg, log)
        description = desc or "(sin descripcion)"
        del b64

        results.append({"time": ts, "description": description})
        cached[ts] = description

        # Guardar cache incremental para poder reanudar si falla
        try:
            cache_path.write_text(
                json.dumps(
                    [{"time": t, "description": d} for t, d in sorted(cached.items())],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as e:
            log.warning(f"[Paso 4] Error guardando cache: {e}")

    frames_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info(f"[Paso 4] {len(results)} descripciones guardadas → {frames_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()