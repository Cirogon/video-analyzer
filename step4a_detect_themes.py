"""
Paso 4.5: Detecta temas recurrentes del video usando el LLM de texto.

Lee:    output/{run_id}/frames.json      → descripciones visuales (ya generadas por paso 4)
        output/{run_id}/transcript.json  → transcripción con timestamps
Guarda: output/{run_id}/themes.json      → lista de temas con timestamps y relevancia

NO vuelve a llamar al modelo de visión. Lee lo que el paso 4 ya generó
y usa el LLM de texto (ollama_model) para identificar temas.

─── Contrato de salida (themes.json) ────────────────────────────────────────
Lista de objetos:
  {
    "tema":       "nombre del tema",
    "descripcion": "descripción breve",
    "timestamps": [t1, t2, ...],   # segundos donde aparece
    "relevancia": 1-10
  }

El paso 5 usa themes.json para enfocar el análisis narrativo.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from config import Config, get_run_dir, setup_logging

THEMES_PROMPT = """\
Sos un analista de contenido de video. Analizá el siguiente material y \
detectá los TEMAS RECURRENTES del video.

DESCRIPCIONES VISUALES DE FRAMES CLAVE:
{frames_summary}

FRAGMENTOS DE TRANSCRIPCIÓN:
{transcript_summary}

Devolvé ÚNICAMENTE un JSON válido con esta estructura (sin texto extra, sin backticks):
{{
  "temas": [
    {{
      "tema": "<nombre corto del tema>",
      "descripcion": "<descripción de 1-2 oraciones>",
      "timestamps": [<lista de segundos donde aparece>],
      "relevancia": <1-10>
    }}
  ],
  "tema_principal": "<el tema más importante del video en una frase>",
  "tipo_contenido": "<charla|entrevista|documental|tutorial|otro>"
}}

Identificá entre 3 y 8 temas. Ordenalos por relevancia descendente.
"""


def build_frames_summary(frames: list[dict], max_entries: int = 30) -> str:
    """
    Construye un resumen de frames para el prompt.
    Usa un subconjunto representativo para no exceder el contexto.
    """
    if not frames:
        return "(sin descripciones de frames)"

    # Subsampling uniforme si hay muchos frames
    if len(frames) > max_entries:
        step = len(frames) / max_entries
        frames = [frames[int(i * step)] for i in range(max_entries)]

    lines = []
    for f in frames:
        t    = f.get("time", 0)
        desc = f.get("description", "").strip()
        if desc and desc != "(sin descripción)":
            # Truncar descripciones largas
            desc_short = desc[:150].replace("\n", " ")
            lines.append(f"[{t:.0f}s] {desc_short}")

    return "\n".join(lines) if lines else "(sin descripciones útiles)"


def build_transcript_summary(transcript: list[dict], max_chars: int = 3000) -> str:
    """
    Construye un resumen del transcript para el prompt.
    Concatena segmentos hasta el límite de chars.
    """
    if not transcript:
        return "(sin transcripción)"

    parts = []
    total = 0
    for seg in transcript:
        text = seg.get("text", "").strip()
        if not text:
            continue
        t    = seg.get("start", 0)
        line = f"[{t:.0f}s] {text}"
        if total + len(line) > max_chars:
            parts.append("... (truncado)")
            break
        parts.append(line)
        total += len(line)

    return "\n".join(parts) if parts else "(sin texto)"


def parse_themes_json(raw: str, log) -> dict | None:
    """Extrae y parsea el JSON de la respuesta del LLM."""
    # Limpiar bloques de código si los hay
    clean = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    clean = clean.strip("`").strip()

    # Intentar parseo directo
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Buscar el primer bloque JSON en la respuesta
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    log.warning("[Paso 4.5] No se pudo parsear JSON de la respuesta del LLM")
    log.warning(f"[Paso 4.5] Respuesta raw (primeros 300 chars): {raw[:300]}")
    return None


def call_llm(prompt: str, cfg: "Config", log) -> str | None:
    """Llama al LLM de texto (no visión) via Ollama."""
    import requests

    payload = {
        "model":  cfg.ollama_model,
        "prompt": prompt,
        "stream": False,
        "think":  False,
    }

    for attempt in range(1, cfg.ollama_retries + 1):
        try:
            t0  = time.time()
            res = requests.post(cfg.ollama_url, json=payload, timeout=cfg.ollama_timeout)
            elapsed = time.time() - t0

            if res.status_code != 200:
                raise ValueError(f"HTTP {res.status_code}: {res.text[:200]}")

            response = res.json().get("response", "").strip()
            # Limpiar bloques <think> si el modelo los genera
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            log.info(f"[Paso 4.5] LLM respondió en {elapsed:.1f}s ({len(response)} chars)")
            return response

        except Exception as e:
            log.warning(f"[Paso 4.5] Intento {attempt} fallido: {e}")
            if attempt < cfg.ollama_retries:
                time.sleep(5)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Detecta temas recurrentes del video usando el LLM de texto"
    )
    parser.add_argument("--run_id", required=True)
    parser.add_argument(
        "--non_interactive",
        action="store_true",
        help="No pedir confirmación interactiva de temas (modo pipeline)",
    )
    args = parser.parse_args()

    cfg     = Config()
    out_dir = get_run_dir(cfg.output_dir, args.run_id)
    log     = setup_logging(out_dir)

    frames_path     = out_dir / "frames.json"
    transcript_path = out_dir / "transcript.json"
    themes_path     = out_dir / "themes.json"

    # ── Verificar que el paso 4 ya corrió ────────────────────────────────────
    if not frames_path.exists():
        log.error(
            f"[Paso 4.5] No se encontró {frames_path}. "
            "Corré el paso 4 primero (step4_describe_frames.py)."
        )
        sys.exit(1)

    # ── Leer frames y transcript ──────────────────────────────────────────────
    try:
        frames = json.loads(frames_path.read_text(encoding="utf-8"))
        log.info(f"[Paso 4.5] {len(frames)} frames cargados desde frames.json")
    except Exception as e:
        log.error(f"[Paso 4.5] Error leyendo frames.json: {e}")
        sys.exit(1)

    transcript: list[dict] = []
    if transcript_path.exists():
        try:
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
            log.info(f"[Paso 4.5] {len(transcript)} segmentos de transcript cargados")
        except Exception:
            log.warning("[Paso 4.5] No se pudo leer transcript.json — continuando sin él")

    # ── Construir prompt ──────────────────────────────────────────────────────
    frames_summary     = build_frames_summary(frames, max_entries=30)
    transcript_summary = build_transcript_summary(transcript, max_chars=3000)

    prompt = THEMES_PROMPT.format(
        frames_summary     = frames_summary,
        transcript_summary = transcript_summary,
    )

    log.info(
        f"[Paso 4.5] Detectando temas con {cfg.ollama_model} "
        f"({len(frames_summary)} chars de frames, {len(transcript_summary)} chars de transcript)..."
    )

    # ── Llamar al LLM ─────────────────────────────────────────────────────────
    raw_response = call_llm(prompt, cfg, log)
    if not raw_response:
        log.error("[Paso 4.5] El LLM no respondió después de todos los reintentos.")
        sys.exit(1)

    # ── Parsear respuesta ─────────────────────────────────────────────────────
    parsed = parse_themes_json(raw_response, log)

    if not parsed:
        # Guardar themes.json mínimo para que el paso 5 no falle
        fallback = {
            "temas": [],
            "tema_principal": "(no detectado)",
            "tipo_contenido": "desconocido",
        }
        themes_path.write_text(
            json.dumps(fallback, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.warning("[Paso 4.5] Se guardó themes.json vacío como fallback.")
        sys.exit(0)

    temas = parsed.get("temas", [])
    tema_principal = parsed.get("tema_principal", "")
    tipo_contenido = parsed.get("tipo_contenido", "")

    # ── Log de resultados ─────────────────────────────────────────────────────
    log.info(f"\n[Paso 4.5] ── Temas detectados ──────────────────────────")
    log.info(f"  Tipo de contenido: {tipo_contenido}")
    log.info(f"  Tema principal:    {tema_principal}")
    log.info(f"  Temas ({len(temas)}):")
    for t in temas:
        log.info(
            f"    [{t.get('relevancia', '?')}/10] {t.get('tema', '?')} — "
            f"{t.get('descripcion', '')[:80]}"
        )
    log.info(f"──────────────────────────────────────────────────────────\n")

    # ── Confirmación interactiva (opcional) ───────────────────────────────────
    if not args.non_interactive and sys.stdin.isatty():
        try:
            resp = input("\n¿Los temas son correctos? [S/n]: ").strip().lower()
            if resp in ("n", "no"):
                print("Podés editar themes.json manualmente y re-correr desde el paso 5.")
        except (EOFError, KeyboardInterrupt):
            pass

    # ── Guardar ───────────────────────────────────────────────────────────────
    themes_path.write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info(f"[Paso 4.5] {len(temas)} temas guardados → {themes_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()