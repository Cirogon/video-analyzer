"""
Paso 5: Analiza transcripción + descripciones de frames con un LLM de texto vía Ollama.
Lee:    output/{run_id}/transcript.json
        output/{run_id}/scenes.json
        output/{run_id}/frames.json
        output/{run_id}/themes.json          (opcional)
        output/{run_id}/vector_store.json    (opcional)
        output/{run_id}/audio_features.json  (opcional)
Guarda: output/{run_id}/analysis.json

─── Contrato de datos (analysis.json) ───────────────────────────────────────
El paso 5 ahora produce un `reel_plan` con ROLES NARRATIVOS EXPLÍCITOS
para que el paso 6 pueda construir un reel con arco completo sin adivinar.

Estructura garantizada en reel_plan.clips[]:
  - role:              hook | body | climax | cta
  - start / end:       timestamps reales del transcript
  - engagement_type:   hook | emotion | education | entertainment | cta
  - virality_score:    1-10
  - razon:             por qué este momento
  - directive:
      entry_point:     frase exacta o descripción del momento de arranque
      exit_point:      frase exacta o descripción del momento de cierre
      overlay_text:    texto corto para subtítulo/overlay (máx 8 palabras)
      cut_instruction: instrucción específica de corte

top_5_clips se genera automáticamente desde reel_plan para retrocompatibilidad.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from config import Config, get_run_dir, setup_logging

SYSTEM_CONTEXT = """\
Eres un editor profesional de contenido para redes sociales con más de 10 años de experiencia.
Respondé siempre en español.
Sabés que:
- Los hooks funcionan cuando generan curiosidad, controversia o valor inmediato en los primeros 3 segundos
- Los clips educativos funcionan mejor si presentan algo sorprendente o contraintuitivo
- Los clips emocionales funcionan en el pico de la reacción (risa, asombro, realización)
- Los mejores cortes empiezan en medio de una frase cuando la energía ya está alta
- Los clips de menos de 15s funcionan mejor en TikTok; los de 30-60s en YouTube Shorts
- Un reel viral tiene estructura: HOOK (engancha) → CUERPO (desarrolla) → CLIMAX (impacta) → CTA (cierra)
- Sin hook el reel pierde el 70% de su alcance en los primeros 2 segundos
"""

PROMPT_TEMPLATE = """\
METADATA DEL VIDEO:
{metadata}

TEMAS PRINCIPALES DEL VIDEO:
{themes}

TRANSCRIPCIÓN CON TIMESTAMPS (segundos):
{transcript}

CAMBIOS DE ESCENA DETECTADOS (segundos): {scenes}

DESCRIPCIÓN VISUAL DE FRAMES CLAVE:
{frames}

MOMENTOS DE ALTA ENERGÍA AUDITIVA:
{audio_features}

Analizá el contenido ENFOCADO EN LOS TEMAS PRINCIPALES y devolvé ÚNICAMENTE un objeto JSON válido:
{{
  "video_chapters": [
    {{"start": <número>, "end": <número>, "titulo": "<título>", "tema": "<tema>"}}
  ],

  "reel_plan": {{
    "narrative_arc": "<descripción de 1 oración del arco completo del reel>",
    "clips": [
      {{
        "role":             "hook",
        "start":            <número>,
        "end":              <número>,
        "engagement_type":  "hook",
        "virality_score":   <1-10>,
        "razon":            "<por qué engancha en los primeros 3 segundos>",
        "directive": {{
          "entry_point":     "<frase textual del transcript o descripción del arranque>",
          "exit_point":      "<frase textual del transcript o descripción del cierre>",
          "overlay_text":    "<texto corto para subtítulo/overlay, máx 8 palabras>",
          "cut_instruction": "<instrucción concreta de corte>"
        }}
      }},
      {{
        "role":             "body",
        "start":            <número>,
        "end":              <número>,
        "engagement_type":  "<education|emotion|entertainment>",
        "virality_score":   <1-10>,
        "razon":            "<por qué desarrolla el tema>",
        "directive": {{
          "entry_point":     "<frase textual o descripción del arranque>",
          "exit_point":      "<frase textual o descripción del cierre>",
          "overlay_text":    "<texto corto para subtítulo/overlay>",
          "cut_instruction": "<instrucción concreta de corte>"
        }}
      }},
      {{
        "role":             "body",
        "start":            <número>,
        "end":              <número>,
        "engagement_type":  "<education|emotion|entertainment>",
        "virality_score":   <1-10>,
        "razon":            "<por qué profundiza el tema>",
        "directive": {{
          "entry_point":     "<frase textual o descripción del arranque>",
          "exit_point":      "<frase textual o descripción del cierre>",
          "overlay_text":    "<texto corto para subtítulo/overlay>",
          "cut_instruction": "<instrucción concreta de corte>"
        }}
      }},
      {{
        "role":             "climax",
        "start":            <número>,
        "end":              <número>,
        "engagement_type":  "<emotion|education|entertainment>",
        "virality_score":   <1-10>,
        "razon":            "<por qué es el pico emocional o revelación del video>",
        "directive": {{
          "entry_point":     "<frase textual o descripción del arranque>",
          "exit_point":      "<frase textual o descripción del cierre>",
          "overlay_text":    "<texto corto para subtítulo/overlay>",
          "cut_instruction": "<instrucción concreta de corte>"
        }}
      }},
      {{
        "role":             "cta",
        "start":            <número>,
        "end":              <número>,
        "engagement_type":  "cta",
        "virality_score":   <1-10>,
        "razon":            "<por qué cierra y genera acción>",
        "directive": {{
          "entry_point":     "<frase textual o descripción del arranque>",
          "exit_point":      "<última frase del CTA — terminá acá, sin relleno>",
          "overlay_text":    "<CTA en máx 6 palabras: ej. 'Guardá esto para mañana'>",
          "cut_instruction": "Cortá en la última palabra del CTA, no después del silencio."
        }}
      }}
    ]
  }},

  "hooks": ["<hook 1 impactante>", "<hook 2>", "<hook 3>"],
  "momentos_virales": [
    {{"start": <número>, "end": <número>, "descripcion": "<qué pasa>", "virality_score": <1-10>}}
  ],
  "titles_for_reels": ["<título 1>", "<título 2>", "<título 3>", "<título 4>", "<título 5>"],
  "cta_suggested": ["<cta 1 urgente>", "<cta 2>", "<cta 3>"],
  "resumen_ejecutivo": "<3-4 oraciones sobre temas y mensaje clave>",
  "coherencia_score": <1-10>,
  "potencial_compartible": <porcentaje 1-100>,
  "notas_edicion": "<consejos prácticos para el montaje final>"
}}

REGLAS CRÍTICAS — REEL_PLAN:
- DEBES generar EXACTAMENTE 5 clips: hook → body → body → climax → cta (en ese orden)
- OBLIGATORIO: exactamente 1 clip con role="hook" y engagement_type="hook"
- OBLIGATORIO: exactamente 1 clip con role="cta"  y engagement_type="cta"
- Cada clip DEBE durar entre 10 y 45 segundos. Si el momento es muy puntual, expandí hacia atrás y adelante para incluir contexto narrativo.
- Los timestamps deben existir en la transcripción proporcionada.
- directive.entry_point y exit_point deben ser frases textuales del transcript, no descripciones genéricas.
- NUNCA generes clips con timestamps idénticos o solapados entre sí.
- Distribución OBLIGATORIA:
  * hook:    del INICIO (primeros 25%) — o del momento más enganchador si hay uno mejor
  * body x2: del MEDIO (25%-75%)
  * climax:  del pico de mayor impacto emocional o revelación (puede estar en cualquier sección)
  * cta:     del FINAL (últimos 25%) — o del momento que mejor cierra el ciclo narrativo

REGLAS CRÍTICAS — RETROCOMPATIBILIDAD:
- NO incluyas top_5_clips en la respuesta (se genera automáticamente desde reel_plan)
- SÍ incluí momentos_virales — el paso 6 los usa como reserva si algún rol queda débil
"""


# ─── Búsqueda semántica ────────────────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed_query(text: str, cfg, log) -> list[float] | None:
    import requests
    base_url = cfg.ollama_url.replace("/api/generate", "")
    url = base_url.rstrip("/") + "/api/embeddings"
    try:
        res = requests.post(
            url,
            json={"model": cfg.embed_model, "prompt": text},
            timeout=30,
        )
        if res.status_code == 200:
            return res.json().get("embedding")
    except Exception as e:
        log.warning(f"[Paso 5] Error embebiendo query '{text[:40]}': {e}")
    return None


def semantic_search(
    query:        str,
    vector_store: list[dict],
    cfg,
    log,
    top_k:        int | None = None,
) -> list[dict]:
    k     = top_k or cfg.semantic_top_k
    q_emb = embed_query(query, cfg, log)
    if q_emb is None:
        return []
    scored = []
    for entry in vector_store:
        emb = entry.get("embedding")
        if not emb:
            continue
        scored.append((cosine_similarity(q_emb, emb), entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:k]]


def build_viral_queries(themes_data: list[dict]) -> list[str]:
    base = [
        "frase inicial que genera curiosidad inmediata — gancho de apertura",
        "momento de mayor tensión emocional o revelación del video",
        "conclusión o mensaje final que invita a compartir o reflexionar",
        "reacción emocional del público o del orador, risa asombro aplauso",
        "frase poderosa que resume el mensaje central del video",
        "llamada a la acción directa o pregunta retórica impactante",
        "historia personal o anécdota memorable y concreta",
        "momento sorprendente o revelador que cambia la perspectiva",
    ]
    for t in themes_data[:4]:
        nombre = t.get("tema", t.get("nombre", "")).strip()
        desc   = t.get("descripcion", "").strip()
        if nombre:
            base.append(f"insight clave o dato contraintuitivo sobre {nombre}")
        if desc:
            words = desc.split()[:6]
            if words:
                base.append(" ".join(words))
    return base


def build_semantic_transcript(
    vector_store: list[dict],
    cfg,
    log,
    themes_data:  list[dict] | None = None,
) -> str:
    VIRAL_QUERIES = build_viral_queries(themes_data or [])
    seen:     set[int]   = set()
    selected: list[dict] = []

    for query in VIRAL_QUERIES:
        for entry in semantic_search(query, vector_store, cfg, log):
            idx = entry["index"]
            if idx not in seen:
                seen.add(idx)
                selected.append(entry)

    selected.sort(key=lambda x: x["start"])

    # Garantizar cobertura del último cuarto del video
    if vector_store:
        total  = vector_store[-1]["end"]
        late   = total * 0.75
        if not any(e["start"] >= late for e in selected):
            for e in sorted(
                [e for e in vector_store if e["start"] >= late and e.get("embedding")],
                key=lambda x: x["start"],
            )[:4]:
                if e["index"] not in seen:
                    seen.add(e["index"])
                    selected.append(e)
            selected.sort(key=lambda x: x["start"])
            log.info(f"[Paso 5] Cobertura final añadida (>{late:.0f}s)")

    if not selected:
        selected = sorted(
            [e for e in vector_store if e.get("embedding")],
            key=lambda x: x["start"],
        )

    if selected and vector_store:
        log.info(
            f"[Paso 5] Semántico: {len(selected)} segs / {len(vector_store)} "
            f"(hasta {selected[-1]['end']:.0f}s de {vector_store[-1]['end']:.0f}s)"
        )

    return "\n".join(f"[{e['start']:.1f}s - {e['end']:.1f}s] {e['text']}" for e in selected)


# ─── Parsing y validación ─────────────────────────────────────────────────────

def parse_json(raw: str, log) -> dict | None:
    for attempt in [raw.strip(), re.search(r"\{[\s\S]*\}", raw), None]:
        if attempt is None:
            break
        text = attempt if isinstance(attempt, str) else attempt.group() if attempt else None
        if not text:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    log.debug(f"JSON no parseable:\n{raw[:300]}")
    return None


def find_word_boundary(transcript: list[dict], target_sec: float, mode: str = "end") -> float:
    best_time = target_sec
    best_dist = float("inf")
    for seg in transcript:
        for word in seg.get("words", []):
            if not isinstance(word, dict):
                continue
            t = word.get("end") if mode == "end" else word.get("start")
            if not isinstance(t, (int, float)):
                continue
            dist = abs(t - target_sec)
            if dist < best_dist:
                best_dist = dist
                best_time = t
    return best_time if best_dist < 2.0 else target_sec


def validate_reel_plan(parsed: dict, transcript: list[dict], log) -> dict:
    """
    Valida y normaliza el reel_plan:
    - Ajusta clips al límite de palabra más cercano
    - Expande/recorta según duración mínima/máxima
    - Fallback a momentos_virales si reel_plan está vacío
    - Genera top_5_clips legacy para retrocompatibilidad con otros pasos
    """
    MIN_CLIP = 8.0
    MAX_CLIP = 60.0

    reel_plan = parsed.get("reel_plan")
    if not isinstance(reel_plan, dict):
        reel_plan = {}
        parsed["reel_plan"] = reel_plan

    clips = reel_plan.get("clips", [])

    # Fallback: construir desde momentos_virales si el LLM no generó reel_plan
    if not clips:
        if momentos := parsed.get("momentos_virales"):
            log.warning("[Paso 5] reel_plan vacío — construyendo desde momentos_virales (fallback)")
            role_seq = ["hook", "body", "body", "climax", "cta"]
            eng_seq  = ["hook", "education", "emotion", "emotion", "cta"]
            clips = [
                {
                    "role":            role_seq[i] if i < len(role_seq) else "body",
                    "start":           m.get("start"),
                    "end":             m.get("end"),
                    "engagement_type": eng_seq[i] if i < len(eng_seq) else "emotion",
                    "virality_score":  m.get("virality_score", 6),
                    "razon":           m.get("descripcion", ""),
                    "directive":       {"entry_point": "", "exit_point": "", "overlay_text": "", "cut_instruction": ""},
                }
                for i, m in enumerate(momentos[:5])
            ]
            reel_plan["clips"]         = clips
            reel_plan["narrative_arc"] = "construido desde momentos_virales (fallback)"

    validated = []
    for clip in clips:
        start = clip.get("start")
        end   = clip.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            log.warning(f"[Paso 5] Clip descartado por timestamps inválidos: {clip.get('role', '?')}")
            continue

        clip["start"] = find_word_boundary(transcript, start, mode="start")
        clip["end"]   = find_word_boundary(transcript, end,   mode="end")

        dur = clip["end"] - clip["start"]
        if dur < MIN_CLIP:
            mid           = (clip["start"] + clip["end"]) / 2
            clip["start"] = find_word_boundary(transcript, mid - MIN_CLIP / 2, mode="start")
            clip["end"]   = find_word_boundary(transcript, mid + MIN_CLIP / 2, mode="end")
            log.info(f"[Paso 5] Clip [{clip.get('role')}] expandido de {dur:.1f}s a {clip['end']-clip['start']:.1f}s")

        dur = clip["end"] - clip["start"]
        if dur > MAX_CLIP:
            clip["end"] = find_word_boundary(transcript, clip["start"] + MAX_CLIP, mode="end")
            log.info(f"[Paso 5] Clip [{clip.get('role')}] recortado a {MAX_CLIP}s")

        if not isinstance(clip.get("directive"), dict):
            clip["directive"] = {"entry_point": "", "exit_point": "", "overlay_text": "", "cut_instruction": ""}

        validated.append(clip)

    reel_plan["clips"] = validated

    # Alertas de roles faltantes
    roles_present = {c.get("role") for c in validated}
    if "hook" not in roles_present:
        log.warning("[Paso 5] ⚠️  Sin clip HOOK — el paso 6 buscará en momentos_virales")
    if "cta" not in roles_present:
        log.warning("[Paso 5] ⚠️  Sin clip CTA  — el paso 6 buscará en cta_suggested")

    # Generar top_5_clips legacy para otros pasos del pipeline
    parsed["top_5_clips"] = [
        {
            "start":           c["start"],
            "end":             c["end"],
            "razon":           c.get("razon", ""),
            "virality_score":  c.get("virality_score", 6),
            "engagement_type": c.get("engagement_type", "other"),
        }
        for c in validated
    ]

    return parsed


# ─── Audio features formateadas ──────────────────────────────────────────────

def format_audio_features(audio_features: dict, log) -> str:
    import math
    peaks       = audio_features.get("energy_peaks", [])
    silent_segs = audio_features.get("silent_segments", [])
    duration    = audio_features.get("duration_sec", 0)

    bucket_sec = 30
    buckets: dict[int, list[float]] = {}
    for p in peaks:
        b = int(p["time"] // bucket_sec)
        buckets.setdefault(b, []).append(p["energy"])

    bucket_scores: dict[int, dict] = {}
    for b, energies in buckets.items():
        avg   = sum(energies) / len(energies)
        score = avg * math.log(len(energies) + 1)
        bucket_scores[b] = {"avg": avg, "count": len(energies), "score": score}

    if bucket_scores:
        max_s = max(v["score"] for v in bucket_scores.values())
        for v in bucket_scores.values():
            v["norm"] = round(v["score"] / max_s * 100) if max_s > 0 else 0

    top_zones = sorted(bucket_scores.items(), key=lambda x: x[1]["score"], reverse=True)

    def label(norm: int) -> str:
        if norm >= 80: return "🔥 MUY ALTA"
        if norm >= 60: return "⚡ ALTA"
        if norm >= 40: return "✅ MEDIA"
        return               "〰 BAJA"

    long_pauses = sorted(
        [s for s in silent_segs if s["end"] - s["start"] >= 1.5],
        key=lambda s: s["end"] - s["start"], reverse=True,
    )

    lines = [
        f"Duración: {duration:.0f}s  |  Picos de energía: {len(peaks)}  |  Silencios largos: {len(long_pauses)}",
        "", "ZONAS DE ENERGÍA (30s por bloque):",
    ]
    for b, info in top_zones[:12]:
        ts = b * bucket_sec
        te = ts + bucket_sec
        lines.append(
            f"  • {ts//60:02.0f}:{ts%60:02.0f}–{te//60:02.0f}:{te%60:02.0f}"
            f"  score={info['norm']:3d}/100  {label(info['norm'])}"
            f"  (avg={info['avg']:.3f}, n={info['count']})"
        )

    if long_pauses:
        lines += ["", "PAUSAS DRAMÁTICAS (buenos puntos de corte):"]
        for p in long_pauses[:5]:
            lines.append(f"  • {p['start']:.1f}s  ({p['end']-p['start']:.1f}s de silencio)")

    lines += [
        "",
        "INSTRUCCIÓN: Asigná hook y climax a zonas score≥60. "
        "Las pausas dramáticas son buenos puntos de entrada/salida de clip.",
    ]

    if top_zones:
        log.info(
            f"[Paso 5] Audio: {len(bucket_scores)} zonas de 30s, "
            f"top: {top_zones[0][0]*bucket_sec:.0f}s (score {top_zones[0][1]['norm']}/100)"
        )

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()

    cfg     = Config()
    out_dir = get_run_dir(cfg.output_dir, args.run_id)
    log     = setup_logging(out_dir)

    transcript_path     = out_dir / "transcript.json"
    scenes_path         = out_dir / "scenes.json"
    frames_path         = out_dir / "frames.json"
    themes_path         = out_dir / "themes.json"
    analysis_path       = out_dir / "analysis.json"
    vector_store_path   = out_dir / "vector_store.json"
    audio_features_path = out_dir / "audio_features.json"

    for p in [transcript_path, scenes_path]:
        if not p.exists():
            log.error(f"Falta {p}. ¿Corriste todos los pasos anteriores?")
            sys.exit(1)

    transcript: list[dict]  = json.loads(transcript_path.read_text("utf-8"))
    _scenes_raw = json.loads(scenes_path.read_text("utf-8"))
    # Soportar formato nuevo {timestamps, video_meta} y legacy (lista plana)
    if isinstance(_scenes_raw, dict):
        scenes: list[float] = _scenes_raw.get("timestamps", [])
    else:
        scenes: list[float] = _scenes_raw
    frames: list[dict] = []
    if frames_path.exists():
        try:
            frames = json.loads(frames_path.read_text("utf-8"))
        except Exception:
            log.warning("[Paso 5] No se pudo leer frames.json — continuando sin visión")

    vector_store: list[dict] = []
    if vector_store_path.exists():
        try:
            vector_store = json.loads(vector_store_path.read_text("utf-8"))
            log.info(f"[Paso 5] Vector store: {len(vector_store)} segmentos")
        except Exception:
            log.warning("[Paso 5] No se pudo leer el vector store")

    audio_features = None
    if audio_features_path.exists():
        try:
            audio_features = json.loads(audio_features_path.read_text("utf-8"))
        except Exception:
            pass

    themes_data_raw: list[dict] = []
    themes_fmt = "No se han definido temas específicos. Detectá los temas principales."
    if themes_path.exists():
        try:
            themes_raw = json.loads(themes_path.read_text("utf-8"))
            # Soporta formato nuevo (dict con clave "temas") y formato legacy (lista directa)
            if isinstance(themes_raw, dict):
                themes_data_raw = themes_raw.get("temas", [])
                tema_principal  = themes_raw.get("tema_principal", "")
                tipo_contenido  = themes_raw.get("tipo_contenido", "")
                header = ""
                if tema_principal:
                    header = f"Tema principal: {tema_principal}\nTipo: {tipo_contenido}\n\n"
                themes_fmt = header + "\n".join(
                    f"• {t.get('tema', t.get('nombre', ''))}: {t.get('descripcion', '')}"
                    for t in themes_data_raw
                )
            else:
                # Formato legacy: lista directa
                themes_data_raw = themes_raw
                themes_fmt = "\n".join(
                    f"• {t.get('tema', t.get('nombre', ''))}: {t.get('descripcion', '')}"
                    for t in themes_data_raw
                )
        except Exception:
            pass

    # Transcript
    if vector_store:
        log.info("[Paso 5] Usando búsqueda semántica...")
        transcript_fmt = build_semantic_transcript(vector_store, cfg, log, themes_data=themes_data_raw)
    else:
        log.info("[Paso 5] Sin vector store, muestreo distribuido...")
        transcript_fmt = "\n".join(
            f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['text']}" for s in transcript
        )
        MAX_LEN = 40000
        if len(transcript_fmt) > MAX_LEN:
            lines  = transcript_fmt.split("\n")
            step   = max(1, len(lines) // int(MAX_LEN / (len(transcript_fmt) / len(lines))))
            sampled = lines[::step]
            transcript_fmt = "\n".join(sampled)
            log.info(f"[Paso 5] Transcript submuestreado: {len(lines)} → {len(sampled)} líneas")

    if frames:
        frames_fmt = "\n".join(f"[{f['time']:.1f}s] {f['description']}" for f in frames)
    else:
        frames_fmt = "(sin descripciones visuales de frames)"

    total_duration = transcript[-1]["end"] if transcript else 0
    word_count     = sum(len(s.get("text", "").split()) for s in transcript)
    quarter        = total_duration / 4

    metadata_block = (
        f"- Duración total: {total_duration:.0f}s ({total_duration/60:.1f} min)\n"
        f"- INICIO (0-{quarter:.0f}s): zona preferida para el HOOK\n"
        f"- MEDIO ({quarter:.0f}-{3*quarter:.0f}s): zona preferida para clips BODY\n"
        f"- FINAL ({3*quarter:.0f}-{total_duration:.0f}s): zona preferida para el CTA\n"
        f"- Segmentos: {len(transcript)}  |  Palabras: ~{word_count}\n"
        f"- Escenas: {len(scenes)}  |  Frames: {len(frames)}"
    )

    audio_features_fmt = (
        format_audio_features(audio_features, log)
        if audio_features
        else "No se detectaron características de audio adicionales."
    )

    prompt = PROMPT_TEMPLATE.format(
        metadata       = metadata_block,
        themes         = themes_fmt,
        transcript     = transcript_fmt,
        scenes         = scenes,
        frames         = frames_fmt,
        audio_features = audio_features_fmt,
    )

    log.info(f"[Paso 5] Enviando al LLM ({cfg.ollama_model})...")

    import requests

    for attempt in range(1, cfg.ollama_retries + 1):
        try:
            t0  = time.time()
            res = requests.post(
                cfg.ollama_url,
                json={
                    "model":  cfg.ollama_model,
                    "system": SYSTEM_CONTEXT,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.2, "num_ctx": 16384, "repeat_penalty": 1.1},
                },
                timeout=cfg.ollama_timeout,
            )
            elapsed = time.time() - t0

            if res.status_code != 200:
                raise ValueError(f"HTTP {res.status_code}: {res.text[:200]}")

            raw    = res.json().get("response", "")
            log.info(f"[Paso 5] LLM respondió en {elapsed:.1f}s (intento {attempt})")

            parsed = parse_json(raw, log)
            if not parsed:
                log.warning(f"[Paso 5] JSON inválido (intento {attempt})")
                if attempt < cfg.ollama_retries:
                    time.sleep(2)
                continue

            parsed = validate_reel_plan(parsed, transcript, log)

            analysis_path.write_text(
                json.dumps(parsed, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            log.info(f"[Paso 5] Análisis guardado → {analysis_path}")

            # ── Resumen en consola ────────────────────────────────────────────
            sep = "─" * 60
            log.info(f"\n{sep}\nANÁLISIS — REEL PLAN\n{sep}")

            if r := parsed.get("resumen_ejecutivo"):
                log.info(f"\n📋 RESUMEN:\n   {r}")

            if chapters := parsed.get("video_chapters"):
                log.info("\n📖 CAPÍTULOS:")
                for ch in chapters:
                    log.info(f"   [{ch['start']:.0f}s-{ch['end']:.0f}s] {ch.get('titulo','')} ({ch.get('tema','')})")

            rp    = parsed.get("reel_plan", {})
            clips = rp.get("clips", [])
            if clips:
                log.info(f"\n🎬 REEL PLAN — {rp.get('narrative_arc', '')}")
                role_emoji = {"hook": "🎣", "body": "📚", "climax": "🔥", "cta": "📣"}
                for i, c in enumerate(clips, 1):
                    role = c.get("role", "?")
                    log.info(
                        f"   {i}. {role_emoji.get(role,'•')} [{role.upper()}] "
                        f"[{c['start']:.0f}s→{c['end']:.0f}s] "
                        f"⭐{c.get('virality_score',0)}/10 ({c.get('engagement_type','')})"
                    )
                    log.info(f"      {c.get('razon','')}")
                    d = c.get("directive", {})
                    if d.get("overlay_text"):
                        log.info(f"      overlay: \"{d['overlay_text']}\"")
                    if d.get("cut_instruction"):
                        log.info(f"      corte: {d['cut_instruction']}")

            roles_p = {c.get("role") for c in clips}
            if "hook" not in roles_p:
                log.warning("⚠️  Sin HOOK en reel_plan")
            if "cta" not in roles_p:
                log.warning("⚠️  Sin CTA en reel_plan")

            if hooks := parsed.get("hooks"):
                log.info("\n🎣 HOOKS SUGERIDOS:")
                for h in hooks:   log.info(f"   • {h}")
            if ctas := parsed.get("cta_suggested"):
                log.info("\n📣 CTAs:")
                for c in ctas:    log.info(f"   • {c}")
            if titles := parsed.get("titles_for_reels"):
                log.info("\n📝 TÍTULOS:")
                for t in titles:  log.info(f"   • {t}")

            log.info(
                f"\n✨ Coherencia: {parsed.get('coherencia_score','?')}/10  |  "
                f"Potencial: {parsed.get('potencial_compartible','?')}%"
            )
            if notas := parsed.get("notas_edicion"):
                log.info(f"\n💡 NOTAS:\n   {notas}")

            log.info(f"\n📁 Output: {out_dir}\n{sep}")
            sys.exit(0)

        except requests.exceptions.ConnectionError:
            log.error("No se pudo conectar a Ollama. ¿Está corriendo en localhost:11434?")
            sys.exit(1)
        except requests.exceptions.Timeout:
            log.warning(f"[Paso 5] Timeout (intento {attempt}/{cfg.ollama_retries})")
        except Exception as e:
            log.warning(f"[Paso 5] Error (intento {attempt}): {e}")

        if attempt < cfg.ollama_retries:
            time.sleep(2)

    log.error("[Paso 5] No se pudo obtener respuesta válida del LLM.")
    sys.exit(1)


if __name__ == "__main__":
    main()
