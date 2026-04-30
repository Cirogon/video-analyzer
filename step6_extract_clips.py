"""
Paso 6: Extrae clips + calcula Virality Score final.
       (Fusiona step6 + step9 — step9_virality_score.py queda deprecado)

Lee:    output/{run_id}/analysis.json        → reel_plan (nuevo) o top_5_clips (legacy)
        output/{run_id}/transcript.json
        output/{run_id}/audio_features.json  (opcional)
        video_path de Config

Guarda: output/{run_id}/clips/clip_N.mp4
        output/{run_id}/clips/manifest.json  → clips con score_breakdown
        output/{run_id}/virality_report.json → ranking + desglose completo

─── Virality Score (4 señales) ──────────────────────────────────────────────
  1. LLM score       (reel_plan.virality_score, escala 1-10)   peso 40%
  2. Audio energy    (audio_features.energy_peaks)              peso 25%
  3. Engagement type (hook > emotion > cta > education > other) peso 20%
  4. Shot dynamics   (manifest shot_plan si existe)             peso 15%

  IMPORTANTE — doble normalización prevenida:
  El score 1-10 del LLM se guarda en _llm_raw en el manifest.
  Si se re-corre el paso 6 sobre un manifest ya procesado,
  _llm_raw evita que el score compuesto (0-100) sea normalizado
  de nuevo como si fuera escala 1-10.

─── Resolución de arco narrativo ────────────────────────────────────────────
El paso 6 lee el reel_plan del paso 5 y RESUELVE activamente el arco:
  - Si falta hook  → busca en momentos_virales el de mayor score
  - Si falta cta   → busca en momentos_virales o cta_suggested
  - Aplica directive.entry_point / exit_point para ajustar start/end del clip
  - Ordena la extracción según el rol narrativo (hook primero, cta último)
"""

import argparse
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from config import Config, get_run_dir, setup_logging, get_media_duration


# ─── Pesos del Virality Score ─────────────────────────────────────────────────

W_LLM        = 0.40
W_AUDIO      = 0.25
W_ENGAGEMENT = 0.20
W_SHOTS      = 0.15

ENGAGEMENT_SCORES = {
    "hook":          1.00,
    "emotion":       0.85,
    "cta":           0.80,
    "education":     0.65,
    "entertainment": 0.55,
    "other":         0.45,
}

DYNAMIC_MOVEMENTS = {"zoom_peak": 1.0, "push_in": 0.75, "pull_out": 0.5}

ROLE_ORDER = {"hook": 0, "body": 1, "climax": 3, "cta": 4, "other": 2}


# ─── Virality Score — señales ─────────────────────────────────────────────────

def score_llm(clip_data: dict, analysis: dict) -> float:
    """Señal 1: score del LLM normalizado 0-1.

    Distingue dos fuentes:
      - reel_plan / top_5_clips: escala 1-10  → normalizar con (x-1)/9
      - manifest ya procesado:   escala 0-100 → el campo se llama _llm_raw
        para evitar confusión con el score compuesto final

    Para no re-procesar el campo virality_score compuesto (0-100) como si
    fuera 1-10 (lo que inflaba el score al re-correr el paso 9), buscamos
    siempre en el analysis.json como fuente de verdad.
    """
    # Si el clip trae _llm_raw ya guardado (re-run), usarlo directamente
    if "_llm_raw" in clip_data:
        return max(0.0, min(1.0, (clip_data["_llm_raw"] - 1) / 9))

    start = clip_data.get("start_original") or clip_data.get("start_padded") or clip_data.get("start") or 0
    end   = clip_data.get("end_original")   or clip_data.get("end_padded")   or clip_data.get("end")   or 0

    best_score   = None
    best_overlap = 0.0

    # Buscar en reel_plan primero, luego en top_5_clips, luego en momentos_virales
    candidates = (
        analysis.get("reel_plan", {}).get("clips", [])
        + analysis.get("top_5_clips", [])
        + analysis.get("momentos_virales", [])
    )
    for c in candidates:
        cs = c.get("start", 0)
        ce = c.get("end",   0)
        overlap = max(0.0, min(end, ce) - max(start, cs))
        dur     = end - start
        ratio   = overlap / dur if dur > 0 else 0.0
        if ratio > best_overlap:
            best_overlap = ratio
            best_score   = c.get("virality_score", 5)

    score = best_score if best_score is not None else 5
    return max(0.0, min(1.0, (score - 1) / 9))


def score_audio(clip_data: dict, audio_features: dict) -> float:
    """Señal 2: energía de audio del clip relativa al video completo, 0-1."""
    start = clip_data.get("start_padded", clip_data.get("start_original", clip_data.get("start", 0)))
    end   = clip_data.get("end_padded",   clip_data.get("end_original",   clip_data.get("end",   0)))

    peaks = audio_features.get("energy_peaks", [])
    if not peaks:
        return 0.5

    global_mean = audio_features.get("rms_mean", 0.037)
    global_std  = audio_features.get("rms_std",  0.045)

    clip_peaks = [p["energy"] for p in peaks if start <= p["time"] <= end]
    if not clip_peaks:
        return 0.5

    clip_mean = sum(clip_peaks) / len(clip_peaks)
    clip_max  = max(clip_peaks)

    # Penalizar silencios largos dentro del clip
    silent_segs = audio_features.get("silent_segments", [])
    silence_in  = sum(
        min(end, s["end"]) - max(start, s["start"])
        for s in silent_segs
        if s["start"] < end and s["end"] > start
    )
    dur              = end - start
    silence_ratio    = silence_in / dur if dur > 0 else 0
    silence_penalty  = max(0.0, 1.0 - silence_ratio * 1.5)

    z          = (clip_mean - global_mean) / global_std if global_std > 0 else 0.0
    normalized = max(0.0, min(1.0, (z + 2) / 4))
    peak_bonus = min(0.15, (clip_max - global_mean) / (global_mean + 1e-6) * 0.05)

    return min(1.0, normalized * silence_penalty + peak_bonus)


def score_engagement(clip_data: dict, analysis: dict) -> float:
    """Señal 3: tipo de engagement, 0-1."""
    if "engagement_type" in clip_data:
        return ENGAGEMENT_SCORES.get(clip_data["engagement_type"], 0.45)

    start = clip_data.get("start_original") or clip_data.get("start_padded") or clip_data.get("start") or 0
    end   = clip_data.get("end_original")   or clip_data.get("end_padded")   or clip_data.get("end")   or 0
    best_type    = "other"
    best_overlap = 0.0

    for c in (analysis.get("reel_plan", {}).get("clips", []) + analysis.get("top_5_clips", [])):
        cs      = c.get("start", 0)
        ce      = c.get("end",   0)
        overlap = max(0.0, min(end, ce) - max(start, cs))
        dur     = end - start
        if dur > 0 and overlap / dur > best_overlap:
            best_overlap = overlap / dur
            best_type    = c.get("engagement_type", "other")

    return ENGAGEMENT_SCORES.get(best_type, 0.45)


def score_shots(clip_data: dict) -> float:
    """Señal 4: dinamismo cinematográfico del shot_plan, 0-1."""
    shot_plan = clip_data.get("shot_plan", [])
    if not shot_plan:
        return 0.5

    total     = len(shot_plan)
    dyn_score = 0.0
    close_ups = 0
    has_peak  = False

    for shot in shot_plan:
        dyn_score += DYNAMIC_MOVEMENTS.get(shot.get("movement", "static"), 0.0)
        if shot.get("shot_type") == "close_up":
            close_ups += 1
        if shot.get("zoom", 1.0) >= 1.25:
            dyn_score += 0.2
        if shot.get("movement") == "zoom_peak":
            has_peak = True

    movement_score = min(1.0, dyn_score / total)
    closeup_ratio  = close_ups / total
    peak_bonus     = 0.15 if has_peak else 0.0

    return min(1.0, movement_score * 0.6 + closeup_ratio * 0.25 + peak_bonus)


def compute_virality_score(clip_data: dict, analysis: dict, audio_features: dict) -> dict:
    """Combina las 4 señales y devuelve score 0-100 + desglose."""
    s_llm        = score_llm(clip_data, analysis)
    s_audio      = score_audio(clip_data, audio_features)
    s_engagement = score_engagement(clip_data, analysis)
    s_shots      = score_shots(clip_data)

    weighted = (
        s_llm        * W_LLM
        + s_audio      * W_AUDIO
        + s_engagement * W_ENGAGEMENT
        + s_shots      * W_SHOTS
    )

    return {
        "virality_score":  round(weighted * 100),
        "score_breakdown": {
            "llm":        round(s_llm        * 100),
            "audio":      round(s_audio      * 100),
            "engagement": round(s_engagement * 100),
            "shots":      round(s_shots      * 100),
        },
    }


def virality_label(score: int) -> str:
    if score >= 80: return "🔥 Viral"
    if score >= 65: return "⚡ Alto"
    if score >= 50: return "✅ Medio"
    if score >= 35: return "⚠️  Bajo"
    return               "❌ Muy bajo"


# ─── Resolución del arco narrativo ───────────────────────────────────────────

def resolve_narrative_arc(analysis: dict, log) -> list[dict]:
    """
    Lee el reel_plan del paso 5 y garantiza que el arco narrativo esté completo.

    - Si existe reel_plan.clips, lo usa como base.
    - Detecta roles faltantes (hook, cta).
    - Busca candidatos en momentos_virales o cta_suggested para completar.
    - Devuelve la lista de clips normalizada con role + engagement_type.
    """
    reel_plan = analysis.get("reel_plan", {})
    clips     = list(reel_plan.get("clips", []))

    # Fallback a top_5_clips si no hay reel_plan (análisis de versión anterior)
    if not clips:
        log.info("[Paso 6] Sin reel_plan — usando top_5_clips (análisis legacy)")
        clips = [
            {**c, "role": "body", "directive": {}}
            for c in analysis.get("top_5_clips", [])
        ]
        if clips:
            clips[0]["role"]            = "hook"
            clips[0]["engagement_type"] = "hook"
            clips[-1]["role"]           = "cta"
            clips[-1]["engagement_type"] = "cta"

    if not clips:
        log.error("[Paso 6] Sin clips en reel_plan ni en top_5_clips.")
        return []

    roles_present = {c.get("role") for c in clips}

    # ── Completar HOOK si falta ───────────────────────────────────────────────
    if "hook" not in roles_present:
        log.warning("[Paso 6] Sin clip HOOK — buscando candidato en momentos_virales...")
        best = max(
            analysis.get("momentos_virales", []),
            key=lambda m: m.get("virality_score", 0),
            default=None,
        )
        if best:
            hook_clip = {
                "role":            "hook",
                "start":           best["start"],
                "end":             best["end"],
                "engagement_type": "hook",
                "virality_score":  best.get("virality_score", 6),
                "razon":           best.get("descripcion", "Mejor momento viral como hook"),
                "directive":       {"entry_point": "", "exit_point": "", "overlay_text": "", "cut_instruction": ""},
            }
            clips.insert(0, hook_clip)
            log.info(f"[Paso 6] Hook añadido desde momentos_virales: [{best['start']:.0f}s→{best['end']:.0f}s]")
        else:
            # Promover el clip de mayor score a hook
            best_idx = max(range(len(clips)), key=lambda i: clips[i].get("virality_score", 0))
            clips[best_idx]["role"]            = "hook"
            clips[best_idx]["engagement_type"] = "hook"
            log.info(f"[Paso 6] Clip {best_idx+1} promovido a HOOK (mayor virality_score)")

    # ── Completar CTA si falta ────────────────────────────────────────────────
    if "cta" not in {c.get("role") for c in clips}:
        log.warning("[Paso 6] Sin clip CTA — buscando candidato...")
        # Buscar en momentos_virales en el último 25% del video
        all_ends = [c.get("end", 0) for c in clips]
        total    = max(all_ends) if all_ends else 0
        late_candidates = [
            m for m in analysis.get("momentos_virales", [])
            if m.get("start", 0) >= total * 0.60   # zona amplia del final
        ]
        if late_candidates:
            best_cta = max(late_candidates, key=lambda m: m.get("virality_score", 0))
            cta_clip = {
                "role":            "cta",
                "start":           best_cta["start"],
                "end":             best_cta["end"],
                "engagement_type": "cta",
                "virality_score":  best_cta.get("virality_score", 6),
                "razon":           best_cta.get("descripcion", "Momento de cierre con llamada a la acción"),
                "directive":       {
                    "entry_point":     "",
                    "exit_point":      "",
                    "overlay_text":    analysis.get("cta_suggested", [""])[0],
                    "cut_instruction": "Cortá en la última palabra, sin relleno posterior.",
                },
            }
            clips.append(cta_clip)
            log.info(f"[Paso 6] CTA añadido desde momentos_virales: [{best_cta['start']:.0f}s→{best_cta['end']:.0f}s]")
        else:
            # Promover el último clip a CTA
            clips[-1]["role"]            = "cta"
            clips[-1]["engagement_type"] = "cta"
            log.info("[Paso 6] Último clip promovido a CTA (sin candidatos en zona final)")

    # ── Ordenar por rol narrativo ─────────────────────────────────────────────
    clips.sort(key=lambda c: (ROLE_ORDER.get(c.get("role", "other"), 2), -c.get("virality_score", 0)))

    log.info(f"[Paso 6] Arco narrativo resuelto: {' → '.join(c.get('role','?').upper() for c in clips)}")
    return clips


# ─── Detección de barras negras ───────────────────────────────────────────────

def detect_active_area(video_path: str, log) -> tuple[int, int, int, int] | None:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-t", "60", "-i", video_path,
             "-vf", "cropdetect=limit=24:round=2:skip=2", "-f", "null", "-"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=90,
        )
        matches = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", result.stderr)
        if not matches:
            log.info("  cropdetect: sin detecciones")
            return None

        most_common, count = Counter(matches).most_common(1)[0]
        w, h, x, y = (int(v) for v in most_common)

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", video_path],
            capture_output=True, text=True, timeout=15,
        )
        orig_w, orig_h = (int(v) for v in probe.stdout.strip().split("x"))

        if w >= orig_w * 0.98 and h >= orig_h * 0.98:
            log.info(f"  cropdetect: sin barras significativas ({w}×{h} ≈ {orig_w}×{orig_h})")
            return None

        log.info(
            f"  cropdetect: área activa {w}×{h} en ({x},{y})"
            f" — original {orig_w}×{orig_h} — moda en {count}/{len(matches)} frames"
        )
        return w, h, x, y

    except Exception as e:
        log.warning(f"  cropdetect falló: {e}")
        return None


def _safe_unlink(path: Path, retries: int = 5, delay: float = 0.3) -> None:
    """
    Elimina un archivo con reintentos para tolerar el bloqueo temporal de
    handles en Windows (WinError 32) que ocurre cuando ffmpeg cierra el
    proceso pero el SO todavía no liberó el handle del archivo de salida.
    """
    for attempt in range(retries):
        try:
            if path.exists():
                path.unlink()
            return
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                # No bloquear el pipeline por no poder borrar un archivo temporal
                pass



# ─── Extracción de clip ───────────────────────────────────────────────────────

def extract_clip(
    video_path:  str,
    start_sec:   float,
    end_sec:     float,
    output_path: Path,
    log,
    active_area: tuple[int, int, int, int] | None = None,
) -> bool:
    duration = end_sec - start_sec
    vf_args  = ["-vf", f"crop={active_area[0]}:{active_area[1]}:{active_area[2]}:{active_area[3]}"] if active_area else []

    encoders = [
        ["-c:v", "h264_nvenc", "-preset", "p4",   "-cq",  "18"],
        ["-c:v", "libx264",    "-preset", "fast",  "-crf", "18"],
    ]

    for video_args in encoders:
        cmd = [
            "ffmpeg", "-y",
            "-i",  video_path,
            "-ss", str(start_sec),
            "-t",  str(duration),
            "-map", "0:v", "-map", "0:a?",
            *vf_args,
            *video_args,
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
            "-vsync", "cfr", "-r", "30",
            "-video_track_timescale", "90000",
            "-avoid_negative_ts", "make_zero",
            str(output_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            log.warning(f"  Timeout con {video_args[1]}")
            if output_path.exists():
                _safe_unlink(output_path)
            continue
        except Exception as e:
            log.error(f"  Error: {e}")
            return False

        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            log.info(f"    Encoder: {video_args[1]}{' + crop' if active_area else ''}")
            return True

        log.warning(f"  ffmpeg ({video_args[1]}) error:\n{result.stderr[-300:]}")
        if output_path.exists():
            _safe_unlink(output_path)

    return False


# ─── Helpers ──────────────────────────────────────────────────────────────────

def extract_transcript_text(transcript: list[dict], start: float, end: float, max_chars: int = 300) -> str:
    segs = [s for s in transcript if s.get("end", 0) >= start and s.get("start", 0) <= end]
    text = " ".join(s.get("text", "").strip() for s in segs).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text or "(sin transcript en este rango)"


def build_clip_script(clip_data: dict, transcript: list[dict] | None = None) -> str:
    """Genera el guion editorial del clip usando datos del reel_plan."""
    razon      = clip_data.get("razon", "").strip()
    engagement = clip_data.get("engagement_type", "").lower()
    role       = clip_data.get("role", "body")
    start      = clip_data.get("start")
    end        = clip_data.get("end")
    score      = clip_data.get("virality_score", "?")
    directive  = clip_data.get("directive", {})

    lines = []

    if transcript and start is not None and end is not None:
        lines.append(f"TEXTO DEL CLIP: \"{extract_transcript_text(transcript, start, end)}\"")

    if razon:
        lines.append(f"ROL: {role.upper()}  |  SCORE: {score}/10  |  TIPO: {engagement}")
        lines.append(f"POR QUÉ: {razon}")

    if directive.get("entry_point"):
        lines.append(f"ENTRADA: {directive['entry_point']}")
    if directive.get("exit_point"):
        lines.append(f"SALIDA:  {directive['exit_point']}")
    if directive.get("overlay_text"):
        lines.append(f"OVERLAY: \"{directive['overlay_text']}\"")

    tip_map = {
        "hook":          "Arrancá en el pico de energía. El primer segundo debe enganchar sin contexto previo.",
        "education":     "Preservá la explicación completa. No cortes antes del insight principal.",
        "emotion":       "Dejá respirar la emoción. No cortes inmediatamente después del momento cumbre.",
        "cta":           "La última palabra es el CTA. Terminá con impacto, sin relleno posterior.",
        "entertainment": "Mantené el ritmo alto. Cortá antes de que baje la energía.",
    }
    if directive.get("cut_instruction"):
        lines.append(f"CORTE: {directive['cut_instruction']}")
    else:
        lines.append(f"CORTE: {tip_map.get(engagement, tip_map.get(role, 'Asegurate de que el mensaje sea claro desde el primer segundo.'))}")

    return "\n".join(lines)


def pre_filter_clips(
    clips_data:          list[dict],
    analysis:            dict,
    audio_features:      dict,
    log,
    min_score:           int = 50,
) -> list[dict]:
    """
    Filtra candidatos antes de cortarlos usando Virality Score (señales LLM + audio + engagement + shots).
    Siempre preserva el clip de rol 'hook' y 'cta' aunque no pasen el umbral,
    para no romper el arco narrativo.
    """
    scored = []
    for clip in clips_data:
        result = compute_virality_score(clip, analysis, audio_features)
        scored.append({**clip, "_pre_score": result["virality_score"]})

    log.info(f"[Paso 6] Pre-filtro de calidad (umbral: {min_score}/100):")
    for s in sorted(scored, key=lambda x: x["_pre_score"], reverse=True):
        emoji = "✅" if s["_pre_score"] >= min_score else ("🔒" if s.get("role") in ("hook", "cta") else "❌")
        log.info(
            f"  {emoji} {s['_pre_score']:3d}/100  [{s.get('start',0):.0f}s–{s.get('end',0):.0f}s]"
            f"  {s.get('role','?'):<8}  {s.get('engagement_type','?'):<12}  {s.get('razon','')[:50]}"
        )

    # Los roles hook y cta son obligatorios aunque no pasen el umbral
    filtered = [
        s for s in scored
        if s["_pre_score"] >= min_score or s.get("role") in ("hook", "cta")
    ]

    for s in filtered:
        del s["_pre_score"]

    n_out = len(clips_data) - len(filtered)
    if n_out > 0:
        log.info(f"[Paso 6] Descartados {n_out} clip(s) (hook/cta siempre preservados)")
    else:
        log.info(f"[Paso 6] Todos los clips pasan el umbral ({len(filtered)}/{len(clips_data)})")

    return filtered


def build_assembly_notes(clips: list[dict], transcript: list[dict] | None = None) -> dict:
    """Analiza coherencia narrativa del conjunto de clips."""
    engagement_types = [c.get("engagement_type", "other") for c in clips]
    roles            = [c.get("role", "other") for c in clips]
    scores           = [c.get("virality_score", 5) for c in clips]
    reasons          = [c.get("razon", "") for c in clips]

    has_hook      = any(r == "hook"   for r in roles)
    has_cta       = any(r == "cta"    for r in roles)
    has_climax    = any(r == "climax" for r in roles)
    has_emotion   = any(e == "emotion"   for e in engagement_types)
    has_education = any(e == "education" for e in engagement_types)

    warnings = []
    tips     = []

    if not has_hook:
        warnings.append("Sin clip HOOK — el reel puede perder audiencia en los primeros 2 segundos.")
    if not has_cta:
        warnings.append("Sin clip CTA — el reel no tiene cierre con llamada a la acción.")

    # Temas repetidos
    seen_kw: dict[str, int] = {}
    for r in reasons:
        for kw in ["respeto", "violencia", "bullying", "motivación", "responsabilidad", "acción"]:
            if kw in r.lower():
                seen_kw[kw] = seen_kw.get(kw, 0) + 1
    repeated = [k for k, v in seen_kw.items() if v >= 3]
    if repeated:
        warnings.append(f"Tema repetido en 3+ clips: {', '.join(repeated)}.")

    avg_score = sum(scores) / len(scores) if scores else 0
    if avg_score < 6:
        warnings.append(f"Score promedio bajo ({avg_score:.1f}/10). Revisá los umbrales del paso 5.")

    # Distribución temporal
    starts = [c.get("start", 0) for c in clips]
    if starts and max(starts) > 0:
        ms = max(starts)
        early = sum(1 for s in starts if s < ms * 0.33)
        late  = sum(1 for s in starts if s >= ms * 0.66)
        if early == len(clips):
            warnings.append("Todos los clips están en el primer tercio — falta el final.")
        elif late == len(clips):
            warnings.append("Todos los clips están en el último tercio — falta el contexto inicial.")

    if has_hook and has_emotion and has_climax:
        tips.append("Arco completo: hook → cuerpo → clímax → CTA. Estructura óptima para redes.")
    elif has_emotion and not has_hook:
        tips.append("Usá el clip emocional como apertura para compensar la falta de hook.")
    if has_education:
        tips.append("Los clips educativos funcionan mejor en el centro del reel.")

    # Orden narrativo sugerido
    order_priority = {"hook": 0, "body": 1, "climax": 2, "cta": 3, "other": 4}
    suggested_order = sorted(
        range(len(clips)),
        key=lambda i: (order_priority.get(roles[i], 4), -scores[i]),
    )
    suggested_filenames = [clips[i].get("filename", f"clip_{i+1}.mp4") for i in suggested_order]

    arc = (
        "completo" if (has_hook and has_climax and has_cta) else
        "parcial"  if sum([has_hook, has_emotion, has_education, has_cta]) >= 2 else
        "débil"
    )

    return {
        "suggested_order":    suggested_filenames,
        "has_hook":           has_hook,
        "has_cta":            has_cta,
        "has_climax":         has_climax,
        "has_emotion":        has_emotion,
        "has_education":      has_education,
        "avg_virality_score": round(avg_score, 1),
        "warnings":           warnings,
        "tips":               tips,
        "narrative_arc":      arc,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id",       required=True)
    parser.add_argument(
        "--min_virality", type=int, default=50,
        help="Score mínimo 0-100 para extraer un clip (default: 50). Usar 0 para desactivar. Hook/CTA siempre se preservan.",
    )
    args = parser.parse_args()

    cfg      = Config()
    out_dir  = get_run_dir(cfg.output_dir, args.run_id)
    log      = setup_logging(out_dir)

    analysis_path       = out_dir / "analysis.json"
    transcript_path     = out_dir / "transcript.json"
    audio_features_path = out_dir / "audio_features.json"
    clips_dir           = out_dir / "clips"
    manifest_path       = clips_dir / "manifest.json"
    report_path         = out_dir / "virality_report.json"

    if not analysis_path.exists():
        log.error(f"No se encontró {analysis_path}. ¿Corriste los pasos anteriores?")
        sys.exit(1)

    try:
        analysis: dict = json.loads(analysis_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.error("Error leyendo analysis.json")
        sys.exit(1)

    transcript: list[dict] = []
    if transcript_path.exists():
        try:
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
            log.info(f"[Paso 6] Transcript cargado: {len(transcript)} segmentos")
        except Exception:
            log.warning("[Paso 6] No se pudo leer el transcript")

    audio_features: dict = {}
    if audio_features_path.exists():
        try:
            audio_features = json.loads(audio_features_path.read_text(encoding="utf-8"))
            log.info("[Paso 6] Audio features cargado")
        except Exception:
            log.warning("[Paso 6] No se pudo leer audio_features")

    # ── Resolver arco narrativo ──────────────────────────────────────────────
    clips_data = resolve_narrative_arc(analysis, log)
    if not clips_data:
        log.error("[Paso 6] Sin clips para extraer.")
        sys.exit(1)

    # ── Pre-filtro de calidad ────────────────────────────────────────────────
    if args.min_virality > 0:
        clips_data = pre_filter_clips(
            clips_data     = clips_data,
            analysis       = analysis,
            audio_features = audio_features,
            log            = log,
            min_score      = args.min_virality,
        )
    else:
        log.info("[Paso 6] Filtro de calidad desactivado (--min_virality 0)")

    if not clips_data:
        log.error(f"[Paso 6] Ningún clip superó el umbral {args.min_virality}/100.")
        sys.exit(1)

    clips_dir.mkdir(parents=True, exist_ok=True)

    video_duration = get_media_duration(cfg.video_path)
    if video_duration == 0.0:
        log.error("No se pudo obtener la duración del video")
        sys.exit(1)

    # ── Detectar barras negras ───────────────────────────────────────────────
    log.info("[Paso 6] Analizando barras negras del video fuente...")
    active_area = detect_active_area(cfg.video_path, log)
    if active_area:
        w, h, x, y = active_area
        log.info(f"[Paso 6] Crop activo: {w}×{h} offset ({x},{y})")
    else:
        log.info("[Paso 6] Sin barras negras, extracción directa")

    log.info(f"[Paso 6] Extrayendo {len(clips_data)} clips con padding {cfg.clip_padding_sec}s...")

    manifest_clips  = []
    extracted_count = 0

    for idx, clip_data in enumerate(clips_data, 1):
        start = clip_data.get("start")
        end   = clip_data.get("end")
        role  = clip_data.get("role", "body")

        if start is None or end is None:
            log.warning(f"  Clip {idx} ({role}): timestamps inválidos, saltando")
            continue

        start_padded = max(0.0, start - cfg.clip_padding_sec)
        end_padded   = min(video_duration, end + cfg.clip_padding_sec)
        duration     = end_padded - start_padded

        if duration < 0.5:
            log.warning(f"  Clip {idx} ({role}): duración muy corta ({duration:.1f}s), saltando")
            continue

        clip_filename = f"clip_{idx}.mp4"
        clip_path     = clips_dir / clip_filename
        role_emoji    = {"hook": "🎣", "body": "📚", "climax": "🔥", "cta": "📣"}.get(role, "•")

        log.info(
            f"  {role_emoji} Extrayendo clip {idx} [{role.upper()}]: "
            f"{start_padded:.1f}s → {end_padded:.1f}s ({duration:.1f}s)..."
        )

        ok = extract_clip(
            video_path  = cfg.video_path,
            start_sec   = start_padded,
            end_sec     = end_padded,
            output_path = clip_path,
            log         = log,
            active_area = active_area,
        )

        if ok:
            size_mb = clip_path.stat().st_size / (1024 * 1024)
            log.info(f"    ✓ {clip_filename} ({size_mb:.1f} MB, {duration:.1f}s)")

            clip_entry = {
                "filename":       clip_filename,
                "role":           role,
                "start_original": start,
                "end_original":   end,
                "start_padded":   start_padded,
                "end_padded":     end_padded,
                "duration_sec":   duration,
                "engagement_type": clip_data.get("engagement_type", "other"),
                # _llm_raw preserva el score 1-10 del LLM para que re-runs
                # no lo confundan con el score compuesto 0-100
                "_llm_raw":       clip_data.get("virality_score"),
                "razon":          clip_data.get("razon", ""),
                "directive":      clip_data.get("directive", {}),
                "guion":          build_clip_script(clip_data, transcript),
                "active_area_applied": (
                    {"w": active_area[0], "h": active_area[1], "x": active_area[2], "y": active_area[3]}
                    if active_area else None
                ),
            }
            manifest_clips.append(clip_entry)
            extracted_count += 1
        else:
            log.warning(f"    ✗ Falló extracción del clip {idx} ({role})")

    if not manifest_clips:
        log.error("[Paso 6] No se pudo extraer ningún clip.")
        sys.exit(1)

    # ── Calcular Virality Score final ────────────────────────────────────────
    log.info(f"\n[Paso 6] ── Calculando Virality Score ({len(manifest_clips)} clips) ──")
    virality_report = []

    for i, clip in enumerate(manifest_clips, 1):
        result = compute_virality_score(clip, analysis, audio_features)
        clip["virality_score"]  = result["virality_score"]
        clip["score_breakdown"] = result["score_breakdown"]

        label = virality_label(result["virality_score"])
        log.info(
            f"  clip_{i} [{clip['role'].upper():<7}] {result['virality_score']:3d}/100  {label}"
            f"  │ LLM:{result['score_breakdown']['llm']:3d}"
            f"  Audio:{result['score_breakdown']['audio']:3d}"
            f"  Eng:{result['score_breakdown']['engagement']:3d}"
            f"  Shots:{result['score_breakdown']['shots']:3d}"
        )

        virality_report.append({
            "clip":           clip["filename"],
            "role":           clip["role"],
            "virality_score": result["virality_score"],
            "label":          label,
            "breakdown":      result["score_breakdown"],
            "razon":          clip.get("razon", ""),
            "start":          clip["start_original"],
            "end":            clip["end_original"],
        })

    # ── Coherencia narrativa ─────────────────────────────────────────────────
    coherence = build_assembly_notes(manifest_clips, transcript)

    arc_emoji = {"completo": "✅", "parcial": "⚠️", "débil": "❌"}.get(coherence["narrative_arc"], "?")
    log.info(f"\n[Paso 6] ── Análisis de coherencia narrativa ──")
    log.info(f"  Arco narrativo: {arc_emoji} {coherence['narrative_arc'].upper()}")
    log.info(f"  Score promedio: {coherence['avg_virality_score']}/10")
    log.info(
        f"  Roles: hook={'✅' if coherence['has_hook'] else '❌'}  "
        f"climax={'✅' if coherence['has_climax'] else '❌'}  "
        f"cta={'✅' if coherence['has_cta'] else '❌'}  "
        f"emoción={'✅' if coherence['has_emotion'] else '❌'}  "
        f"educativo={'✅' if coherence['has_education'] else '❌'}"
    )
    if coherence["warnings"]:
        log.info("  ⚠️  Advertencias:")
        for w in coherence["warnings"]:
            log.info(f"    • {w}")
    if coherence["tips"]:
        log.info("  💡 Tips:")
        for t in coherence["tips"]:
            log.info(f"    • {t}")
    log.info(f"  Orden sugerido: {' → '.join(coherence['suggested_order'])}")

    # ── Ranking virality ─────────────────────────────────────────────────────
    ranked = sorted(virality_report, key=lambda x: x["virality_score"], reverse=True)
    log.info("\n  ── Ranking por plataforma ──────────────────────────")
    log.info("  TikTok / Instagram / YouTube  →  hooks y emoción primero:")
    for r in ranked:
        log.info(f"    {r['clip']}  {r['virality_score']:3d}/100  {r['label']}")

    if ranked:
        top = ranked[0]
        log.info(f"\n  🏆 Mejor clip: {top['clip']}  {top['virality_score']}/100")
        log.info(f"     \"{top['razon']}\"")

    # ── Guardar manifest ─────────────────────────────────────────────────────
    manifest_data = {
        "assembly_order": coherence["suggested_order"],
        "coherence":      coherence,
        "clips":          manifest_clips,
    }
    manifest_path.write_text(
        json.dumps(manifest_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info(f"\n[Paso 6] {extracted_count}/{len(clips_data)} clips → {clips_dir}")
    log.info(f"[Paso 6] Manifest guardado → {manifest_path}")

    # ── Guardar virality report ───────────────────────────────────────────────
    report_path.write_text(
        json.dumps({
            "run_id":   args.run_id,
            "clips":    ranked,
            "weights":  {"llm": W_LLM, "audio": W_AUDIO, "engagement": W_ENGAGEMENT, "shots": W_SHOTS},
            "coherence": coherence,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info(f"[Paso 6] Virality report → {report_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()