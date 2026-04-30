"""
run_pipeline.py — Orquestador principal.

Ejecuta cada paso como un proceso Python separado, en orden.
Cada proceso carga solo lo que necesita, libera GPU/RAM al terminar,
y el siguiente arranca desde cero.

Pipeline completo (automático):
  0   → split video en chunks (solo si se pide explícitamente)
  1   → extraer audio
  2   → transcribir con Whisper
  2.5 → analizar energía de audio
  2.8 → embeddings del transcript
  3   → detectar escenas
  4   → describir frames con LLaVA
  4.5 → detectar temas
  5   → analizar con LLM de texto
  6   → extraer clips + Virality Score
  6.5 → detectar protagonista

OPCIONAL (ejecutar manualmente):
  9   → Burn-in de captions animados
      python step9_captions.py --run_id <run_id>

Uso:
    python run_pipeline.py
    python run_pipeline.py --non_interactive
    python run_pipeline.py --skip_vision
    python run_pipeline.py --only_steps 1 2
    python run_pipeline.py --run_id 20260417_203924 --only_steps 5
    python run_pipeline.py --min_virality 65        ← solo clips con score ≥ 65
    python run_pipeline.py --min_virality 0         ← desactivar filtro
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Cada tupla: (id_str, script, descripción)
STEPS = [
    ("0",   "step0_split_video.py",          "Dividir video en chunks (opcional)"),
    ("1",   "step1_extract_audio.py",        "Extraer audio"),
    ("2",   "step2_transcribe.py",           "Transcribir con Whisper"),
    ("2.5", "step2b_audio_features.py",      "Analizar energía de audio"),
    ("2.8", "step2c_embed_transcript.py",    "Generar embeddings del transcript"),
    ("3",   "step3_detect_scenes.py",        "Detectar escenas"),
    ("4",   "step4_describe_frames.py",      "Describir frames con LLaVA"),
    ("4.5", "step4a_detect_themes.py",       "Detectar temas"),
    ("5",   "step5_analyze.py",              "Analizar con LLM de texto"),
    ("6",   "step6_extract_clips.py",        "Extraer clips + Virality Score"),
    ("6.5", "step6b_track_protagonist.py",   "Detectar protagonista"),
    # Paso 9 removido del pipeline automático: ejecutar manualmente con
    # python step9_captions.py --run_id <run_id>
]


def _build_cmd(
    script: str,
    run_id: str,
    non_interactive: bool,
    min_virality: int,
) -> list[str]:
    cmd = [sys.executable, script, "--run_id", run_id]
    if script == "step4a_detect_themes.py" and non_interactive:
        cmd.append("--non_interactive")
    if script == "step6_extract_clips.py":
        cmd += ["--min_virality", str(min_virality)]
    return cmd


def _build_env(video_path: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if video_path:
        env["VIDEO_PATH"] = video_path
    return env


def run_step(
    script:          str,
    run_id:          str,
    non_interactive: bool = False,
    min_virality:    int  = 50,
    video_path:      str | None = None,
) -> bool:
    """
    Ejecuta un paso del pipeline como subproceso.
    Pasa argumentos específicos según el script.
    """
    cmd = _build_cmd(script, run_id, non_interactive, min_virality)
    env = _build_env(video_path)
    result = subprocess.run(cmd, cwd=Path(__file__).parent, env=env)
    return result.returncode == 0


def run_step_capture(
    script:          str,
    run_id:          str,
    non_interactive: bool = False,
    min_virality:    int  = 50,
    video_path:      str | None = None,
) -> tuple[bool, str]:
    cmd = _build_cmd(script, run_id, non_interactive, min_virality)
    env = _build_env(video_path)

    res = subprocess.run(
        cmd,
        cwd=Path(__file__).parent,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
    return res.returncode == 0, out.strip()


def normalize_step_id(s: str) -> str:
    """Normaliza IDs de paso: '4.50' → '4.5', '2.0' → '2', '1' → '1'."""
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def _select_steps(only_steps: list[str] | None, skip_vision: bool) -> list[tuple[str, str, str]]:
    steps_to_run = list(STEPS)

    if not only_steps or "0" not in [normalize_step_id(s) for s in only_steps]:
        steps_to_run = [s for s in steps_to_run if s[0] != "0"]

    if skip_vision:
        steps_to_run = [s for s in steps_to_run if s[0] not in ("4", "4.5")]

    if only_steps:
        requested = {normalize_step_id(s) for s in only_steps}
        if "5" in requested:
            for auto in ("6", "6.5"):
                requested.add(auto)
        elif "6" in requested:
            for auto in ("6.5",):
                requested.add(auto)
        steps_to_run = [s for s in STEPS if s[0] in requested]

    return steps_to_run


def _is_non_blocking_failure(step_id: str) -> bool:
    return step_id == "4.5"


def _tail_text(text: str, max_chars: int = 50000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _launch_ui() -> None:
    try:
        import gradio as gr
    except Exception:
        print("Falta Gradio. Instalá con: pip install gradio")
        sys.exit(1)

    from config import Config

    def _default_video_path() -> str:
        try:
            return Config().video_path
        except Exception:
            return ""

    def _run_output_dir(run_id: str) -> Path:
        return Path("output") / run_id

    def _collect_clips(run_id: str) -> list[str]:
        clips_dir = _run_output_dir(run_id) / "clips"
        if not clips_dir.exists():
            return []
        return [str(p) for p in sorted(clips_dir.glob("clip_*.mp4"))]

    def _open_output_folder(run_id: str, log_text: str) -> str:
        p = _run_output_dir(run_id)
        if not run_id:
            return log_text + "\n\n[UI] run_id vacío."
        if not p.exists():
            return log_text + f"\n\n[UI] No existe: {p}"
        try:
            if os.name == "nt":
                os.startfile(str(p))
                return log_text + f"\n\n[UI] Abriendo carpeta: {p}"
        except Exception as e:
            return log_text + f"\n\n[UI] No se pudo abrir carpeta: {e}"
        return log_text + f"\n\n[UI] Carpeta: {p}"

    def _run_pipeline_ui(video_path_ui: str, skip_vision_ui: bool, non_interactive_ui: bool, min_virality_ui: int):
        run_id_ui = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_text = "\n".join(
            [
                f"run_id: {run_id_ui}",
                f"video_path: {video_path_ui}",
                f"skip_vision: {skip_vision_ui}",
                f"min_virality: {min_virality_ui}",
            ]
        )
        yield run_id_ui, log_text, [], str(_run_output_dir(run_id_ui)), None

        steps_to_run_ui = _select_steps(only_steps=None, skip_vision=skip_vision_ui)

        for step_id, script, description in steps_to_run_ui:
            log_text += f"\n\n▶ Paso {step_id}: {description}"
            yield run_id_ui, _tail_text(log_text), [], str(_run_output_dir(run_id_ui)), None

            ok, out = run_step_capture(
                script=script,
                run_id=run_id_ui,
                non_interactive=non_interactive_ui,
                min_virality=min_virality_ui,
                video_path=video_path_ui,
            )

            if out:
                log_text += "\n" + out

            if ok:
                log_text += f"\n✓ Paso {step_id} completado"
            else:
                if _is_non_blocking_failure(step_id):
                    log_text += f"\n⚠️ Paso {step_id} falló. Continuando sin themes.json..."
                    yield run_id_ui, _tail_text(log_text), [], str(_run_output_dir(run_id_ui)), None
                    continue
                log_text += f"\n✗ Paso {step_id} falló. Pipeline detenido."
                yield run_id_ui, _tail_text(log_text), [], str(_run_output_dir(run_id_ui)), None
                return

            yield run_id_ui, _tail_text(log_text), [], str(_run_output_dir(run_id_ui)), None

        clips = _collect_clips(run_id_ui)
        if clips:
            log_text += "\n\nClips generados:\n" + "\n".join(clips)
        log_text += f"\n\nOutput: {_run_output_dir(run_id_ui)}"
        preview = clips[0] if clips else None
        yield run_id_ui, _tail_text(log_text), clips, str(_run_output_dir(run_id_ui)), preview

    with gr.Blocks(title="VideoAnalyzer") as demo:
        gr.Markdown("# VideoAnalyzer — generar clips")
        video_path_ui = gr.Textbox(label="Video path", value=_default_video_path())
        with gr.Row():
            skip_vision_ui = gr.Checkbox(label="Skip visión (pasos 4 y 4.5)", value=False)
            non_interactive_ui = gr.Checkbox(label="No interactivo", value=True)
        min_virality_ui = gr.Slider(0, 100, value=50, step=1, label="Min virality (paso 6)")
        run_btn = gr.Button("Correr pipeline (hasta clips)")
        run_id_out = gr.Textbox(label="run_id", interactive=False)
        logs_out = gr.Textbox(label="Logs", lines=22, interactive=False)
        clips_out = gr.Files(label="Clips", interactive=False)
        output_dir_out = gr.Textbox(label="Output dir", interactive=False)
        preview_out = gr.Video(label="Preview (primer clip)", interactive=False)
        open_folder_btn = gr.Button("Abrir carpeta output")
        run_btn.click(
            _run_pipeline_ui,
            inputs=[video_path_ui, skip_vision_ui, non_interactive_ui, min_virality_ui],
            outputs=[run_id_out, logs_out, clips_out, output_dir_out, preview_out],
        )
        open_folder_btn.click(
            _open_output_folder,
            inputs=[run_id_out, logs_out],
            outputs=[logs_out],
        )

    try:
        demo.queue(concurrency_count=1)
    except TypeError:
        demo.queue()
    demo.launch()


def main():
    parser = argparse.ArgumentParser(description="Pipeline modular de análisis de video")
    parser.add_argument(
        "--non_interactive",
        action="store_true",
        help="Ejecutar en modo no-interactivo (auto-confirmar temas)",
    )
    parser.add_argument(
        "--skip_vision",
        action="store_true",
        help="Omitir los pasos 4 y 4.5 (visión + temas)",
    )
    parser.add_argument(
        "--only_steps",
        nargs="+",
        metavar="N",
        help="Correr solo estos pasos (ej: --only_steps 4 5  o  --only_steps 2.5 2.8)",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Reutilizar un run_id existente (para retomar desde un paso)",
    )
    parser.add_argument(
        "--min_virality",
        type=int,
        default=50,
        metavar="N",
        help=(
            "Score mínimo 0-100 para extraer un clip en el paso 6 (default: 50). "
            "Usar 0 para desactivar el filtro."
        ),
    )
    parser.add_argument(
        "--video_path",
        default=None,
        help="Ruta al video de entrada (override sin editar config.py).",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Lanzar interfaz web local (Gradio).",
    )
    args = parser.parse_args()

    if args.ui:
        _launch_ui()
        return

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = args.video_path

    print(f"\n{'='*55}")
    print(f"  VideoAnalyzer — Pipeline modular")
    print(f"  run_id: {run_id}")
    if args.min_virality > 0:
        print(f"  Filtro viral: ≥ {args.min_virality}/100")
    else:
        print(f"  Filtro viral: desactivado")
    print(f"{'='*55}\n")

    steps_to_run = _select_steps(only_steps=args.only_steps, skip_vision=args.skip_vision)

    for step_id, script, description in steps_to_run:
        print(f"\n▶  Paso {step_id}: {description}...")
        success = run_step(
            script          = script,
            run_id          = run_id,
            non_interactive = args.non_interactive,
            min_virality    = args.min_virality,
            video_path      = video_path,
        )
        if success:
            print(f"✓  Paso {step_id} completado")
        else:
            if _is_non_blocking_failure(step_id):
                print(f"⚠️  Paso {step_id} falló. Continuando sin themes.json...")
                continue
            print(f"✗  Paso {step_id} falló. Pipeline detenido.")
            print(f"\n   Para retomar desde aquí:")
            print(f"   python run_pipeline.py --run_id {run_id} --only_steps {step_id}")
            sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Pipeline completado. Output en: output/{run_id}/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
