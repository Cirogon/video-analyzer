"""
Paso 2c: Genera embeddings de cada segmento del transcript con nomic-embed-text.
Lee:    output/{run_id}/transcript.json
Guarda: output/{run_id}/vector_store.json

Cada entrada del vector store:
  {
    "index":     <int>,
    "start":     <float>,
    "end":       <float>,
    "text":      <str>,
    "embedding": [<float>, ...]   # 768 dimensiones
  }

Usa /api/embed (batch) en lugar de /api/embeddings (1 a 1) para
reducir el overhead HTTP y procesar hasta 20x más rápido.
Fallback automático a /api/embeddings si el servidor no soporta batch.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from config import Config, get_run_dir, setup_logging

DEFAULT_BATCH_SIZE = 100  # ajustable según VRAM disponible


def get_embed_base_url(ollama_url: str) -> str:
    return ollama_url.replace("/api/generate", "").rstrip("/")


def embed_batch(texts: list[str], model: str, base_url: str, log) -> list[list[float]] | None:
    """
    Llama a /api/embed con múltiples textos en una sola request.
    Retorna lista de vectores en el mismo orden que `texts`, o None si falla.
    Disponible en Ollama >= 0.1.31.
    """
    import requests
    url = base_url + "/api/embed"
    try:
        res = requests.post(
            url,
            json={"model": model, "input": texts},
            timeout=120,
        )
        if res.status_code == 404:
            return None  # endpoint no disponible, usar fallback
        if res.status_code != 200:
            log.warning(f"  /api/embed HTTP {res.status_code}: {res.text[:120]}")
            return None
        data = res.json()
        # Ollama devuelve {"embeddings": [[...], [...]]}
        embeddings = data.get("embeddings") or data.get("embedding")
        if not embeddings or len(embeddings) != len(texts):
            log.warning(f"  /api/embed: respuesta inesperada ({len(embeddings or [])} vs {len(texts)} esperados)")
            return None
        return embeddings
    except Exception as e:
        log.warning(f"  /api/embed error: {e}")
        return None


def embed_single(text: str, model: str, base_url: str, log) -> list[float] | None:
    """
    Fallback: /api/embeddings con un solo texto.
    Disponible en todas las versiones de Ollama.
    """
    import requests
    url = base_url + "/api/embeddings"
    for attempt in range(3):
        try:
            res = requests.post(
                url,
                json={"model": model, "prompt": text},
                timeout=60,
            )
            if res.status_code != 200:
                log.warning(f"  /api/embeddings HTTP {res.status_code}")
                time.sleep(2)
                continue
            return res.json().get("embedding")
        except Exception as e:
            log.warning(f"  Error embedding (intento {attempt + 1}): {e}")
            time.sleep(2)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Textos por request a /api/embed (default: {DEFAULT_BATCH_SIZE})",
    )
    args = parser.parse_args()

    cfg = Config()
    out_dir = get_run_dir(cfg.output_dir, args.run_id)
    log = setup_logging(out_dir)
    base_url = get_embed_base_url(cfg.ollama_url)

    transcript_path   = out_dir / "transcript.json"
    vector_store_path = out_dir / "vector_store.json"

    if not transcript_path.exists():
        log.error(f"No se encontró {transcript_path}. ¿Corriste el paso 2?")
        sys.exit(1)

    transcript: list[dict] = json.loads(transcript_path.read_text("utf-8"))

    # Cargar caché para reanudar si fue interrumpido
    existing: dict[int, dict] = {}
    if vector_store_path.exists():
        try:
            for entry in json.loads(vector_store_path.read_text("utf-8")):
                existing[entry["index"]] = entry
            log.info(f"[Paso 2c] Caché cargado: {len(existing)} embeddings ya generados")
        except Exception:
            log.warning("[Paso 2c] No se pudo leer el vector store existente, regenerando.")

    # Separar segmentos que ya tienen embedding de los que falta procesar
    pending: list[tuple[int, dict]] = [
        (idx, seg) for idx, seg in enumerate(transcript)
        if idx not in existing
    ]

    log.info(
        f"[Paso 2c] {len(existing)} en caché · {len(pending)} pendientes · "
        f"batch_size={args.batch_size} · modelo={cfg.embed_model}"
    )

    # Detectar si /api/embed está disponible con un batch de prueba
    use_batch = False
    if pending:
        test_text = pending[0][1].get("text", "test") or "test"
        test_result = embed_batch([test_text], cfg.embed_model, base_url, log)
        use_batch = test_result is not None
        log.info(
            f"[Paso 2c] Modo: {'BATCH (/api/embed) — más rápido' if use_batch else 'INDIVIDUAL (/api/embeddings) — actualizá Ollama para batch'}"
        )

    # Inicializar results con los ya cacheados
    results: list[dict | None] = [None] * len(transcript)
    for idx, entry in existing.items():
        results[idx] = entry

    t_start = time.time()
    processed = 0
    failed = 0

    if use_batch:
        # ── MODO BATCH ──────────────────────────────────────────────────────
        batch_size = args.batch_size
        i = 0
        while i < len(pending):
            batch = pending[i: i + batch_size]
            texts = [seg.get("text", "").strip() for _, seg in batch]

            embeddings = embed_batch(texts, cfg.embed_model, base_url, log)

            if embeddings is None:
                # Batch falló: procesar uno a uno como fallback
                log.warning(f"  Batch falló, procesando {len(batch)} segmentos individualmente...")
                embeddings = []
                for text in texts:
                    if text:
                        emb = embed_single(text, cfg.embed_model, base_url, log)
                        embeddings.append(emb or [])
                    else:
                        embeddings.append([])

            for (idx, seg), emb in zip(batch, embeddings):
                if not emb:
                    failed += 1
                entry = {
                    "index":     idx,
                    "start":     seg.get("start", 0),
                    "end":       seg.get("end", 0),
                    "text":      seg.get("text", "").strip(),
                    "embedding": emb or [],
                }
                results[idx] = entry
                existing[idx] = entry

            processed += len(batch)
            i += batch_size

            # Guardar caché y mostrar progreso con ETA
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (len(pending) - processed) / rate if rate > 0 else 0
            log.info(
                f"  ... {len(existing)}/{len(transcript)} embebidos "
                f"({rate:.1f}/s · ETA {eta:.0f}s)"
            )
            vector_store_path.write_text(
                json.dumps([r for r in results if r], ensure_ascii=False),
                encoding="utf-8",
            )

    else:
        # ── MODO INDIVIDUAL (fallback) ───────────────────────────────────────
        for i, (idx, seg) in enumerate(pending):
            text = seg.get("text", "").strip()
            emb: list[float] = []
            if text:
                result = embed_single(text, cfg.embed_model, base_url, log)
                if result:
                    emb = result
                else:
                    failed += 1

            entry = {
                "index":     idx,
                "start":     seg.get("start", 0),
                "end":       seg.get("end", 0),
                "text":      text,
                "embedding": emb,
            }
            results[idx] = entry
            existing[idx] = entry
            processed += 1

            if processed % 20 == 0:
                elapsed = time.time() - t_start
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (len(pending) - processed) / rate if rate > 0 else 0
                log.info(
                    f"  ... {len(existing)}/{len(transcript)} embebidos "
                    f"({rate:.1f}/s · ETA {eta:.0f}s)"
                )
                vector_store_path.write_text(
                    json.dumps([r for r in results if r], ensure_ascii=False),
                    encoding="utf-8",
                )

    # Guardar resultado final
    final = [r for r in results if r is not None]
    vector_store_path.write_text(
        json.dumps(final, ensure_ascii=False),
        encoding="utf-8",
    )

    total_time = time.time() - t_start
    valid = sum(1 for r in final if r["embedding"])
    log.info(
        f"[Paso 2c] Completado en {total_time:.1f}s · "
        f"{valid}/{len(transcript)} embeddings válidos · "
        f"{failed} fallidos → {vector_store_path}"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()