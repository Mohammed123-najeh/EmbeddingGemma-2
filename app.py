"""Local web UI for EmbeddingGemma 2 image & text search.

    python app.py                      # http://127.0.0.1:8000
    python app.py --images D:/photos   # index another folder
    python app.py --help               # all options
"""
from __future__ import annotations

import argparse
import io
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from engine import (DEFAULT_IMAGE_BUDGET, DEFAULT_IMAGE_DIR, DEFAULT_INDEX_DIR, DEFAULT_TEXTS_FILE,
                    DIMS, IMAGE_BUDGETS, ROOT, Engine, load_image)


def create_app(engine: Engine) -> FastAPI:
    def startup() -> None:
        try:
            engine.load_model()
        except Exception as e:
            engine.status.update(phase="error", error=f"Model failed to load: {type(e).__name__}: {e}")
            raise
        engine.safe_build_index()
        info = engine.info()
        if info["images"] == 0:
            print(f"\n  No images found in {engine.image_dir}\n"
                  f"  Add your own, or run:  python download_samples.py   then press Re-index.\n")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        threading.Thread(target=startup, daemon=True).start()
        yield

    engine.image_dir.mkdir(parents=True, exist_ok=True)
    engine.thumb_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="EmbeddingGemma 2 Search", lifespan=lifespan)
    app.mount("/media", StaticFiles(directory=engine.image_dir), name="media")
    app.mount("/thumbs", StaticFiles(directory=engine.thumb_dir), name="thumbs")

    @app.get("/")
    def home():
        return FileResponse(ROOT / "static" / "index.html")

    @app.get("/api/status")
    def status():
        return engine.info()

    @app.post("/api/search")
    def search(
        text: str = Form(""),
        ref: str = Form(""),
        k: int = Form(24),
        dim: int = Form(DIMS[0]),
        use_prompt: bool = Form(True),
        image: UploadFile | None = File(None),
    ):
        if engine.model is None:
            raise HTTPException(503, "Model is still loading.")
        if dim not in DIMS:
            raise HTTPException(400, f"dim must be one of {DIMS}")
        k = max(1, min(k, 200))
        text = text.strip()

        t0 = time.perf_counter()
        source, cached = None, False
        if ref:
            try:
                vec, source = engine.item_vector(ref)
            except KeyError:
                raise HTTPException(404, "That item is no longer in the index - try searching again.")
        else:
            img = None
            if image is not None and image.filename:
                try:
                    img = load_image(io.BytesIO(image.file.read()))
                except Exception:
                    raise HTTPException(400, "Could not read the uploaded image.")
            if not text and img is None:
                raise HTTPException(400, "Type a query or add an image.")
            vec, cached = engine.query_vector(text or None, img, use_prompt)
        t1 = time.perf_counter()
        result = engine.search(vec, k=k, dim=dim, exclude_id=ref or None)
        t2 = time.perf_counter()

        result["timing"] = {"encode_ms": round((t1 - t0) * 1000, 1),
                            "search_ms": round((t2 - t1) * 1000, 2),
                            "cached": cached}
        if source is not None:
            result["source"] = engine.public(source, 1.0)
        return result

    @app.post("/api/reindex")
    def reindex():
        if engine.model is None or engine.busy:
            return {"started": False}
        threading.Thread(target=engine.safe_build_index, daemon=True).start()
        return {"started": True}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="EmbeddingGemma 2 image & text search UI")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR,
                        help="folder of images to index, searched recursively (default: ./images)")
    parser.add_argument("--texts", type=Path, default=DEFAULT_TEXTS_FILE,
                        help="text file with one document per line (default: ./texts.txt)")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_DIR,
                        help="where embeddings and thumbnails are cached (default: ./index)")
    parser.add_argument("--image-tokens", type=int, default=DEFAULT_IMAGE_BUDGET, choices=IMAGE_BUDGETS,
                        help="vision tokens per image: more = finer detail, slower indexing (default: 1120)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser on start")
    args = parser.parse_args()

    engine = Engine(args.images, args.texts, args.index, args.image_tokens)
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    print(f"\n  EmbeddingGemma 2 search  ->  {url}\n  images: {engine.image_dir}\n")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
