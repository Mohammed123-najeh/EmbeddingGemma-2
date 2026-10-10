"""Indexing and search on top of EmbeddingGemma 2 (images + text in one vector space)."""
from __future__ import annotations

import hashlib
import json
import threading
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-2"
ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = ROOT / "images"
DEFAULT_TEXTS_FILE = ROOT / "texts.txt"
DEFAULT_INDEX_DIR = ROOT / "index"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DIMS = (768, 512, 256, 128)                 # Matryoshka sizes the model supports
IMAGE_BUDGETS = (70, 140, 280, 560, 1120)   # vision tokens per image; more = finer detail, slower
DEFAULT_IMAGE_BUDGET = 1120
QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: none | text: "
BATCH_SIZE = 8
THUMB_SIZE = 480
QUERY_CACHE_SIZE = 256


def load_image(source) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(source))
    return img.convert("RGB")


def pick_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        # bfloat16 where supported; the model does not support float16
        return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def _item_id(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


class Engine:
    def __init__(self, image_dir: Path = DEFAULT_IMAGE_DIR, texts_file: Path = DEFAULT_TEXTS_FILE,
                 index_dir: Path = DEFAULT_INDEX_DIR, image_budget: int = DEFAULT_IMAGE_BUDGET):
        if image_budget not in IMAGE_BUDGETS:
            raise ValueError(f"image_budget must be one of {IMAGE_BUDGETS}")
        self.image_dir = Path(image_dir).resolve()
        self.texts_file = Path(texts_file).resolve()
        self.index_dir = Path(index_dir).resolve()
        self.thumb_dir = self.index_dir / "thumbs"
        self.image_budget = image_budget
        self.model: SentenceTransformer | None = None
        self.device, self.dtype = "cpu", torch.float32
        self.status = {"phase": "starting", "done": 0, "total": 0, "error": None}
        # (items, {dim: normalized matrix}, is_image mask, {id: row}) swapped atomically
        self._index = ([], {}, np.zeros(0, dtype=bool), {})
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._gpu_lock = threading.Lock()
        self._index_lock = threading.Lock()

    # ---------- model ----------

    def load_model(self) -> None:
        self.status.update(phase="loading model", error=None)
        self.device, self.dtype = pick_device()
        self.model = SentenceTransformer(
            MODEL_ID, device=self.device, model_kwargs={"torch_dtype": self.dtype}
        )
        # Warm up the kernels so the first real search isn't slow.
        self._encode([QUERY_PREFIX + "warm up"])
        self._encode([{"image": Image.new("RGB", (64, 64))}], **self._image_kwargs())

    def _encode(self, inputs: list, **kwargs) -> np.ndarray:
        with self._gpu_lock, torch.inference_mode():
            out = self.model.encode(
                inputs, normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False, **kwargs,
            )
        return np.asarray(out, dtype=np.float32)

    def _image_kwargs(self) -> dict:
        return {"processing_kwargs": {"image": {"max_soft_tokens": self.image_budget}}}

    # ---------- indexing ----------

    @property
    def busy(self) -> bool:
        return self._index_lock.locked()

    def safe_build_index(self) -> None:
        try:
            self.build_index()
        except Exception as e:
            traceback.print_exc()
            self.status.update(phase="error", error=f"{type(e).__name__}: {e}")

    def build_index(self) -> None:
        with self._index_lock:
            images, texts = self._scan()
            cache = self._load_cache()
            todo_images = [it for it in images if it["key"] not in cache]
            todo_texts = [it for it in texts if it["key"] not in cache]
            self.status.update(phase="indexing", done=0,
                               total=len(todo_images) + len(todo_texts), error=None)
            self.thumb_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()

            fresh: dict[str, np.ndarray] = {}
            batches = [todo_images[i:i + BATCH_SIZE] for i in range(0, len(todo_images), BATCH_SIZE)]
            # Decode images and write thumbnails on worker threads while the model encodes
            # the previous batch, so the GPU is not left waiting on disk and JPEG decoding.
            with ThreadPoolExecutor(max_workers=4) as pool:
                pending = [pool.submit(self._prepare_image, it) for it in batches[0]] if batches else []
                for n, batch in enumerate(batches):
                    loaded = [f.result() for f in pending]
                    if n + 1 < len(batches):
                        pending = [pool.submit(self._prepare_image, it) for it in batches[n + 1]]
                    ok = [(it, img) for it, img in zip(batch, loaded) if img is not None]
                    if ok:
                        vecs = self._encode([{"image": img} for _, img in ok],
                                            batch_size=BATCH_SIZE, **self._image_kwargs())
                        fresh.update(zip((it["key"] for it, _ in ok), vecs))
                    self.status["done"] += len(batch)

            if todo_texts:
                vecs = self._encode([DOC_PREFIX + it["text"] for it in todo_texts], batch_size=32)
                fresh.update(zip((it["key"] for it in todo_texts), vecs))
                self.status["done"] += len(todo_texts)

            for it in images:  # thumbnails may have been deleted while vectors were cached
                if it["key"] in cache and not (self.thumb_dir / f"{it['id']}.jpg").exists():
                    self._prepare_image(it)

            vectors = {**cache, **fresh}
            items = [it for it in images + texts if it["key"] in vectors]
            dim = next(iter(vectors.values())).shape[0] if vectors else DIMS[0]
            emb = (np.stack([vectors[it["key"]] for it in items]) if items
                   else np.zeros((0, dim), dtype=np.float32))

            np.save(self.index_dir / "embeddings.npy", emb)
            meta = {"model": MODEL_ID, "image_budget": self.image_budget,
                    "image_dir": str(self.image_dir), "items": items}
            (self.index_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
            self._remove_orphan_thumbs({it["id"] for it in items})
            self._set_index(items, emb)

            took = time.perf_counter() - started
            if todo_images or todo_texts:
                print(f"indexed {len(todo_images)} images + {len(todo_texts)} texts in {took:.1f}s")
            self.status.update(phase="ready", last_index_seconds=round(took, 1),
                               newly_indexed=len(fresh))

    def _prepare_image(self, item: dict) -> Image.Image | None:
        try:
            img = load_image(self.image_dir / item["path"])
        except Exception as e:
            print(f"skipping unreadable image {item['path']}: {e}")
            return None
        thumb = img.copy()
        thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))
        thumb.save(self.thumb_dir / f"{item['id']}.jpg", quality=85)
        return img

    def _scan(self) -> tuple[list[dict], list[dict]]:
        self.image_dir.mkdir(parents=True, exist_ok=True)
        images = []
        for p in sorted(self.image_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                st = p.stat()
                rel = p.relative_to(self.image_dir).as_posix()
                key = f"image:{rel}:{st.st_size}:{st.st_mtime_ns}"
                images.append({"id": _item_id(key), "kind": "image", "path": rel, "key": key})
        texts, seen = [], set()
        if self.texts_file.exists():
            for line in self.texts_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and line not in seen:
                    seen.add(line)
                    key = "text:" + line
                    texts.append({"id": _item_id(key), "kind": "text", "text": line, "key": key})
        return images, texts

    def _load_cache(self) -> dict[str, np.ndarray]:
        meta_path, emb_path = self.index_dir / "meta.json", self.index_dir / "embeddings.npy"
        if not (meta_path.exists() and emb_path.exists()):
            return {}
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (meta.get("model") != MODEL_ID or meta.get("image_budget") != self.image_budget
                or meta.get("image_dir") != str(self.image_dir)):
            return {}
        emb = np.load(emb_path)
        return {it["key"]: emb[i] for i, it in enumerate(meta["items"])}

    def _remove_orphan_thumbs(self, keep: set[str]) -> None:
        for f in self.thumb_dir.glob("*.jpg"):
            if f.stem not in keep:
                f.unlink(missing_ok=True)

    def _set_index(self, items: list[dict], emb: np.ndarray) -> None:
        views = {}
        for d in DIMS:
            v = emb[:, :d]
            views[d] = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
        is_image = np.array([it["kind"] == "image" for it in items], dtype=bool)
        rows = {it["id"]: i for i, it in enumerate(items)}
        self._index = (items, views, is_image, rows)

    # ---------- search ----------

    def query_vector(self, text: str | None, image: Image.Image | None,
                     use_prompt: bool = True) -> tuple[np.ndarray, bool]:
        """Returns (vector, served_from_cache)."""
        if text and use_prompt:
            text = QUERY_PREFIX + text
        if image is not None and text:
            return self._encode([{"text": text, "image": image}], **self._image_kwargs())[0], False
        if image is not None:
            return self._encode([{"image": image}], **self._image_kwargs())[0], False
        # Text queries are cached so re-running a search (e.g. at another vector size) is instant.
        if text in self._query_cache:
            self._query_cache.move_to_end(text)
            return self._query_cache[text], True
        vec = self._encode([text])[0]
        self._query_cache[text] = vec
        if len(self._query_cache) > QUERY_CACHE_SIZE:
            self._query_cache.popitem(last=False)
        return vec, False

    def item_vector(self, item_id: str) -> tuple[np.ndarray, dict]:
        items, views, _, rows = self._index
        if item_id not in rows:
            raise KeyError(item_id)
        row = rows[item_id]
        return views[DIMS[0]][row], items[row]

    def search(self, vec: np.ndarray, k: int = 24, dim: int = DIMS[0],
               exclude_id: str | None = None) -> dict:
        items, views, is_image, rows = self._index
        if not items:
            return {"images": [], "texts": [], "searched": 0}
        q = vec[:dim]
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        scores = views[dim] @ q
        if exclude_id in rows:
            scores[rows[exclude_id]] = -np.inf
        result = {"searched": len(items)}
        # Rank each modality separately: text-to-text scores are naturally higher than
        # text-to-image scores (the "modality gap"), so a single mixed list would be skewed.
        for name, mask in (("images", is_image), ("texts", ~is_image)):
            idx = np.flatnonzero(mask & np.isfinite(scores))
            top = idx[np.argsort(-scores[idx])[:k]]
            result[name] = [self.public(items[i], scores[i]) for i in top]
        return result

    @staticmethod
    def public(item: dict, score: float) -> dict:
        out = {"id": item["id"], "kind": item["kind"], "score": round(float(score), 4)}
        if item["kind"] == "image":
            out["path"] = item["path"]
        else:
            out["text"] = item["text"]
        return out

    def info(self) -> dict:
        items, _, is_image, _ = self._index
        gpu = torch.cuda.get_device_name(0) if self.device == "cuda" else None
        return {
            **self.status,
            "model": MODEL_ID,
            "model_loaded": self.model is not None,
            "device": self.device,
            "gpu": gpu,
            "dtype": str(self.dtype).replace("torch.", ""),
            "image_budget": self.image_budget,
            "images": int(is_image.sum()),
            "texts": int(len(items) - is_image.sum()),
            "dims": list(DIMS),
            "busy": self.busy,
        }
