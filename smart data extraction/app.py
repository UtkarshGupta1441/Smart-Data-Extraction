import asyncio
import logging
import os
import traceback
from typing import Dict, List
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from yolo3 import DEFAULT_OUTPUT_DIR, run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
OUTPUT_URL_PREFIX = "/outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Smart Data Extraction", version="1.0")
app.mount(OUTPUT_URL_PREFIX, StaticFiles(directory=OUTPUT_DIR), name="outputs")

templates = Jinja2Templates(directory=TEMPLATES_DIR)
run_lock = asyncio.Lock()

SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

class RunRequest(BaseModel):
    url: str


def normalize_url(raw_url: str) -> str:
    cleaned = (raw_url or "").strip()
    if not cleaned:
        raise ValueError("URL is required.")

    if not cleaned.startswith(("http://", "https://")):
        cleaned = "https://" + cleaned

    parsed = urlparse(cleaned)
    if not parsed.netloc:
        raise ValueError("Invalid URL.")

    return cleaned


def collect_output_images() -> List[Dict[str, str]]:
    images: List[Dict[str, str]] = []
    if not os.path.isdir(OUTPUT_DIR):
        return images

    for root, _, files in os.walk(OUTPUT_DIR):
        for name in files:
            if not name.lower().endswith(SUPPORTED_IMAGE_EXTS):
                continue

            full_path = os.path.join(root, name)
            rel_path = os.path.relpath(full_path, OUTPUT_DIR)
            rel_url = rel_path.replace(os.sep, "/")
            folder = os.path.dirname(rel_url)

            images.append({
                "name": name,
                "folder": folder or "root",
                "url": f"{OUTPUT_URL_PREFIX}/{rel_url}",
                "updated": str(int(os.path.getmtime(full_path)))
            })

    images.sort(key=lambda item: (-int(item["updated"]), item["name"]))
    return images


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/images")
async def list_images():
    images = collect_output_images()
    return {"count": len(images), "images": images}


@app.post("/api/run")
async def run_extraction(payload: RunRequest):
    try:
        target_url = normalize_url(payload.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async with run_lock:
        try:
            await run_in_threadpool(run_pipeline, target_url)
        except Exception as exc:
            tb = traceback.format_exc()
            logging.error("Pipeline error:\n%s", tb)
            raise HTTPException(status_code=500, detail=str(exc) or repr(exc)) from exc

    images = collect_output_images()
    return {
        "status": "ok",
        "url": target_url,
        "count": len(images),
        "images": images
    }
