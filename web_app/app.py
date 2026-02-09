"""FastAPI backend for real-time LLVC voice conversion."""

import asyncio
import json
import os
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from web_app.audio_processor import StreamingInferenceEngine, MODELS_REGISTRY
from web_app.memory_monitor import MemoryMonitor

app = FastAPI(title="LLVC Real-Time Voice Conversion")

WEB_APP_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(WEB_APP_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(WEB_APP_DIR, "templates"))

# Global state
engine = StreamingInferenceEngine()
monitor = MemoryMonitor()
inference_executor = ThreadPoolExecutor(max_workers=1)
uploaded_files = {}  # file_id -> {"path": ..., "sr": ..., "duration": ...}


# --- REST endpoints ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/models")
async def list_models():
    models = {}
    for key, info in MODELS_REGISTRY.items():
        models[key] = {
            "name": info["name"],
            "description": info["description"],
        }
    return {
        "models": models,
        "current": engine.current_model_key,
        "is_ready": engine.is_ready,
    }


@app.post("/api/models/load")
async def load_model(body: dict):
    model_key = body.get("model_key")
    if model_key not in MODELS_REGISTRY:
        return JSONResponse({"error": f"Unknown model: {model_key}"}, status_code=400)

    engine.is_ready = False
    monitor.start_session()
    loop = asyncio.get_event_loop()

    t0 = time.perf_counter()
    load_info = await loop.run_in_executor(inference_executor, engine.load_model, model_key)
    load_time = time.perf_counter() - t0

    mem = monitor.measure()
    monitor.stop_session()

    return {
        "status": "loaded",
        "load_time_ms": round(load_time * 1000, 1),
        "memory": mem,
        **load_info,
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())[:8]
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    tmp_path = os.path.join(tempfile.gettempdir(), f"llvc_upload_{file_id}{suffix}")

    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    # Decode and resample to model SR
    audio, sr = torchaudio.load(tmp_path)
    audio = audio.mean(0, keepdim=False)  # mono
    target_sr = engine.sr or 16000
    if sr != target_sr:
        audio = torchaudio.transforms.Resample(sr, target_sr)(audio)

    audio_np = audio.numpy()
    duration = len(audio_np) / target_sr

    uploaded_files[file_id] = {
        "path": tmp_path,
        "audio": audio_np,
        "sr": target_sr,
        "duration": duration,
        "filename": file.filename,
    }

    return {
        "file_id": file_id,
        "filename": file.filename,
        "duration_s": round(duration, 3),
        "sample_rate": target_sr,
        "num_samples": len(audio_np),
    }


@app.get("/api/status")
async def status():
    mem = monitor.measure() if monitor._active else {"rss_mb": 0}
    return {
        "is_ready": engine.is_ready,
        "current_model": engine.current_model_key,
        "sr": engine.sr,
        "chunk_len": engine.get_chunk_len(),
        "chunk_factor": engine.chunk_factor,
        "memory": mem,
    }


# --- WebSocket: real-time mic streaming ---

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    if not engine.is_ready:
        await ws.send_text(json.dumps({"type": "error", "message": "Model not loaded"}))
        await ws.close()
        return

    engine.init_streaming_state()
    monitor.start_session()
    loop = asyncio.get_event_loop()

    try:
        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            if "text" in msg:
                data = json.loads(msg["text"])
                action = data.get("action")
                if action == "stop":
                    break
                elif action == "reset_buffers":
                    engine.init_streaming_state()
                    await ws.send_text(json.dumps({"type": "reset_done"}))
                    continue
                elif action == "set_chunk_factor":
                    engine.chunk_factor = int(data["value"])
                    engine.init_streaming_state()
                    await ws.send_text(json.dumps({
                        "type": "config_updated",
                        "chunk_len": engine.get_chunk_len(),
                        "chunk_factor": engine.chunk_factor,
                    }))
                    continue

            elif "bytes" in msg:
                audio_bytes = msg["bytes"]
                audio_np = np.frombuffer(audio_bytes, dtype=np.float32).copy()

                try:
                    output_np, metrics = await loop.run_in_executor(
                        inference_executor, engine.process_chunk, audio_np
                    )
                except Exception as e:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": str(e),
                    }))
                    continue

                mem = monitor.measure()
                metrics["memory"] = mem
                metrics["type"] = "audio_chunk"

                await ws.send_text(json.dumps(metrics))
                await ws.send_bytes(output_np.tobytes())

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        monitor.stop_session()


# --- WebSocket: file streaming conversion ---

@app.websocket("/ws/file-stream")
async def ws_file_stream(ws: WebSocket):
    await ws.accept()
    if not engine.is_ready:
        await ws.send_text(json.dumps({"type": "error", "message": "Model not loaded"}))
        await ws.close()
        return

    monitor.start_session()
    loop = asyncio.get_event_loop()

    try:
        # Wait for file_id from client
        msg = await ws.receive_text()
        data = json.loads(msg)
        file_id = data.get("file_id")

        if file_id not in uploaded_files:
            await ws.send_text(json.dumps({"type": "error", "message": "File not found"}))
            await ws.close()
            return

        file_info = uploaded_files[file_id]
        audio_np = file_info["audio"]
        total_chunks = len(audio_np) // engine.get_chunk_len() + (
            1 if len(audio_np) % engine.get_chunk_len() != 0 else 0
        )

        await ws.send_text(json.dumps({
            "type": "file_start",
            "filename": file_info["filename"],
            "duration_s": file_info["duration"],
            "total_chunks": total_chunks,
            "sr": file_info["sr"],
        }))

        def _process_generator():
            return list(engine.process_file_audio(audio_np))

        results = await loop.run_in_executor(inference_executor, _process_generator)

        for output_chunk, metrics in results:
            mem = monitor.measure()
            metrics["memory"] = mem
            metrics["type"] = "audio_chunk"
            metrics["total_chunks"] = total_chunks

            await ws.send_text(json.dumps(metrics))
            await ws.send_bytes(output_chunk.tobytes())

        await ws.send_text(json.dumps({
            "type": "file_complete",
            "total_chunks": total_chunks,
        }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        monitor.stop_session()
