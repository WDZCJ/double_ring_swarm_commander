"""Web 可视化后端服务（web/server.py）。

提供：
- HTTP 静态文件服务（/ → web/static/index.html）
- WebSocket /ws 实时推送 demo 运行数据
- SSE /events 实时推送（WebSocket 后备方案）
- REST API /api/state 获取当前完整状态快照

数据流：
- demo_web.py 在独立线程运行 DualLoopRunner
- 通过 hook 将数据推入线程安全 Queue
- Web 服务事件循环中的 bridge 任务将数据从 thread Queue 转移到 async broadcast
"""

from __future__ import annotations

import asyncio
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# 线程安全队列：runner 线程 → Web 服务事件循环
_thread_queue: queue.Queue[dict] = queue.Queue()

_ws_clients: list[WebSocket] = []
_sse_clients: list[asyncio.Queue] = []

app = FastAPI(title="双环Agent架构可视化")

# 静态文件目录
_static_dir = Path(__file__).parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# 运行时交互 REST API（/api/scenarios、/api/intents、/api/runtime/* 等）
from munition_system.web import interaction  # noqa: E402  (需在 app 定义后注册)

app.include_router(interaction.router)


def get_static_dir() -> Path:
    return _static_dir


async def broadcast(data: dict) -> None:
    """向所有客户端广播数据。"""
    text = json.dumps(data, ensure_ascii=False, default=str)

    # WebSocket 客户端
    dead_ws: list[WebSocket] = []
    for ws in _ws_clients:
        try:
            await ws.send_text(text)
        except Exception:
            dead_ws.append(ws)
    for ws in dead_ws:
        _ws_clients.remove(ws)

    # SSE 客户端
    dead_sse: list[asyncio.Queue] = []
    for q in _sse_clients:
        try:
            q.put_nowait(text)
        except Exception:
            dead_sse.append(q)
    for q in dead_sse:
        _sse_clients.remove(q)


def queue_broadcast(data: dict) -> None:
    """线程安全的广播入口：供 runner 线程调用。"""
    _thread_queue.put(data)


async def _bridge_thread_to_async() -> None:
    """桥接线程队列到异步广播。"""
    while True:
        try:
            # 非阻塞检查线程队列
            data = _thread_queue.get_nowait()
            await broadcast(data)
        except queue.Empty:
            pass
        await asyncio.sleep(0.05)  # 50ms 轮询间隔


@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(_bridge_thread_to_async())


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    html_path = _static_dir / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>Web UI not found</h1><p>Please create web/static/index.html</p>"


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


@app.get("/events")
async def sse_endpoint() -> StreamingResponse:
    """Server-Sent Events 后备方案。"""
    q: asyncio.Queue = asyncio.Queue()
    _sse_clients.append(q)

    async def event_generator():
        try:
            while True:
                text = await q.get()
                yield f"data: {text}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if q in _sse_clients:
                _sse_clients.remove(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/state")
async def get_state() -> dict:
    """获取当前 runner 状态快照。"""
    from munition_system.web.state import get_current_state

    return get_current_state()


def run_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    """启动 Web 服务（阻塞）。

    Windows 上需要 SelectorEventLoop 才能支持 WebSocket。
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


def start_server_in_thread(host: str = "127.0.0.1", port: int = 8080) -> threading.Thread:
    """在后台线程启动 Web 服务。"""
    t = threading.Thread(target=run_server, args=(host, port), daemon=True)
    t.start()
    return t
