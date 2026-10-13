"""Karllo AI FastAPI application."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from karllo import __version__
from karllo.api.deps import owner_required
from karllo.conversations import (
    append_messages,
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
    rename_conversation,
    save_conversation,
)
from karllo.hardware import estimate_run, hardware_report
from karllo.memory.store import (
    add_memory,
    delete_memory,
    enabled_prefix,
    list_memories,
    update_memory,
)
from karllo.model.checkpoints import (
    ensure_original_checkpoint,
    list_checkpoints,
    load_manifest,
    read_active,
    set_active,
    verify_checkpoint,
)
from karllo.model.config import KarlloConfig
from karllo.model.inference import GenerationSettings, InferenceEngine
from karllo.model.tokenizer import KarlloTokenizer
from karllo.paths import (
    CHECKPOINTS_DIR,
    DATASETS_DIR,
    EVAL_DIR,
    ROOT,
    RUNS_DIR,
    SEED_DIR,
    UPLOADS_DIR,
    ensure_dirs,
)
from karllo.research.web import prefix_for_prompt as research_prefix
from karllo.research.web import research_status, search_web
from karllo.retrieval.knowledge import KnowledgeBase, embedding_status
from karllo.security import (
    allowed_upload_name,
    create_session,
    ensure_owner,
    revoke_session,
    verify_password,
)
from karllo.training.data import (
    MAX_UPLOAD_BYTES,
    analyze_dataset,
    get_dataset,
    list_datasets,
    new_dataset_record,
)
from karllo.training.engine import TrainConfig, MODES
from karllo.training.evaluate import compare_checkpoints, evaluate_checkpoint, save_eval_report

ensure_dirs()
ensure_original_checkpoint()
ENGINE = InferenceEngine()
KB = KnowledgeBase()
RUNNING: dict[str, subprocess.Popen] = {}


def _try_load_engine() -> None:
    try:
        ENGINE.load_active()
    except Exception:
        pass


_try_load_engine()

app = FastAPI(title="Karllo AI", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginIn(BaseModel):
    password: str


class ChatIn(BaseModel):
    conversation_id: Optional[str] = None
    content: str
    use_memory: bool = True
    use_knowledge: bool = False
    use_research: bool = False
    temperature: float = 0.9
    top_k: int = 40
    max_new_tokens: int = 80


class RenameIn(BaseModel):
    title: str


class MemoryIn(BaseModel):
    content: str
    enabled: Optional[bool] = True


class MemoryPatch(BaseModel):
    content: Optional[str] = None
    enabled: Optional[bool] = None


class TrainIn(BaseModel):
    mode: str
    dataset_id: str
    model_size: str = "tiny"
    max_steps: int = 30
    batch_size: int = 2
    grad_accum: int = 2
    lr: float = 3e-4
    ckpt_every: int = 15
    source_checkpoint: Optional[str] = None
    vocab_size: int = 512
    n_positions: Optional[int] = None
    max_examples: int = 400
    train_tokenizer: bool = True


class ActivateIn(BaseModel):
    path: str
    confirm: bool = False


class EvalIn(BaseModel):
    path: str
    compare_to: Optional[str] = None


class KnowledgeTextIn(BaseModel):
    name: str
    text: str


class ResearchIn(BaseModel):
    query: str


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "name": "Karllo AI", "version": __version__}


@app.get("/api/auth/status")
def auth_status() -> dict[str, Any]:
    rec = ensure_owner()
    return {"must_change": rec.get("must_change", False), "hint": rec.get("hint")}


@app.post("/api/auth/login")
def login(body: LoginIn) -> JSONResponse:
    if not verify_password(body.password):
        raise HTTPException(status_code=401, detail="Invalid password")
    token = create_session()
    resp = JSONResponse({"ok": True, "token": token})
    resp.set_cookie(
        "karllo_session",
        token,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 12,
        path="/",
    )
    return resp


@app.post("/api/auth/logout")
def logout(token: str = Depends(owner_required)) -> dict[str, Any]:
    revoke_session(token)
    return {"ok": True}


@app.get("/api/me")
def me(_: str = Depends(owner_required)) -> dict[str, Any]:
    return {"role": "owner", "product": "Karllo AI"}


@app.get("/api/hardware")
def hardware(_: str = Depends(owner_required)) -> dict[str, Any]:
    return hardware_report()


@app.get("/api/model")
def model_status(_: str = Depends(owner_required)) -> dict[str, Any]:
    if ENGINE.model is None:
        _try_load_engine()
    active = read_active()
    status = ENGINE.status()
    cfg = KarlloConfig.small()
    return {
        **status,
        "active": active,
        "presets": {
            "tiny": KarlloConfig.tiny().to_dict() | {"est_params": KarlloConfig.tiny().estimated_parameters()},
            "small": cfg.to_dict() | {"est_params": cfg.estimated_parameters()},
            "medium": KarlloConfig.medium().to_dict()
            | {"est_params": KarlloConfig.medium().estimated_parameters()},
        },
        "distinctions": {
            "training": "Updates model weights via gradient descent.",
            "knowledge": "Retrieves passages; does not change weights.",
            "memory": "Stores user-approved notes; does not change weights.",
            "research": "Fetches public pages when a provider is configured; does not change weights.",
        },
    }


# --- conversations / chat ---


@app.get("/api/conversations")
def conv_list(_: str = Depends(owner_required)) -> list[dict[str, Any]]:
    return list_conversations()


@app.post("/api/conversations")
def conv_create(_: str = Depends(owner_required)) -> dict[str, Any]:
    return create_conversation()


@app.get("/api/conversations/{cid}")
def conv_get(cid: str, _: str = Depends(owner_required)) -> dict[str, Any]:
    try:
        return get_conversation(cid)
    except FileNotFoundError:
        raise HTTPException(404, "Conversation not found")


@app.patch("/api/conversations/{cid}")
def conv_rename(cid: str, body: RenameIn, _: str = Depends(owner_required)) -> dict[str, Any]:
    try:
        return rename_conversation(cid, body.title)
    except FileNotFoundError:
        raise HTTPException(404, "Conversation not found")


@app.delete("/api/conversations/{cid}")
def conv_delete(cid: str, confirm: bool = False, _: str = Depends(owner_required)):
    if not confirm:
        raise HTTPException(400, "Pass confirm=true to delete")
    delete_conversation(cid)
    return {"ok": True}


@app.post("/api/chat/stop/{cid}")
def chat_stop(cid: str, _: str = Depends(owner_required)) -> dict[str, Any]:
    ENGINE.request_stop(cid)
    return {"ok": True}


@app.post("/api/chat")
def chat(body: ChatIn, _: str = Depends(owner_required)) -> StreamingResponse:
    if not body.content.strip():
        raise HTTPException(400, "Empty message")
    if ENGINE.model is None:
        _try_load_engine()
    cid = body.conversation_id
    if not cid:
        rec = create_conversation()
        cid = rec["id"]
    else:
        try:
            rec = get_conversation(cid)
        except FileNotFoundError:
            rec = create_conversation()
            cid = rec["id"]

    user_msg = {"role": "user", "content": body.content.strip(), "ts": time.time()}
    rec = append_messages(cid, [user_msg])

    prefix_parts: list[str] = []
    sources: list[dict[str, Any]] = []
    research_results: list[dict[str, Any]] = []
    if body.use_memory:
        mem = enabled_prefix()
        if mem:
            prefix_parts.append(mem)
    if body.use_knowledge:
        p, hits = KB.prefix_for_prompt(body.content)
        if p:
            prefix_parts.append(p)
            sources = hits
    if body.use_research:
        rs = search_web(body.content)
        if rs.get("ok"):
            research_results = rs.get("results") or []
            rp = research_prefix(research_results)
            if rp:
                prefix_parts.append(rp)
        elif not rs.get("configured"):
            research_results = []
            prefix_parts.append("Web research is not configured. No pages were fetched.")

    settings = GenerationSettings(
        max_new_tokens=max(8, min(body.max_new_tokens, 200)),
        temperature=max(0.0, min(body.temperature, 1.5)),
        top_k=max(0, min(body.top_k, 100)),
    )
    messages = [{"role": m["role"], "content": m["content"]} for m in rec["messages"]]
    prefix = "\n\n".join(prefix_parts)
    model_status_now = ENGINE.status()

    def event_stream():
        meta = {
            "conversation_id": cid,
            "model": model_status_now,
            "sources": sources,
            "research": research_results,
            "research_configured": research_status()["configured"],
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
        pieces: list[str] = []
        try:
            for piece in ENGINE.stream(
                messages, settings=settings, prefix=prefix, conversation_id=cid
            ):
                pieces.append(piece)
                yield f"event: token\ndata: {json.dumps({'t': piece})}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
            return
        text = "".join(pieces)
        assistant = {
            "role": "assistant",
            "content": text,
            "ts": time.time(),
            "sources": sources,
            "research": research_results,
            "untrained": not model_status_now.get("trained"),
        }
        append_messages(cid, [assistant])
        yield f"event: done\ndata: {json.dumps(assistant)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- datasets ---


@app.get("/api/datasets")
def datasets_list(_: str = Depends(owner_required)) -> list[dict[str, Any]]:
    return list_datasets()


@app.post("/api/datasets/upload")
async def datasets_upload(
    file: UploadFile = File(...),
    license_note: str = Form(""),
    _: str = Depends(owner_required),
) -> dict[str, Any]:
    filename = file.filename or "upload.txt"
    if not allowed_upload_name(filename):
        raise HTTPException(400, "Only .txt, .jsonl, .json files are accepted")
    ensure_dirs()
    dest = UPLOADS_DIR / f"{uuid.uuid4().hex[:8]}-{Path(filename).name}"
    size = 0
    with dest.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 64)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(400, "Upload exceeds 20 MB web limit. Use the notebook for larger corpora.")
            out.write(chunk)
    tok = None
    try:
        from karllo.model.checkpoints import read_active
        from karllo.model.tokenizer import KarlloTokenizer as KT

        pointer = read_active()
        if pointer:
            tok_path = Path(pointer["path"]) / "tokenizer.json"
            if tok_path.exists():
                tok = KT.load(tok_path)
    except Exception:
        tok = None
    analysis = analyze_dataset(dest, tokenizer=tok)
    record = new_dataset_record(
        name=Path(filename).stem,
        file_path=dest,
        source="upload",
        analysis=analysis,
        license_note=license_note,
    )
    dest.unlink(missing_ok=True)
    return record


@app.post("/api/datasets/seed")
def datasets_seed(_: str = Depends(owner_required)) -> dict[str, Any]:
    """Register bundled pilot datasets. Original public-domain-style educational text."""
    created = []
    for path in sorted(SEED_DIR.glob("*")):
        if path.suffix.lower() not in {".txt", ".jsonl", ".json"}:
            continue
        existing = [d for d in list_datasets() if Path(d["file"]).name == path.name]
        if existing:
            created.append(existing[0])
            continue
        analysis = analyze_dataset(path)
        rec = new_dataset_record(
            name=path.stem,
            file_path=path,
            source="seed",
            analysis=analysis,
            license_note="Bundled Karllo pilot data. Original educational text.",
        )
        created.append(rec)
    return {"datasets": created}


@app.get("/api/datasets/{ds_id}")
def datasets_get(ds_id: str, _: str = Depends(owner_required)) -> dict[str, Any]:
    try:
        return get_dataset(ds_id)
    except FileNotFoundError:
        raise HTTPException(404, "Dataset not found")
    except ValueError:
        raise HTTPException(400, "Invalid id")


@app.delete("/api/datasets/{ds_id}")
def datasets_delete(ds_id: str, confirm: bool = False, _: str = Depends(owner_required)):
    if not confirm:
        raise HTTPException(400, "Pass confirm=true to delete this dataset")
    try:
        rec = get_dataset(ds_id)
    except FileNotFoundError:
        raise HTTPException(404, "Dataset not found")
    dest = DATASETS_DIR / rec["id"]
    if dest.exists():
        shutil.rmtree(dest)
    return {"ok": True}


@app.post("/api/datasets/hf")
def datasets_hf(_: str = Depends(owner_required)) -> dict[str, Any]:
    return {
        "ok": False,
        "configured": False,
        "error": (
            "Hugging Face hub import is not enabled in this environment. "
            "Export a dataset to JSONL locally and upload it. Respect dataset licenses."
        ),
    }


# --- training ---


def _active_run() -> Optional[dict[str, Any]]:
    if not RUNS_DIR.exists():
        return None
    latest = None
    latest_mtime = 0.0
    for child in RUNS_DIR.iterdir():
        st = child / "status.json"
        if st.exists() and st.stat().st_mtime >= latest_mtime:
            latest_mtime = st.stat().st_mtime
            latest = json.loads(st.read_text(encoding="utf-8"))
            latest["run_id"] = child.name
            latest["run_dir"] = str(child)
    return latest


@app.get("/api/training")
def training_status(_: str = Depends(owner_required)) -> dict[str, Any]:
    run = _active_run()
    hw = hardware_report()
    active = read_active()
    logs = []
    if run and run.get("run_dir"):
        logp = Path(run["run_dir"]) / "log.jsonl"
        if logp.exists():
            lines = logp.read_text(encoding="utf-8").splitlines()[-80:]
            for line in lines:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    logs.append({"raw": line})
    pid_alive = False
    if run and run.get("run_id") in RUNNING:
        pid_alive = RUNNING[run["run_id"]].poll() is None
    return {
        "run": run,
        "hardware": hw,
        "active_checkpoint": active,
        "logs": logs,
        "worker_alive": pid_alive,
        "modes": list(MODES),
    }


@app.post("/api/training/start")
def training_start(body: TrainIn, _: str = Depends(owner_required)) -> dict[str, Any]:
    if body.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    # refuse overlapping jobs
    for rid, proc in list(RUNNING.items()):
        if proc.poll() is None:
            raise HTTPException(409, f"Training run {rid} is still running")
        RUNNING.pop(rid, None)
    try:
        ds = get_dataset(body.dataset_id)
    except FileNotFoundError:
        raise HTTPException(404, "Dataset not found")
    if body.mode != "pretrain":
        if not body.source_checkpoint:
            pointer = read_active()
            if not pointer:
                raise HTTPException(400, "No source checkpoint. Train from scratch first or select one.")
            body.source_checkpoint = pointer["path"]
        if not Path(body.source_checkpoint).exists():
            raise HTTPException(400, "source_checkpoint path does not exist")
    if body.mode == "pretrain":
        body.source_checkpoint = None
        body.train_tokenizer = True

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out = CHECKPOINTS_DIR / "runs" / run_id
    ckpt_out.mkdir(parents=True, exist_ok=True)

    hw = hardware_report()
    size_vocab = {"tiny": 512, "small": 4096, "medium": 8192}
    size_ctx = {"tiny": 64, "small": 256, "medium": 512}
    vocab = body.vocab_size if body.mode == "pretrain" else 4096
    if body.mode == "pretrain" and body.vocab_size in (4096, 512) and body.model_size in size_vocab:
        vocab = size_vocab[body.model_size]
    npos = body.n_positions or size_ctx.get(body.model_size)
    tcfg = TrainConfig(
        mode=body.mode,
        dataset_file=ds["file"],
        run_dir=str(run_dir),
        checkpoint_out=str(ckpt_out),
        source_checkpoint=body.source_checkpoint,
        model_size=body.model_size,
        vocab_size=vocab,
        n_positions=npos,
        lr=body.lr,
        batch_size=body.batch_size,
        grad_accum=body.grad_accum,
        max_steps=min(max(body.max_steps, 1), 5000),
        ckpt_every=max(body.ckpt_every, 1),
        device=hw["device"],
        max_examples=body.max_examples,
        train_tokenizer=body.train_tokenizer if body.mode == "pretrain" else False,
    )
    cfg_path = run_dir / "train_config.json"
    cfg_path.write_text(json.dumps(tcfg.__dict__, indent=2), encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "state": "queued",
                "mode": body.mode,
                "complete": False,
                "dataset_id": body.dataset_id,
                "dataset_name": ds.get("name"),
                "run_id": run_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    env["KARLLO_ROOT"] = str(ROOT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "karllo.training.worker", "--config", str(cfg_path)],
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
        stdout=(run_dir / "stdout.log").open("w"),
        stderr=subprocess.STDOUT,
    )
    RUNNING[run_id] = proc
    return {"ok": True, "run_id": run_id, "pid": proc.pid, "config": tcfg.__dict__}


@app.post("/api/training/stop")
def training_stop(run_id: Optional[str] = None, _: str = Depends(owner_required)):
    run = _active_run()
    rid = run_id or (run or {}).get("run_id")
    if not rid:
        raise HTTPException(404, "No training run")
    stop_path = RUNS_DIR / rid / "STOP"
    stop_path.write_text("stop", encoding="utf-8")
    return {"ok": True, "run_id": rid}


@app.post("/api/training/resume")
def training_resume(_: str = Depends(owner_required)) -> dict[str, Any]:
    return {
        "ok": False,
        "error": (
            "Resume starts a new continue_pretrain or sft job from the latest checkpoint "
            "of a run. Use Start with mode=continue_pretrain and source_checkpoint set to "
            "that run's latest folder. Automatic mid-step resume is supported in the "
            "saved optimizer file but must be launched as a new job."
        ),
    }


# --- checkpoints ---


@app.get("/api/checkpoints")
def ckpt_list(_: str = Depends(owner_required)) -> dict[str, Any]:
    return {"items": list_checkpoints(), "active": read_active()}


@app.post("/api/checkpoints/verify")
def ckpt_verify(body: ActivateIn, _: str = Depends(owner_required)) -> dict[str, Any]:
    path = Path(body.path)
    if not path.exists():
        raise HTTPException(404, "Checkpoint path not found")
    # stay under checkpoints dir
    try:
        path.resolve().relative_to(CHECKPOINTS_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "Checkpoint must live under the checkpoints directory")
    return verify_checkpoint(path)


@app.post("/api/checkpoints/activate")
def ckpt_activate(body: ActivateIn, _: str = Depends(owner_required)) -> dict[str, Any]:
    if not body.confirm:
        raise HTTPException(400, "Set confirm=true after reviewing evaluation. This never overwrites original.")
    path = Path(body.path)
    try:
        path.resolve().relative_to(CHECKPOINTS_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "Checkpoint must live under the checkpoints directory")
    try:
        pointer = set_active(path)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    ENGINE.load_path(path)
    return pointer


# --- knowledge / memory / research ---


@app.get("/api/knowledge")
def knowledge_list(_: str = Depends(owner_required)) -> dict[str, Any]:
    return {"documents": KB.list_documents(), "embeddings": embedding_status()}


@app.post("/api/knowledge")
def knowledge_add(body: KnowledgeTextIn, _: str = Depends(owner_required)) -> dict[str, Any]:
    return KB.add_document(body.name, body.text)


@app.post("/api/knowledge/upload")
async def knowledge_upload(
    file: UploadFile = File(...),
    _: str = Depends(owner_required),
) -> dict[str, Any]:
    filename = file.filename or "doc.txt"
    if Path(filename).suffix.lower() not in {".txt", ".md", ".jsonl"}:
        raise HTTPException(400, "Upload a .txt, .md, or .jsonl document")
    data = await file.read()
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(400, "Document exceeds 5 MB")
    text = data.decode("utf-8", errors="replace")
    return KB.add_document(Path(filename).name, text)


@app.delete("/api/knowledge/{doc_id}")
def knowledge_delete(doc_id: str, confirm: bool = False, _: str = Depends(owner_required)):
    if not confirm:
        raise HTTPException(400, "Pass confirm=true to delete")
    KB.delete_document(doc_id)
    return {"ok": True}


@app.get("/api/knowledge/search")
def knowledge_search(q: str, _: str = Depends(owner_required)) -> dict[str, Any]:
    return KB.search(q)


@app.get("/api/memory")
def memory_list(_: str = Depends(owner_required)) -> list[dict[str, Any]]:
    return list_memories()


@app.post("/api/memory")
def memory_add(body: MemoryIn, _: str = Depends(owner_required)) -> dict[str, Any]:
    return add_memory(body.content)


@app.patch("/api/memory/{mem_id}")
def memory_patch(mem_id: str, body: MemoryPatch, _: str = Depends(owner_required)) -> dict[str, Any]:
    try:
        return update_memory(mem_id, content=body.content, enabled=body.enabled)
    except FileNotFoundError:
        raise HTTPException(404, "Memory not found")


@app.delete("/api/memory/{mem_id}")
def memory_delete(mem_id: str, confirm: bool = False, _: str = Depends(owner_required)):
    if not confirm:
        raise HTTPException(400, "Pass confirm=true to delete")
    try:
        delete_memory(mem_id)
    except FileNotFoundError:
        raise HTTPException(404, "Memory not found")
    return {"ok": True}


@app.get("/api/research/status")
def research_st(_: str = Depends(owner_required)) -> dict[str, Any]:
    return research_status()


@app.post("/api/research")
def research_query(body: ResearchIn, _: str = Depends(owner_required)) -> dict[str, Any]:
    return search_web(body.query)


# --- evaluation ---


@app.post("/api/evaluate")
def evaluate(body: EvalIn, _: str = Depends(owner_required)) -> dict[str, Any]:
    path = Path(body.path)
    try:
        path.resolve().relative_to(CHECKPOINTS_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "Checkpoint must live under the checkpoints directory")
    if body.compare_to:
        other = Path(body.compare_to)
        report = compare_checkpoints(other, path, EVAL_DIR / "test.jsonl")
    else:
        report = evaluate_checkpoint(path, EVAL_DIR / "test.jsonl")
    save_eval_report(report, f"eval-{int(time.time())}")
    return report


@app.get("/api/estimate")
def estimate(
    model_size: str = "tiny",
    tokens: int = 20_000,
    _: str = Depends(owner_required),
) -> dict[str, Any]:
    sizes = {
        "tiny": KarlloConfig.tiny(),
        "small": KarlloConfig.small(),
        "medium": KarlloConfig.medium(),
    }
    cfg = sizes.get(model_size, KarlloConfig.tiny())
    hw = hardware_report()
    return {
        "config": cfg.to_dict(),
        "est_params": cfg.estimated_parameters(),
        "hardware": hw,
        "estimate": estimate_run(cfg.estimated_parameters(), tokens, hw["device"]),
    }


# Static frontend if built
FRONTEND_DIST = ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="ui")
