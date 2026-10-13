import React, { useEffect, useMemo, useState } from "react";
import { NavLink, Route, Routes, useNavigate, useParams } from "react-router-dom";
import { api, apiUrl, getApiBase, login, me, setApiBase } from "./api";
import { Markdown } from "./markdown";

type Conv = { id: string; title: string; updated_at?: number; n_messages?: number };

function IconK() {
  return <div className="logo">K</div>;
}

export default function App() {
  const [authed, setAuthed] = useState<boolean | null>(null);
  const [theme, setTheme] = useState(() => localStorage.getItem("karllo-theme") || "dark");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [convs, setConvs] = useState<Conv[]>([]);
  const [model, setModel] = useState<any>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("karllo-theme", theme);
  }, [theme]);

  async function refresh() {
    try {
      await me();
      setAuthed(true);
      const [c, m] = await Promise.all([api<Conv[]>("/api/conversations"), api("/api/model")]);
      setConvs(c);
      setModel(m);
    } catch (e: any) {
      if (e.status === 401) setAuthed(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  if (authed === null) {
    return (
      <div className="login">
        <div className="login-card">Loading Karllo…</div>
      </div>
    );
  }
  if (!authed) return <Login onOk={() => refresh()} />;

  return (
    <div className="app">
      <div className={`mobile-overlay ${sidebarOpen ? "show" : ""}`} onClick={() => setSidebarOpen(false)} />
      <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="brand">
          <IconK />
          <div>
            <h1>Karllo AI</h1>
            <p>Independent model</p>
          </div>
        </div>
        <nav className="nav">
          <button
            className="linkish"
            onClick={async () => {
              const c = await api<Conv>("/api/conversations", { method: "POST" });
              setSidebarOpen(false);
              window.location.hash = `#/c/${c.id}`;
            }}
          >
            New conversation
          </button>
          <NavLink to="/" end onClick={() => setSidebarOpen(false)}>
            Chat
          </NavLink>
          <NavLink to="/datasets" onClick={() => setSidebarOpen(false)}>
            Datasets
          </NavLink>
          <NavLink to="/training" onClick={() => setSidebarOpen(false)}>
            Training
          </NavLink>
          <NavLink to="/checkpoints" onClick={() => setSidebarOpen(false)}>
            Checkpoints
          </NavLink>
          <NavLink to="/knowledge" onClick={() => setSidebarOpen(false)}>
            Knowledge
          </NavLink>
          <NavLink to="/memory" onClick={() => setSidebarOpen(false)}>
            Memory
          </NavLink>
          <NavLink to="/eval" onClick={() => setSidebarOpen(false)}>
            Evaluation
          </NavLink>
          <NavLink to="/settings" onClick={() => setSidebarOpen(false)}>
            Settings
          </NavLink>
        </nav>
        <div className="conv-list">
          <h3>Conversations</h3>
          {convs.map((c) => (
            <div key={c.id} className="conv-item" style={{ paddingRight: 4 }}>
              <NavLink to={`/c/${c.id}`} onClick={() => setSidebarOpen(false)} style={{ flex: 1, color: "inherit", textDecoration: "none", overflow: "hidden" }}>
                {c.title}
              </NavLink>
              <button
                className="icon-btn"
                title="Rename"
                onClick={async () => {
                  const title = prompt("Rename conversation", c.title);
                  if (!title) return;
                  await api(`/api/conversations/${c.id}`, { method: "PATCH", body: JSON.stringify({ title }) });
                  refresh();
                }}
              >
                ✎
              </button>
              <button
                className="icon-btn"
                title="Delete"
                onClick={async () => {
                  if (!confirm("Delete this conversation?")) return;
                  await api(`/api/conversations/${c.id}?confirm=true`, { method: "DELETE" });
                  refresh();
                }}
              >
                ×
              </button>
            </div>
          ))}
        </div>
        <div className="side-foot">
          <span>{model?.trained ? "Trained checkpoint" : "Untrained init"}</span>
          <button className="ghost" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
            {theme === "dark" ? "Light" : "Dark"}
          </button>
        </div>
      </aside>
      <section className="main">
        <div className="topbar">
          <button className="ghost hamburger" onClick={() => setSidebarOpen(true)}>
            Menu
          </button>
          <h2>Karllo AI</h2>
          <span style={{ color: "var(--muted)", fontSize: 12 }}>
            {model?.device || "cpu"} · {(model?.parameters?.total || 0).toLocaleString()} params
          </span>
        </div>
        <div className="content">
          <Routes>
            <Route path="/" element={<Chat convs={convs} onConvs={refresh} model={model} />} />
            <Route path="/c/:id" element={<Chat convs={convs} onConvs={refresh} model={model} />} />
            <Route path="/datasets" element={<Datasets />} />
            <Route path="/training" element={<Training model={model} />} />
            <Route path="/checkpoints" element={<Checkpoints onChange={refresh} />} />
            <Route path="/knowledge" element={<Knowledge />} />
            <Route path="/memory" element={<Memory />} />
            <Route path="/eval" element={<EvalPage />} />
            <Route path="/settings" element={<Settings model={model} />} />
          </Routes>
        </div>
      </section>
    </div>
  );
}

function Login({ onOk }: { onOk: () => void }) {
  const [password, setPassword] = useState("");
  const [server, setServer] = useState(() => getApiBase());
  const [err, setErr] = useState("");
  const [hint, setHint] = useState("");
  useEffect(() => {
    setApiBase(server);
    api("/api/auth/status")
      .then((s) => setHint(s.hint || ""))
      .catch(() => {});
  }, []);
  return (
    <div className="login">
      <form
        className="login-card"
        onSubmit={async (e) => {
          e.preventDefault();
          setErr("");
          setApiBase(server);
          try {
            await login(password);
            onOk();
          } catch (ex: any) {
            setErr(ex.message === "unauthorized" || ex.status === 401 ? "Invalid password" : ex.message || "Could not reach Karllo server");
          }
        }}
      >
        <div className="logo" style={{ width: 48, height: 48, borderRadius: 14 }}>
          K
        </div>
        <h1>Karllo AI</h1>
        <p className="lead">Owner sign-in. Training and datasets are protected.</p>
        <label className="field">
          Karllo server URL
          <input
            type="text"
            value={server}
            placeholder="Leave empty on the website; APK needs http(s)://host:8000"
            onChange={(e) => setServer(e.target.value)}
          />
        </label>
        <label className="field">
          Password
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoFocus />
        </label>
        {hint && <p className="lead">{hint}</p>}
        {err && <p style={{ color: "var(--danger)" }}>{err}</p>}
        <button className="primary" type="submit" style={{ marginTop: 12, width: "100%" }}>
          Continue
        </button>
      </form>
    </div>
  );
}

function Chat({ convs, onConvs, model }: { convs: Conv[]; onConvs: () => void; model: any }) {
  const { id } = useParams();
  const nav = useNavigate();
  const [rec, setRec] = useState<any>(null);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState("");
  const [busy, setBusy] = useState(false);
  const [useMemory, setUseMemory] = useState(true);
  const [useKnowledge, setUseKnowledge] = useState(false);
  const [useResearch, setUseResearch] = useState(false);
  const [meta, setMeta] = useState<any>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!id) {
      setRec({ messages: [] });
      return;
    }
    api(`/api/conversations/${id}`)
      .then(setRec)
      .catch(() => setRec({ messages: [] }));
  }, [id]);

  async function send(text?: string) {
    const content = (text ?? input).trim();
    if (!content || busy) return;
    setInput("");
    setBusy(true);
    setStreaming("");
    setError("");
    const optimistic = {
      ...(rec || { messages: [] }),
      messages: [...((rec && rec.messages) || []), { role: "user", content }],
    };
    setRec(optimistic);
    try {
      const token = localStorage.getItem("karllo-token");
      const res = await fetch(apiUrl("/api/chat"), {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          conversation_id: id || null,
          content,
          use_memory: useMemory,
          use_knowledge: useKnowledge,
          use_research: useResearch,
        }),
      });
      if (!res.ok || !res.body) throw new Error("Chat request failed");
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      let acc = "";
      let convId = id;
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const chunks = buf.split("\n\n");
        buf = chunks.pop() || "";
        for (const chunk of chunks) {
          const ev = chunk.match(/^event: (\w+)/m)?.[1];
          const dataLine = chunk.split("\n").find((l) => l.startsWith("data: "));
          if (!ev || !dataLine) continue;
          const data = JSON.parse(dataLine.slice(6));
          if (ev === "meta") {
            setMeta(data);
            convId = data.conversation_id;
            if (!id) nav(`/c/${convId}`);
          } else if (ev === "token") {
            acc += data.t;
            setStreaming(acc);
          } else if (ev === "error") {
            setError(data.error);
          } else if (ev === "done") {
            setStreaming("");
            if (convId) {
              const fresh = await api(`/api/conversations/${convId}`);
              setRec(fresh);
            }
            onConvs();
          }
        }
      }
    } catch (e: any) {
      setError(e.message || "Generation failed");
    } finally {
      setBusy(false);
    }
  }

  async function stop() {
    if (id) await api(`/api/chat/stop/${id}`, { method: "POST" });
  }

  async function regen() {
    const msgs = rec?.messages || [];
    const lastUser = [...msgs].reverse().find((m: any) => m.role === "user");
    if (!lastUser) return;
    await send(lastUser.content);
  }

  const messages = rec?.messages || [];
  const trained = model?.trained;

  return (
    <div className="chat">
      <div className="messages">
        <div className="thread">
          {!trained && (
            <div className="banner">
              The active checkpoint is not a trained language model. Responses are next-token samples from
              random or barely-trained weights — not an intelligent assistant. Run Training, then activate a
              verified checkpoint.
            </div>
          )}
          {messages.length === 0 && !streaming && (
            <div className="empty">
              <h3>Talk to Karllo</h3>
              <p>Uses the selected Karllo checkpoint only. No hidden hosted chatbot.</p>
              <div className="suggestions">
                {[
                  "What is the difference between training and retrieval?",
                  "Explain what an untrained checkpoint can do.",
                  "Name the planets in the solar system.",
                  "How do I start a tiny pretraining run?",
                ].map((s) => (
                  <button key={s} className="sug" onClick={() => send(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m: any, i: number) => (
            <div key={i} className={`msg ${m.role}`}>
              <div className="avatar">{m.role === "assistant" ? "K" : "You"}</div>
              <div>
                <div className="bubble">
                  <Markdown text={m.content || ""} />
                </div>
                {(m.sources?.length || m.research?.length) && (
                  <div className="sources">
                    Retrieved evidence (not model weights)
                    {(m.sources || []).map((s: any, j: number) => (
                      <div key={j}>
                        {s.doc_name}: {s.text?.slice(0, 180)}
                      </div>
                    ))}
                    {(m.research || []).map((s: any, j: number) => (
                      <a key={j} href={s.url} target="_blank" rel="noreferrer">
                        {s.title}
                      </a>
                    ))}
                  </div>
                )}
                {m.role === "assistant" && i === messages.length - 1 && (
                  <div className="actions">
                    <button
                      className="ghost"
                      onClick={() => navigator.clipboard.writeText(m.content || "")}
                    >
                      Copy
                    </button>
                    <button className="ghost" onClick={regen} disabled={busy}>
                      Regenerate
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
          {streaming && (
            <div className="msg assistant">
              <div className="avatar">K</div>
              <div className="bubble">
                <Markdown text={streaming} />
                <span className="cursor" />
              </div>
            </div>
          )}
          {error && <div className="banner">{error}</div>}
          {meta?.research && !meta.research_configured && useResearch && (
            <div className="banner info">Web research is not configured. No pages were fetched.</div>
          )}
        </div>
      </div>
      <div className="composer-wrap">
        <div className="composer">
          <textarea
            value={input}
            placeholder="Message Karllo…"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
          />
          <div className="composer-row">
            <div className="toggles">
              <button className={`chip ${useMemory ? "on" : ""}`} onClick={() => setUseMemory(!useMemory)}>
                Memory
              </button>
              <button className={`chip ${useKnowledge ? "on" : ""}`} onClick={() => setUseKnowledge(!useKnowledge)}>
                Knowledge
              </button>
              <button className={`chip ${useResearch ? "on" : ""}`} onClick={() => setUseResearch(!useResearch)}>
                Web research
              </button>
            </div>
            {busy ? (
              <button className="danger" onClick={stop}>
                Stop
              </button>
            ) : (
              <button className="primary" onClick={() => send()} disabled={!input.trim()}>
                Send
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Datasets() {
  const [items, setItems] = useState<any[]>([]);
  const [err, setErr] = useState("");
  const [preview, setPreview] = useState<any>(null);
  const load = () => api("/api/datasets").then(setItems);
  useEffect(() => {
    load();
  }, []);
  return (
    <div className="page">
      <h2>Dataset manager</h2>
      <p className="lead">
        Upload TXT, JSONL, or JSON. Knowledge upload is not training. Respect licenses and privacy. Web
        uploads are capped at 20 MB.
      </p>
      <div className="row" style={{ marginBottom: 16 }}>
        <input
          type="file"
          accept=".txt,.jsonl,.json"
          onChange={async (e) => {
            const f = e.target.files?.[0];
            if (!f) return;
            const fd = new FormData();
            fd.append("file", f);
            fd.append("license_note", "Uploaded by owner");
            try {
              await api("/api/datasets/upload", { method: "POST", body: fd });
              load();
            } catch (ex: any) {
              setErr(ex.message);
            }
          }}
        />
        <button className="primary" onClick={() => api("/api/datasets/seed", { method: "POST" }).then(load)}>
          Load pilot datasets
        </button>
        <button
          className="ghost"
          onClick={async () => {
            const r = await api("/api/datasets/hf", { method: "POST" });
            setErr(r.error);
          }}
        >
          Hugging Face import
        </button>
      </div>
      {err && <div className="banner">{err}</div>}
      <table className="table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Examples</th>
            <th>Tokens</th>
            <th>Suggested mode</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {items.map((d) => (
            <tr key={d.id}>
              <td>{d.name}</td>
              <td>{d.analysis?.example_count}</td>
              <td>
                {d.analysis?.tokens} <span style={{ color: "var(--faint)" }}>{d.analysis?.token_note}</span>
              </td>
              <td>{d.analysis?.suggested_mode}</td>
              <td className="row">
                <button className="ghost" onClick={() => setPreview(d)}>
                  Preview
                </button>
                <button
                  className="ghost"
                  onClick={async () => {
                    if (!confirm("Delete this dataset?")) return;
                    await api(`/api/datasets/${d.id}?confirm=true`, { method: "DELETE" });
                    load();
                  }}
                >
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {preview && (
        <div className="card" style={{ marginTop: 16 }}>
          <h3>Preview — {preview.name}</h3>
          <p className="lead">
            empty {preview.analysis?.counts?.empty} · malformed {preview.analysis?.counts?.malformed} ·
            duplicates {preview.analysis?.counts?.duplicates} · {Math.round((preview.analysis?.bytes || 0) / 1024)} KB
          </p>
          {(preview.analysis?.preview || []).map((p: any, i: number) => (
            <pre key={i} className="logs">
              [{p.kind}] {p.excerpt}
            </pre>
          ))}
        </div>
      )}
    </div>
  );
}

function Training({ model }: { model: any }) {
  const [st, setSt] = useState<any>(null);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [form, setForm] = useState({
    mode: "pretrain",
    dataset_id: "",
    model_size: "tiny",
    max_steps: 30,
    batch_size: 2,
    lr: 0.003,
  });
  const [err, setErr] = useState("");
  const load = () => {
    api("/api/training").then(setSt);
    api("/api/datasets").then(setDatasets);
  };
  useEffect(() => {
    load();
    const t = setInterval(load, 1500);
    return () => clearInterval(t);
  }, []);
  const run = st?.run;
  const hist = run?.loss_history || [];
  const path = useMemo(() => {
    if (hist.length < 2) return "";
    const xs = hist.map((h: any, i: number) => [i, h.train_loss]);
    const maxL = Math.max(...xs.map((p: any) => p[1]));
    const minL = Math.min(...xs.map((p: any) => p[1]));
    const pts = xs
      .map((p: any, i: number) => {
        const x = (i / (xs.length - 1)) * 300;
        const y = 140 - ((p[1] - minL) / (maxL - minL || 1)) * 120;
        return `${x},${y}`;
      })
      .join(" ");
    return pts;
  }, [hist]);

  return (
    <div className="page">
      <h2>Training dashboard</h2>
      <p className="lead">
        Real gradient descent on Karllo weights. Knowledge, memory, and chat history are not training.
        A run is complete only when the worker finishes successfully.
      </p>
      <div className="grid">
        <div className="card">
          <h3>Architecture</h3>
          <div className="stat">{model?.config?.architecture || "karllo-decoder-v1"}</div>
          <div>{(model?.parameters?.total || 0).toLocaleString()} parameters</div>
        </div>
        <div className="card">
          <h3>Hardware</h3>
          <div className="stat">{st?.hardware?.device || "cpu"}</div>
          <div>{st?.hardware?.note}</div>
        </div>
        <div className="card">
          <h3>Run state</h3>
          <div className="stat">{run?.state || "idle"}</div>
          <div>
            step {run?.step || 0}/{run?.max_steps || "—"} · complete {String(!!run?.complete)}
          </div>
        </div>
        <div className="card">
          <h3>Loss</h3>
          <div className="stat">{run?.train_loss?.toFixed?.(4) ?? "—"}</div>
          <div>val {run?.val_loss?.toFixed?.(4) ?? "—"} · lr {run?.lr ?? "—"}</div>
        </div>
      </div>
      {path && (
        <div className="card" style={{ marginTop: 12 }}>
          <h3>Train loss</h3>
          <svg viewBox="0 0 300 160" className="chart">
            <polyline fill="none" stroke="#5eead4" strokeWidth="2" points={path} />
          </svg>
        </div>
      )}
      <div className="card" style={{ marginTop: 16 }}>
        <h3>Start a run</h3>
        <div className="row" style={{ marginTop: 8 }}>
          <label className="field">
            Mode
            <select value={form.mode} onChange={(e) => setForm({ ...form, mode: e.target.value })}>
              <option value="pretrain">Pretrain from scratch</option>
              <option value="continue_pretrain">Continued pretraining</option>
              <option value="sft">Supervised fine-tuning</option>
            </select>
          </label>
          <label className="field">
            Dataset
            <select value={form.dataset_id} onChange={(e) => setForm({ ...form, dataset_id: e.target.value })}>
              <option value="">Select…</option>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.analysis?.example_count})
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            Size
            <select value={form.model_size} onChange={(e) => setForm({ ...form, model_size: e.target.value })}>
              <option value="tiny">tiny (CPU tests)</option>
              <option value="small">small</option>
              <option value="medium">medium (prefer GPU)</option>
            </select>
          </label>
          <label className="field">
            Steps
            <input
              type="number"
              value={form.max_steps}
              onChange={(e) => setForm({ ...form, max_steps: Number(e.target.value) })}
            />
          </label>
          <label className="field">
            Batch
            <input
              type="number"
              value={form.batch_size}
              onChange={(e) => setForm({ ...form, batch_size: Number(e.target.value) })}
            />
          </label>
          <label className="field">
            LR
            <input type="number" step="0.0001" value={form.lr} onChange={(e) => setForm({ ...form, lr: Number(e.target.value) })} />
          </label>
        </div>
        {err && <div className="banner">{err}</div>}
        <div className="row" style={{ marginTop: 12 }}>
          <button
            className="primary"
            onClick={async () => {
              setErr("");
              try {
                await api("/api/training/start", { method: "POST", body: JSON.stringify(form) });
                load();
              } catch (e: any) {
                setErr(e.message);
              }
            }}
          >
            Start
          </button>
          <button className="danger" onClick={() => api("/api/training/stop", { method: "POST" })}>
            Stop
          </button>
          <button
            className="ghost"
            onClick={async () => {
              const r = await api("/api/training/resume", { method: "POST" });
              setErr(r.error);
            }}
          >
            Resume help
          </button>
        </div>
        <p className="lead" style={{ marginTop: 10 }}>
          Tokens processed: {run?.tokens_processed ?? 0}. Checkpoint saved: {String(!!run?.checkpoint_saved)}.
          Weights changed: {String(run?.trained_weights_changed)}. Worker alive: {String(!!st?.worker_alive)}.
        </p>
      </div>
      <div className="card" style={{ marginTop: 12 }}>
        <h3>Logs</h3>
        <div className="logs">
          {(st?.logs || [])
            .map((l: any) => JSON.stringify(l))
            .join("\n") || "No logs yet."}
        </div>
        {run?.error && <div className="banner">{run.error}</div>}
      </div>
    </div>
  );
}

function Checkpoints({ onChange }: { onChange: () => void }) {
  const [data, setData] = useState<any>({ items: [] });
  const [msg, setMsg] = useState("");
  const load = () => api("/api/checkpoints").then(setData);
  useEffect(() => {
    load();
  }, []);
  return (
    <div className="page">
      <h2>Checkpoints</h2>
      <p className="lead">
        Original init is never overwritten. Activate only after verify + evaluation. Random init is not a
        pretrained assistant.
      </p>
      {msg && <div className="banner info">{msg}</div>}
      <table className="table">
        <thead>
          <tr>
            <th>Kind</th>
            <th>Trained</th>
            <th>Mode</th>
            <th>Step</th>
            <th>Notes</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {(data.items || []).map((c: any) => (
            <tr key={c.path}>
              <td>{c.kind}</td>
              <td>{String(c.trained)}</td>
              <td>{c.training_mode || "—"}</td>
              <td>{c.step}</td>
              <td>{c.notes}</td>
              <td className="row">
                <button
                  className="ghost"
                  onClick={async () => {
                    const r = await api("/api/checkpoints/verify", {
                      method: "POST",
                      body: JSON.stringify({ path: c.path }),
                    });
                    setMsg(r.ok ? `Verified ${c.path}` : `Issues: ${(r.issues || []).join(", ")}`);
                  }}
                >
                  Verify
                </button>
                <button
                  className="primary"
                  onClick={async () => {
                    if (!confirm("Activate this checkpoint for chat? Original weights stay on disk.")) return;
                    await api("/api/checkpoints/activate", {
                      method: "POST",
                      body: JSON.stringify({ path: c.path, confirm: true }),
                    });
                    setMsg("Activated " + c.path);
                    onChange();
                    load();
                  }}
                >
                  Activate
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="lead">Active: {data.active?.path || "none"}</p>
    </div>
  );
}

function Knowledge() {
  const [data, setData] = useState<any>({ documents: [], embeddings: {} });
  const [name, setName] = useState("notes");
  const [text, setText] = useState("");
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<any>(null);
  const load = () => api("/api/knowledge").then(setData);
  useEffect(() => {
    load();
  }, []);
  return (
    <div className="page">
      <h2>Knowledge retrieval</h2>
      <p className="lead">
        Optional document search. {data.embeddings?.label}. Retrieval does not train Karllo.
      </p>
      <div className="card">
        <label className="field">
          Name
          <input value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        <label className="field" style={{ marginTop: 8 }}>
          Text or Question:/Answer: book
          <textarea className="field" rows={6} value={text} onChange={(e) => setText(e.target.value)} />
        </label>
        <div className="row" style={{ marginTop: 8 }}>
          <button
            className="primary"
            onClick={async () => {
              await api("/api/knowledge", { method: "POST", body: JSON.stringify({ name, text }) });
              setText("");
              load();
            }}
          >
            Add document
          </button>
          <input
            type="file"
            accept=".txt,.md,.jsonl"
            onChange={async (e) => {
              const f = e.target.files?.[0];
              if (!f) return;
              const fd = new FormData();
              fd.append("file", f);
              await api("/api/knowledge/upload", { method: "POST", body: fd });
              load();
            }}
          />
        </div>
      </div>
      <table className="table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Chunks</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {(data.documents || []).map((d: any) => (
            <tr key={d.id}>
              <td>{d.name}</td>
              <td>{d.n_chunks}</td>
              <td>
                <button
                  className="ghost"
                  onClick={async () => {
                    if (!confirm("Delete this document from the knowledge base?")) return;
                    await api(`/api/knowledge/${d.id}?confirm=true`, { method: "DELETE" });
                    load();
                  }}
                >
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="row" style={{ marginTop: 16 }}>
        <input value={q} placeholder="Search…" onChange={(e) => setQ(e.target.value)} />
        <button className="ghost" onClick={async () => setHits(await api(`/api/knowledge/search?q=${encodeURIComponent(q)}`))}>
          Search
        </button>
      </div>
      {hits && (
        <div className="card" style={{ marginTop: 12 }}>
          <h3>{hits.method?.label}</h3>
          {(hits.hits || []).map((h: any) => (
            <p key={h.id}>
              <strong>{h.doc_name}</strong> ({h.score}): {h.text}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

function Memory() {
  const [items, setItems] = useState<any[]>([]);
  const [content, setContent] = useState("");
  const load = () => api("/api/memory").then(setItems);
  useEffect(() => {
    load();
  }, []);
  return (
    <div className="page">
      <h2>Conversation memory</h2>
      <p className="lead">
        User-approved notes injected into chat context. Separate from weights and from training datasets.
        Private conversations are not used for training unless you export them.
      </p>
      <div className="row">
        <input
          style={{ flex: 1 }}
          value={content}
          placeholder="Add a preference or project fact…"
          onChange={(e) => setContent(e.target.value)}
        />
        <button
          className="primary"
          onClick={async () => {
            await api("/api/memory", { method: "POST", body: JSON.stringify({ content }) });
            setContent("");
            load();
          }}
        >
          Save memory
        </button>
      </div>
      <table className="table">
        <thead>
          <tr>
            <th>Memory</th>
            <th>On</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {items.map((m) => (
            <tr key={m.id}>
              <td>
                <input
                  defaultValue={m.content}
                  onBlur={async (e) => {
                    await api(`/api/memory/${m.id}`, {
                      method: "PATCH",
                      body: JSON.stringify({ content: e.target.value }),
                    });
                  }}
                  style={{ width: "100%" }}
                />
              </td>
              <td>
                <input
                  type="checkbox"
                  defaultChecked={m.enabled}
                  onChange={async (e) => {
                    await api(`/api/memory/${m.id}`, {
                      method: "PATCH",
                      body: JSON.stringify({ enabled: e.target.checked }),
                    });
                  }}
                />
              </td>
              <td>
                <button
                  className="ghost"
                  onClick={async () => {
                    if (!confirm("Delete this memory?")) return;
                    await api(`/api/memory/${m.id}?confirm=true`, { method: "DELETE" });
                    load();
                  }}
                >
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EvalPage() {
  const [ckpts, setCkpts] = useState<any[]>([]);
  const [path, setPath] = useState("");
  const [compare, setCompare] = useState("");
  const [report, setReport] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    api("/api/checkpoints").then((d) => {
      setCkpts(d.items || []);
      if (d.items?.[0]) setPath(d.items[0].path);
    });
  }, []);
  return (
    <div className="page">
      <h2>Learning and evaluation</h2>
      <p className="lead">
        Fixed eval set in data/eval/test.jsonl. Compare checkpoints, then approve activation manually.
        Unverified web content is never auto-trained.
      </p>
      <div className="row">
        <label className="field">
          Candidate
          <select value={path} onChange={(e) => setPath(e.target.value)}>
            {ckpts.map((c) => (
              <option key={c.path} value={c.path}>
                {c.kind} · {c.path}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          Previous (optional)
          <select value={compare} onChange={(e) => setCompare(e.target.value)}>
            <option value="">None</option>
            {ckpts.map((c) => (
              <option key={c.path} value={c.path}>
                {c.kind} · {c.path}
              </option>
            ))}
          </select>
        </label>
        <button
          className="primary"
          disabled={busy}
          onClick={async () => {
            setBusy(true);
            try {
              const r = await api("/api/evaluate", {
                method: "POST",
                body: JSON.stringify({ path, compare_to: compare || null }),
              });
              setReport(r);
            } finally {
              setBusy(false);
            }
          }}
        >
          {busy ? "Evaluating…" : "Evaluate"}
        </button>
      </div>
      {report && (
        <pre className="logs" style={{ marginTop: 16 }}>
          {JSON.stringify(report, null, 2)}
        </pre>
      )}
    </div>
  );
}

function Settings({ model }: { model: any }) {
  const [hw, setHw] = useState<any>(null);
  const [est, setEst] = useState<any>(null);
  const [research, setResearch] = useState<any>(null);
  const [server, setServer] = useState(() => getApiBase());
  useEffect(() => {
    api("/api/hardware").then(setHw);
    api("/api/research/status").then(setResearch);
    api("/api/estimate?model_size=tiny&tokens=20000").then(setEst);
  }, []);
  return (
    <div className="page">
      <h2>Settings</h2>
      <p className="lead">Karllo does not call Gemini or any hidden hosted chatbot.</p>
      <div className="card" style={{ marginBottom: 16 }}>
        <h3>Karllo server</h3>
        <p className="lead">Required in the Android APK. Leave empty when the UI is served from the same host as the API.</p>
        <div className="row">
          <input
            style={{ flex: 1 }}
            value={server}
            placeholder="http://192.168.1.10:8000"
            onChange={(e) => setServer(e.target.value)}
          />
          <button
            className="primary"
            onClick={() => {
              setApiBase(server);
              window.location.reload();
            }}
          >
            Save
          </button>
        </div>
      </div>
      <div className="grid">
        <div className="card">
          <h3>Active model</h3>
          <pre className="logs">{JSON.stringify(model, null, 2)}</pre>
        </div>
        <div className="card">
          <h3>Hardware (measured)</h3>
          <pre className="logs">{JSON.stringify(hw, null, 2)}</pre>
        </div>
        <div className="card">
          <h3>Web research</h3>
          <pre className="logs">{JSON.stringify(research, null, 2)}</pre>
        </div>
        <div className="card">
          <h3>Tiny run estimate</h3>
          <pre className="logs">{JSON.stringify(est, null, 2)}</pre>
        </div>
      </div>
    </div>
  );
}
