import { useState, useRef, useEffect, FormEvent } from "react";
import {
  Box, TextField, Button, Paper, Typography, CircularProgress, Avatar,
  Tabs, Tab, Table, TableBody, TableCell, TableHead, TableRow,
  Dialog, DialogTitle, DialogContent, DialogActions, IconButton,
  Chip, Collapse, Divider, Tooltip,
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import SmartToyIcon from "@mui/icons-material/SmartToy";
import PersonIcon from "@mui/icons-material/Person";
import DeleteIcon from "@mui/icons-material/Delete";
import RefreshIcon from "@mui/icons-material/Refresh";
import BarChartIcon from "@mui/icons-material/BarChart";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";

const API_BASE_URL: string = import.meta.env.VITE_API_URL || "http://localhost:8002";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
type MessageRole = "assistant" | "user";

interface Message {
  id: string;
  role: MessageRole;
  content: string;
  streaming?: boolean;
  isEmbedding?: boolean;
  isGenerate?: boolean;
  confidence?: number;
  ragHit?: boolean;
}

interface MessageContext {
  role: string;
  content: string;
}

interface KBCase {
  case_id: string;
  category: string;
  question: string;
  created_at: string;
}

interface AnalyticsData {
  total_queries_today: number;
  rag_hit_rate_pct: number;
  avg_confidence: number;
  top_categories: { category: string; count: number }[];
  ollama_reachable: boolean;
}

// ---------------------------------------------------------------------------
// Helper: generate a stable message id
// ---------------------------------------------------------------------------
function genId() {
  return Math.random().toString(36).slice(2);
}

// ---------------------------------------------------------------------------
// Admin Tab
// ---------------------------------------------------------------------------
function AdminTab() {
  const [password, setPassword] = useState(import.meta.env.VITE_ADMIN_PASSWORD || "");
  const [authed, setAuthed] = useState(false);
  const [cases, setCases] = useState<KBCase[]>([]);
  const [loading, setLoading] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<KBCase | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchCases = async (pw: string) => {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`${API_BASE_URL}/knowledge-base`, {
        headers: { "X-Admin-Password": pw },
      });
      if (r.status === 401) { setError("Wrong password"); setAuthed(false); return; }
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      setCases(data.cases);
      setAuthed(true);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      const r = await fetch(`${API_BASE_URL}/knowledge-base/${deleteTarget.case_id}`, {
        method: "DELETE",
        headers: { "X-Admin-Password": password },
      });
      if (!r.ok) throw new Error(await r.text());
      setCases(prev => prev.filter(c => c.case_id !== deleteTarget.case_id));
    } catch (e: any) {
      setError(e.message);
    } finally {
      setDeleteTarget(null);
    }
  };

  if (!authed) {
    return (
      <Box sx={{ p: 3, maxWidth: 400, mx: "auto" }}>
        <Typography variant="h6" gutterBottom>Admin Login</Typography>
        <TextField
          fullWidth
          type="password"
          label="Admin Password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          onKeyDown={e => e.key === "Enter" && fetchCases(password)}
          sx={{ mb: 2 }}
        />
        {error && <Typography color="error" sx={{ mb: 1 }}>{error}</Typography>}
        <Button variant="contained" onClick={() => fetchCases(password)}>Login</Button>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 2 }}>
      <Box sx={{ display: "flex", alignItems: "center", mb: 2, gap: 1 }}>
        <Typography variant="h6">Knowledge Base ({cases.length} cases)</Typography>
        <Tooltip title="Refresh">
          <IconButton onClick={() => fetchCases(password)} size="small">
            <RefreshIcon />
          </IconButton>
        </Tooltip>
      </Box>
      {error && <Typography color="error" sx={{ mb: 1 }}>{error}</Typography>}
      {loading ? (
        <CircularProgress />
      ) : (
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Category</TableCell>
              <TableCell>Question</TableCell>
              <TableCell>Created</TableCell>
              <TableCell />
            </TableRow>
          </TableHead>
          <TableBody>
            {cases.map(c => (
              <TableRow key={c.case_id}>
                <TableCell><Chip label={c.category} size="small" /></TableCell>
                <TableCell sx={{ maxWidth: 400 }}>{c.question}</TableCell>
                <TableCell sx={{ whiteSpace: "nowrap" }}>
                  {new Date(c.created_at).toLocaleDateString()}
                </TableCell>
                <TableCell>
                  <Tooltip title="Delete">
                    <IconButton size="small" color="error" onClick={() => setDeleteTarget(c)}>
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}

      <Dialog open={!!deleteTarget} onClose={() => setDeleteTarget(null)}>
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <Typography>Delete case: <em>{deleteTarget?.question}</em>?</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteTarget(null)}>Cancel</Button>
          <Button color="error" variant="contained" onClick={handleDelete}>Delete</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Analytics Panel
// ---------------------------------------------------------------------------
function AnalyticsPanel() {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  const fetch_ = async () => {
    setLoading(true);
    try {
      const r = await fetch(`${API_BASE_URL}/analytics`);
      setData(await r.json());
    } catch {
      // silently ignore
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (open && !data) fetch_();
  }, [open]);

  return (
    <Box sx={{ borderTop: 1, borderColor: "divider" }}>
      <Box
        sx={{ display: "flex", alignItems: "center", px: 2, py: 0.5, cursor: "pointer" }}
        onClick={() => setOpen(p => !p)}
      >
        <BarChartIcon sx={{ mr: 1, fontSize: 18 }} />
        <Typography variant="caption" sx={{ flex: 1 }}>Analytics</Typography>
        <IconButton size="small" onClick={(e) => { e.stopPropagation(); fetch_(); }}>
          <RefreshIcon fontSize="small" />
        </IconButton>
        {open ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
      </Box>
      <Collapse in={open}>
        <Box sx={{ px: 2, pb: 2 }}>
          {loading && <CircularProgress size={16} />}
          {data && (
            <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
              <Box>
                <Typography variant="caption" color="text.secondary">Queries today</Typography>
                <Typography variant="h6">{data.total_queries_today}</Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">RAG hit rate</Typography>
                <Typography variant="h6">{data.rag_hit_rate_pct}%</Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">Avg confidence</Typography>
                <Typography variant="h6">{(data.avg_confidence * 100).toFixed(1)}%</Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">Ollama</Typography>
                <Chip
                  size="small"
                  label={data.ollama_reachable ? "Online" : "Offline"}
                  color={data.ollama_reachable ? "success" : "error"}
                />
              </Box>
              {data.top_categories.length > 0 && (
                <Box sx={{ width: "100%" }}>
                  <Typography variant="caption" color="text.secondary">Top categories</Typography>
                  <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mt: 0.5 }}>
                    {data.top_categories.map(tc => (
                      <Chip key={tc.category} size="small" label={`${tc.category} (${tc.count})`} />
                    ))}
                  </Box>
                </Box>
              )}
            </Box>
          )}
        </Box>
      </Collapse>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Chat Tab
// ---------------------------------------------------------------------------
function ChatTab() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: genId(),
      role: "assistant",
      content: "¡Hola!, bienvenido a soporte ¿En qué puedo ayudarte hoy? (Usa /emb para embeddings, /gen para generación directa)",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [useStream, setUseStream] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const appendToken = (msgId: string, token: string) => {
    setMessages(prev =>
      prev.map(m => m.id === msgId ? { ...m, content: m.content + token } : m)
    );
  };

  const finalizeMessage = (msgId: string, meta?: { confidence?: number; ragHit?: boolean }) => {
    setMessages(prev =>
      prev.map(m => m.id === msgId ? { ...m, streaming: false, ...meta } : m)
    );
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!input.trim()) return;

    const isEmbedding = input.trim().startsWith("/emb");
    const isGenerate = input.trim().startsWith("/gen");
    const actualInput = isEmbedding || isGenerate ? input.slice(4).trim() : input.trim();

    const userMsg: Message = { id: genId(), role: "user", content: input.trim() };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);
    setError(null);

    try {
      if (isEmbedding) {
        if (!actualInput) throw new Error("Provide text after /emb");
        const r = await fetch(`${API_BASE_URL}/embeddings`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ texts: [actualInput], use_gpu: false }),
        });
        if (!r.ok) throw new Error((await r.json()).detail || "Failed");
        const data = await r.json();
        setMessages(prev => [...prev, {
          id: genId(),
          role: "assistant",
          content: `Embeddings for "${actualInput}":\n\`\`\`json\n${JSON.stringify(data.embeddings[0], null, 2)}\n\`\`\``,
          isEmbedding: true,
        }]);
        return;
      }

      if (isGenerate) {
        if (!actualInput) throw new Error("Provide text after /gen");
        const contextMessages: MessageContext[] = messages.slice(-5).map(m => ({
          role: m.role, content: m.content,
        }));
        const r = await fetch(`${API_BASE_URL}/generate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: actualInput, context: contextMessages, max_length: 100, use_gpu: false }),
        });
        if (!r.ok) throw new Error((await r.json()).detail || "Failed");
        const data = await r.json();
        setMessages(prev => [...prev, {
          id: genId(), role: "assistant", content: data.generated_text, isGenerate: true,
        }]);
        return;
      }

      // Normal support flow
      if (useStream) {
        const streamMsgId = genId();
        setMessages(prev => [...prev, {
          id: streamMsgId, role: "assistant", content: "", streaming: true,
        }]);

        const r = await fetch(`${API_BASE_URL}/support-stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: actualInput, session_id: sessionId, use_gpu: false }),
        });
        if (!r.ok) throw new Error((await r.json()).detail || "Stream failed");

        const reader = r.body!.getReader();
        const decoder = new TextDecoder();
        let meta: { confidence?: number; ragHit?: boolean } = {};
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";
          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const payload = line.slice(6).trim();
            if (payload === "[DONE]") { finalizeMessage(streamMsgId, meta); break; }
            try {
              const parsed = JSON.parse(payload);
              if (typeof parsed === "object" && parsed.type === "metadata") {
                if (parsed.session_id) setSessionId(parsed.session_id);
                meta = { confidence: parsed.confidence, ragHit: parsed.rag_hit };
              } else if (typeof parsed === "string") {
                appendToken(streamMsgId, parsed);
              }
            } catch {
              // bare token
              appendToken(streamMsgId, payload);
            }
          }
        }
        finalizeMessage(streamMsgId, meta);
      } else {
        // Non-streaming
        const r = await fetch(`${API_BASE_URL}/support`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: actualInput, session_id: sessionId, use_gpu: false }),
        });
        if (!r.ok) throw new Error((await r.json()).detail || "Failed");
        const data = await r.json();
        if (data.session_id) setSessionId(data.session_id);
        setMessages(prev => [...prev, {
          id: genId(),
          role: "assistant",
          content: data.response,
          confidence: data.confidence,
          ragHit: data.rag_hit,
        }]);
      }
    } catch (err: any) {
      setError(err.message);
      setMessages(prev => [...prev, {
        id: genId(), role: "assistant", content: `Error: ${err.message}`,
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Toolbar */}
      <Box sx={{ px: 2, py: 1, display: "flex", gap: 1, alignItems: "center", borderBottom: 1, borderColor: "divider" }}>
        <Button
          size="small"
          variant={useStream ? "contained" : "outlined"}
          onClick={() => setUseStream(p => !p)}
        >
          {useStream ? "Streaming ON" : "Streaming OFF"}
        </Button>
        {sessionId && (
          <Typography variant="caption" color="text.secondary">
            Session: {sessionId.slice(0, 8)}…
          </Typography>
        )}
      </Box>

      {/* Messages */}
      <Box sx={{ flex: 1, overflow: "auto", p: 2, display: "flex", flexDirection: "column", gap: 2 }}>
        {messages.map(msg => (
          <Box key={msg.id} sx={{ display: "flex", gap: 2, alignItems: "flex-start", maxWidth: 800, mx: "auto", width: "100%" }}>
            <Avatar sx={{ bgcolor: msg.role === "assistant" ? "primary.main" : "secondary.main" }}>
              {msg.role === "assistant" ? <SmartToyIcon /> : <PersonIcon />}
            </Avatar>
            <Paper sx={{
              p: 2, flex: 1, borderRadius: 2,
              bgcolor: msg.isEmbedding ? "rgba(0,0,0,0.03)" : msg.isGenerate ? "rgba(0,100,0,0.03)" : "background.paper",
            }}>
              <Typography sx={{ whiteSpace: "pre-wrap", wordBreak: "break-word", fontFamily: msg.isEmbedding ? "monospace" : "inherit" }}>
                {msg.content}
                {msg.streaming && <span style={{ animation: "blink 1s step-end infinite" }}>▋</span>}
              </Typography>
              {msg.confidence !== undefined && !msg.streaming && (
                <Box sx={{ mt: 1, display: "flex", gap: 1 }}>
                  <Chip
                    size="small"
                    label={msg.ragHit ? `RAG ${(msg.confidence * 100).toFixed(0)}%` : "LLM fallback"}
                    color={msg.ragHit ? "success" : "warning"}
                  />
                </Box>
              )}
            </Paper>
          </Box>
        ))}
        {isLoading && !messages.at(-1)?.streaming && (
          <Box sx={{ display: "flex", justifyContent: "center", p: 2 }}>
            <CircularProgress />
          </Box>
        )}
        <div ref={messagesEndRef} />
      </Box>

      {/* Input */}
      <Box sx={{ borderTop: 1, borderColor: "divider", bgcolor: "background.paper", p: 2 }}>
        <Box component="form" onSubmit={handleSubmit} sx={{ maxWidth: 800, mx: "auto", display: "flex", gap: 1 }}>
          <TextField
            fullWidth multiline maxRows={4}
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Escribe tu mensaje... (/emb texto | /gen texto)"
            disabled={isLoading}
            variant="outlined"
          />
          <Button type="submit" variant="contained" disabled={isLoading || !input.trim()} sx={{ minWidth: 56 }}>
            {isLoading ? <CircularProgress size={24} /> : <SendIcon />}
          </Button>
        </Box>
        {error && <Typography color="error" align="center" sx={{ mt: 1 }}>{error}</Typography>}
      </Box>

      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }`}</style>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Root component
// ---------------------------------------------------------------------------
const ChatInterface = () => {
  const [tab, setTab] = useState(0);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", height: "100vh", bgcolor: "background.default", overflow: "hidden" }}>
      <Box sx={{ borderBottom: 1, borderColor: "divider" }}>
        <Tabs value={tab} onChange={(_, v) => setTab(v)}>
          <Tab label="Support Chat" />
          <Tab label="Admin" />
        </Tabs>
      </Box>

      <Box sx={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
        {tab === 0 ? (
          <Box sx={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <Box sx={{ flex: 1, overflow: "hidden" }}>
              <ChatTab />
            </Box>
            <AnalyticsPanel />
          </Box>
        ) : (
          <Box sx={{ flex: 1, overflow: "auto" }}>
            <AdminTab />
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default ChatInterface;
