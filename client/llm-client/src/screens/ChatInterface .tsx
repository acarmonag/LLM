import { useState, useRef, useEffect, FormEvent } from "react";
import {
  Box,
  TextField,
  Button,
  Paper,
  Typography,
  CircularProgress,
  Avatar,
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import SmartToyIcon from "@mui/icons-material/SmartToy";
import PersonIcon from "@mui/icons-material/Person";

const API_BASE_URL = "http://localhost:8002";

type MessageRole = "assistant" | "user";

interface Message {
  role: MessageRole;
  content: string;
}

const ChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "¡Hola! ¿En qué puedo ayudarte hoy?" },
  ]);
  const [input, setInput] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [embeddings, setEmbeddings] = useState<number[][]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = (): void => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: FormEvent<HTMLFormElement>): Promise<void> => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: Message = { role: "user", content: input.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE_URL}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input.trim(), max_length: 100, use_gpu: true }),
      });
      const data = await res.json();
      
      if (res.ok) {
        const assistantMessage: Message = {
          role: "assistant",
          content: data.generated_text,
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } else {
        throw new Error(data.detail || "Error generating text");
      }
    } catch (err) {
      setError(err.message);
      const errorMessage: Message = {
        role: "assistant",
        content: `Error: ${err.message}`,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGetEmbeddings = async () => {
    setIsLoading(true);
    setError(null);
    setEmbeddings([]);
    
    try {
      const res = await fetch(`${API_BASE_URL}/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: [input], use_gpu: true }),
      });
      const data = await res.json();
      
      if (res.ok) {
        setEmbeddings(data.embeddings);
      } else {
        throw new Error(data.detail || "Error fetching embeddings");
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // ... existing JSX return code ...
  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        bgcolor: "background.default",
        overflow: "hidden",
      }}
    >
      {/* Messages Area */}
      <Box
        sx={{
          flex: 1,
          overflow: "auto",
          display: "flex",
          flexDirection: "column",
          width: "100%",
        }}
      >
        {/* ... existing messages mapping code ... */}
        
        {embeddings.length > 0 && (
          <Box sx={{ p: 2, maxWidth: "800px", mx: "auto", width: "100%" }}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6">Embeddings:</Typography>
              <Typography
                component="pre"
                sx={{
                  overflowX: "auto",
                  whiteSpace: "pre-wrap",
                  fontSize: "0.875rem",
                }}
              >
                {JSON.stringify(embeddings, null, 2)}
              </Typography>
            </Paper>
          </Box>
        )}
        
        <div ref={messagesEndRef} />
      </Box>

      {/* Input Area */}
      <Box
        sx={{
          borderTop: 1,
          borderColor: "divider",
          bgcolor: "background.paper",
          width: "100%",
        }}
      >
        <Box
          sx={{
            maxWidth: "800px",
            mx: "auto",
            p: 2,
            width: "100%",
          }}
        >
          <Paper
            component="form"
            onSubmit={handleSubmit}
            elevation={1}
            sx={{
              p: 1,
              display: "flex",
              gap: 1,
              width: "100%",
              bgcolor: "background.paper",
              border: 1,
              borderColor: "grey.800",
              "&:hover": {
                borderColor: "grey.700",
              },
            }}
          >
            <TextField
              fullWidth
              multiline
              maxRows={4}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Escribe un mensaje..."
              disabled={isLoading}
              variant="standard"
              InputProps={{
                disableUnderline: true,
                sx: {
                  p: 1,
                  color: "text.primary",
                  "& ::placeholder": {
                    color: "text.secondary",
                    opacity: 1,
                  },
                },
              }}
            />
            <Button
              onClick={handleGetEmbeddings}
              variant="contained"
              disabled={isLoading || !input.trim()}
              sx={{
                minWidth: "40px",
                width: "40px",
                height: "40px",
                borderRadius: "8px",
                mr: 1,
              }}
            >
              <SmartToyIcon />
            </Button>
            <Button
              type="submit"
              variant="contained"
              disabled={isLoading || !input.trim()}
              sx={{
                minWidth: "40px",
                width: "40px",
                height: "40px",
                borderRadius: "8px",
              }}
            >
              <SendIcon />
            </Button>
          </Paper>
          {error && (
            <Typography
              variant="caption"
              sx={{
                display: "block",
                mt: 1,
                color: "error.main",
              }}
            >
              {error}
            </Typography>
          )}
          <Typography
            variant="caption"
            align="center"
            sx={{
              display: "block",
              mt: 1,
              color: "text.secondary",
            }}
          >
            El modelo puede producir información incorrecta.
          </Typography>
        </Box>
      </Box>
    </Box>
  );
};

export default ChatInterface;