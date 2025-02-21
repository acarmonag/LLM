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
      const response = await fetch(`${API_BASE_URL}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: input.trim(),
          max_length: 100,
          use_gpu: true,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to generate response");
      }

      const data = await response.json();
      if (data.generated_text) {
        const assistantMessage: Message = {
          role: "assistant",
          content: data.generated_text,
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } else {
        throw new Error("No text generated");
      }
    } catch (err: any) {
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
          p: 2,
          display: "flex",
          flexDirection: "column",
          gap: 2,
        }}
      >
        {messages.map((message, index) => (
          <Box
            key={index}
            sx={{
              display: "flex",
              gap: 2,
              alignItems: "flex-start",
              maxWidth: "800px",
              mx: "auto",
              width: "100%",
            }}
          >
            <Avatar
              sx={{
                bgcolor: message.role === "assistant" ? "primary.main" : "secondary.main",
              }}
            >
              {message.role === "assistant" ? <SmartToyIcon /> : <PersonIcon />}
            </Avatar>
            <Paper
              sx={{
                p: 2,
                flex: 1,
                bgcolor: "background.paper",
                borderRadius: 2,
              }}
            >
              <Typography
                sx={{
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                }}
              >
                {message.content}
              </Typography>
            </Paper>
          </Box>
        ))}
        {isLoading && (
          <Box
            sx={{
              display: "flex",
              justifyContent: "center",
              p: 2,
            }}
          >
            <CircularProgress />
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
          p: 2,
        }}
      >
        <Box
          component="form"
          onSubmit={handleSubmit}
          sx={{
            maxWidth: "800px",
            mx: "auto",
            display: "flex",
            gap: 1,
          }}
        >
          <TextField
            fullWidth
            multiline
            maxRows={4}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Escribe tu mensaje..."
            disabled={isLoading}
            variant="outlined"
            sx={{
              "& .MuiOutlinedInput-root": {
                bgcolor: "background.paper",
              },
            }}
          />
          <Button
            type="submit"
            variant="contained"
            disabled={isLoading || !input.trim()}
            sx={{ minWidth: 100 }}
          >
            {isLoading ? <CircularProgress size={24} /> : <SendIcon />}
          </Button>
        </Box>
        {error && (
          <Typography
            color="error"
            align="center"
            sx={{ mt: 1 }}
          >
            {error}
          </Typography>
        )}
      </Box>
    </Box>
  );
};

export default ChatInterface;