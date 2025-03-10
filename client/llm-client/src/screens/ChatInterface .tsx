import { useState, useRef, useEffect, FormEvent } from "react";
import {
  Box, TextField, Button, Paper, Typography, CircularProgress, Avatar,
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import SmartToyIcon from "@mui/icons-material/SmartToy";
import PersonIcon from "@mui/icons-material/Person";

const API_BASE_URL = "http://localhost:8002";

type MessageRole = "assistant" | "user";

interface Message {
  role: MessageRole;
  content: string;
  isEmbedding?: boolean;
  isGenerate?: boolean;
}

interface MessageContext {
  role: string;
  content: string;
}

interface SimilarCase {
  case: {
    question: string;
    answer: string;
    category: string;
    priority: number;
  };
  similarity: number;
}

interface GenerateResponse {
  generated_text: string;
  similar_cases?: SimilarCase[];
  system_info: any;
}

interface GenerateWithContext {
  text: string;
  context: MessageContext[];
  max_length: number;
  use_gpu: boolean;
}

const ChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "¡Hola!, bienvenido a soporte ¿En qué puedo ayudarte hoy? (Usa /emb para embeddings, /gen para generación directa)" },
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

    const isEmbeddingRequest = input.trim().startsWith('/emb');
    const isGenerateRequest = input.trim().startsWith('/gen');
    const actualInput = isEmbeddingRequest ? input.slice(4).trim() : 
                       isGenerateRequest ? input.slice(4).trim() : 
                       input.trim();

    const userMessage: Message = { 
      role: "user", 
      content: input.trim(),
      isEmbedding: isEmbeddingRequest,
      isGenerate: isGenerateRequest
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setError(null);

    try {
      if (isEmbeddingRequest) {
        if (!actualInput) {
          throw new Error("Please provide text after /emb");
        }
        
        const response = await fetch(`${API_BASE_URL}/embeddings`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            texts: [actualInput],
            use_gpu: true,
          }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || "Failed to generate embeddings");
        }

        const data = await response.json();
        const embeddings = data.embeddings[0];
        const formattedEmbeddings = JSON.stringify(embeddings, null, 2);

        const assistantMessage: Message = {
          role: "assistant",
          content: `Embeddings for "${actualInput}":\n\`\`\`json\n${formattedEmbeddings}\n\`\`\``,
          isEmbedding: true,
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } else if (isGenerateRequest) {
        if (!actualInput) {
          throw new Error("Please provide text after /gen");
        }

        // Get last 5 messages for context
        const contextMessages: MessageContext[] = messages.slice(-5).map(msg => ({
          role: msg.role,
          content: msg.content
        }));

        const generatePayload: GenerateWithContext = {
          text: actualInput,
          context: contextMessages,
          max_length: 100,
          use_gpu: true
        };

        const response = await fetch(`${API_BASE_URL}/generate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(generatePayload),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || "Failed to generate text");
        }

        const data = await response.json();
        if (data.generated_text) {
          const assistantMessage: Message = {
            role: "assistant",
            content: data.generated_text,
            isGenerate: true,
          };
          setMessages((prev) => [...prev, assistantMessage]);
        } else {
          throw new Error("No text generated");
        }
      } else {
        // Support system logic
        const similarResponse = await fetch(`${API_BASE_URL}/get-similar-cases`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: actualInput,
            max_length: 100,
            use_gpu: true,
          }),
        });

        if (!similarResponse.ok) {
          throw new Error("Failed to find similar cases");
        }

        const similarData = await similarResponse.json();
        const mostSimilarCase = similarData.similar_cases[0];

        if (mostSimilarCase && mostSimilarCase.similarity > 0.7) {
          const assistantMessage: Message = {
            role: "assistant",
            content: mostSimilarCase.case.answer,
          };
          setMessages((prev) => [...prev, assistantMessage]);
        } else {
          // Get last 5 messages for context in fallback generation
          const contextMessages: MessageContext[] = messages.slice(-5).map(msg => ({
            role: msg.role,
            content: msg.content
          }));

          const generatePayload: GenerateWithContext = {
            text: actualInput,
            context: contextMessages,
            max_length: 100,
            use_gpu: true
          };

          const response = await fetch(`${API_BASE_URL}/generate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(generatePayload),
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
        }
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
                bgcolor: message.isEmbedding ? "rgba(0, 0, 0, 0.03)" : 
                        message.isGenerate ? "rgba(0, 100, 0, 0.03)" : 
                        "background.paper",
                borderRadius: 2,
                fontFamily: message.isEmbedding ? "monospace" : "inherit",
              }}
            >
              <Typography
                sx={{
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  fontFamily: message.isEmbedding ? "monospace" : "inherit",
                }}
              >
                {message.content}
              </Typography>
            </Paper>
          </Box>
        ))}
        {isLoading && (
          <Box sx={{ display: "flex", justifyContent: "center", p: 2 }}>
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
            placeholder="Escribe tu mensaje... (Usa /emb para embeddings, /gen para generación directa)"
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
          <Typography color="error" align="center" sx={{ mt: 1 }}>
            {error}
          </Typography>
        )}
      </Box>
    </Box>
  );
};

export default ChatInterface;