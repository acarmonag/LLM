import { ThemeProvider, createTheme } from "@mui/material";
import ChatInterface from "./screens/ChatInterface ";

const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#3b82f6",
    },
    background: {
      default: "#171717",
      paper: "#1f1f1f",
    },
    text: {
      primary: "#ffffff",
      secondary: "#9ca3af",
    },
    divider: "#2d2d2d",
  },
});

function App() {
  return (
    <ThemeProvider theme={darkTheme}>
      <ChatInterface />
    </ThemeProvider>
  );
}

export default App;
