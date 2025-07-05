import React, { useState, useRef, useEffect, memo } from "react"; 
import {
  Box,
  Typography,
  Paper,
  TextField,
  IconButton,
  CircularProgress,
  InputAdornment,
  CssBaseline,
  createTheme,
  ThemeProvider,
  Divider,
  Fab,
  Tooltip,
} from "@mui/material";
import {
  Send,
  UploadFile,
  Brightness4,
  Brightness7,
  Source,
  Download,
} from "@mui/icons-material";
import MenuBookIcon from "@mui/icons-material/MenuBook";
import { useTheme } from "@mui/material/styles";
import uzhLogoLight from "./assets/uzh-logo-dark.png";
import uzhLogoDark from "./assets/uzh-logo-light.png";
import logoLight from "./assets/SpiritRAG-logo.png";
import logoDark from "./assets/SpiritRAG-logo-dark.png";
import LeftBar from "./LeftBar"; 
import RightBar from "./RightBar";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm"; 
import rehypeRaw from "rehype-raw"; 

// Typewriter effect component
const TypewriterText = ({ text, scrollRef }) => {
  const [displayedText, setDisplayedText] = useState("");

  useEffect(() => {
    setDisplayedText(""); 
    let index = 0;
    const interval = setInterval(() => {
      if (index < text.length) {
        setDisplayedText((prev) => text.slice(0, index + 1)); 
        index++;
        if (scrollRef?.current) {
          scrollRef.current.scrollIntoView({ behavior: "smooth" }); 
        }
      } else {
        clearInterval(interval);
      }
    }, 1); // Adjust the speed of the typewriter effect
    return () => clearInterval(interval);
  }, [text, scrollRef]);

//   useEffect(() => {
//   setDisplayedText(text); // Instantly set the full text
//   if (scrollRef?.current) {
//     scrollRef.current.scrollIntoView({ behavior: "smooth" });
//   }
// }, [text, scrollRef]);

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]} 
      rehypePlugins={[rehypeRaw]} 
    >
      {displayedText}
    </ReactMarkdown>
  );
};

const Message = memo(({ text, sender, isTyping, scrollRef }) => {
  const theme = useTheme();

  return (
    <Paper
      elevation={1}
      sx={{
        p: 2,
        m: 1,
        maxWidth: "75%",
        alignSelf: sender === "user" ? "flex-end" : "flex-start",
        backgroundColor:
          sender === "user"
            ? theme.palette.mode === "dark"
              ? "#1565c0"
              : "#e0f7fa"
            : theme.palette.background.paper,
      }}
    >
      <div style={{ textAlign: "justify" }}>
        {sender === "bot" ? (
          isTyping ? (
            <TypewriterText text={text} scrollRef={scrollRef} />
          ) : (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]} 
              rehypePlugins={[rehypeRaw]} 
              components={{
                p: ({ node, ...props }) => <div {...props} />, 
              }}
            >
              {text}
            </ReactMarkdown>
          )
        ) : (
          <Typography variant="body1">{text}</Typography>
        )}
      </div>
    </Paper>
  );
});

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [mode, setMode] = useState("light");
  const [sources, setSources] = useState([]);
  const [metadata, setMetadata] = useState([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isLeftSidebarOpen, setIsLeftSidebarOpen] = useState(false);
  const [selectedPdf, setSelectedPdf] = useState(null);
  const [highlightedSubjects, setHighlightedSubjects] = useState([]);
  const [showEval, setShowEval] = useState(false);
  const fileInputRef = useRef();
  const messagesEndRef = useRef(null);

  const [userQuery, setUserQuery] = useState(""); 
  const [searchDataset, setSearchDataset] = useState(null);

  const theme = createTheme({
    palette: {
      mode: mode,
    },
  });

  const toggleColorMode = () => {
    setMode((prev) => (prev === "light" ? "dark" : "light"));
  };

  const toggleSidebar = () => {
    setIsSidebarOpen((prev) => !prev);
  };

  const toggleLeftSidebar = () => {
    setIsLeftSidebarOpen((prev) => !prev);
  };

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  };

  const fetchMetadata = async (fileNames) => {
    const metadataList = [];
    for (const fileName of fileNames) {
      const folderName = fileName.split("-parsed.json")[0].replace(/-\w+$/, "");

      const possiblePaths = [
        `/path/to/your/metadata`,
      ];


      let metadata = null;

      for (const path of possiblePaths) {
        try {
          const response = await fetch(
            `/api/metadata?path=${encodeURIComponent(
              path
            )}`
          );
          if (response.ok) {
            const data = await response.json();
            metadataList.push(
              ...data.map((item) => ({ fileName, metadata: item }))
            );
            break;
          }
        } catch (error) {
          console.error(`Error fetching metadata from ${path}:`, error);
        }
      }

      if (!metadata) {
        console.error(`Metadata not found for ${fileName}`);
      }
    }
    setMetadata(metadataList);
  };

  const fetchHighlightedSubjects = async (response, subjects) => {
    try {
      const res = await fetch("/api/semantic_similarity", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ response, subjects }),
      });

      if (res.ok) {
        const data = await res.json();
        const relevantSubjects = data
          .filter((item) => item.similarity > 0.4)
          .map((item) => toCamelCase(item.subject.trim()));
        setHighlightedSubjects(relevantSubjects);
      } else {
        console.error("Failed to fetch highlighted subjects");
      }
    } catch (error) {
      console.error("Error fetching highlighted subjects:", error);
    }
  };

  const handleSend = async () => {
    if (!input.trim() && !file) return;

    setMessages((prev) => [...prev, { text: input, sender: "user" }]);
    setUserQuery(input); // Store the user query
    setInput("");
    setLoading(true);

    const formData = new FormData();
    formData.append("query", input);
    if (file) {
      formData.append("file", file);
    }

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setMessages((prev) => [
          ...prev,
          { text: data.generated_text, sender: "bot" },
        ]);
        setSources(data.source_files || []);
        fetchMetadata(data.source_files || []);
        fetchHighlightedSubjects(input, data.source_files || []);
      } else {
        const errorData = await response.json();
        setMessages((prev) => [
          ...prev,
          { text: "Failed to generate a response.", sender: "bot" },
        ]);
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { text: "An error occurred while generating a response.", sender: "bot" },
      ]);
    } finally {
      setLoading(false);
      scrollToBottom();
    }
  };

  const handleFileChange = (event) => {
    if (event.target.files.length > 0) {
      const selectedFile = event.target.files[0];
      setFile(selectedFile);
    }
  };

  const renderLanguages = (languages, folderName) => {
    const languageMap = {
      ar: "العربية",
      zh: "中文",
      en: "English",
      fr: "Français",
      ru: "Русский",
      es: "Español",
      de: "Deutsch",
      it: "Italiano",
      pt: "Português",
    };

    return languages
      .map((lang) => lang.trim())
      .filter((lang) => languageMap[lang])
      .map((lang, index) => (
        <span
          key={lang}
          style={{
            cursor: "pointer",
            color: "#1C8D96",
            textDecoration: "underline",
            whiteSpace: "nowrap",
          }}
          onClick={() => handleLanguageClick(folderName, lang)}
        >
          {languageMap[lang]}
        </span>
      ));
  };

  const toCamelCase = (text) => {
    const correctedText = text.replace(/\s*:\s*/g, ": ");
    return correctedText
      .toLowerCase()
      .split(" ")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  const renderSubjects = (subjects) => {
    const formattedSubjects = subjects
      .split(",")
      .map((subject) => toCamelCase(subject.trim()));

    return formattedSubjects.map((subject, index) => (
      <span key={index}>
        <span
          style={{
            color: highlightedSubjects.includes(subject)
              ? "#1C8D96"
              : "text.secondary",
          }}
        >
          {subject}
        </span>
        {index < formattedSubjects.length - 1 && ", "}
      </span>
    ));
  };

  const handleLanguageClick = async (folderName, language) => {
    console.log(`Fetching PDF for folder: ${folderName}, language: ${language}`);
    const cleanFolderName = folderName.replace(/-\w+$/, "");
    const possibleDatasets = ["education"];
    let pdfUrl = null;

    for (const folder of possibleDatasets) {
      const url = `/api/pdf/${folder}/crawled_data/${cleanFolderName}/${cleanFolderName}-${language}.pdf`;
      console.log(`Checking URL: ${url}`);
      pdfUrl = url;
      break;
      // try {
      //   const response = await fetch(url, { method: "HEAD" });
      //   if (response.ok) {
      //     pdfUrl = url;
      //     break; // Exit the loop once a valid URL is found
      //   }
      // } catch (error) {
      //   console.error(`Error checking PDF at ${url}:`, error);
      // }

    }

    if (pdfUrl) {
      setSelectedPdf(pdfUrl);
      setIsLeftSidebarOpen(true);
    } else {
      console.error("PDF not found for the selected language and folder.");
    }
  };

  const getLastMessageBySender = (messages, sender) => {
  const filteredMessages = messages.filter((msg) => msg.sender === sender);
  // console.log(`Filtered Messages for sender "${sender}":`, filteredMessages); // Debugging log
  return filteredMessages.length > 0 ? filteredMessages[filteredMessages.length - 1].text : "";
};

  const downloadChatHistory = async () => {
    try {
      const interactions = getUserBotInteractions(messages, metadata); // Extract interactions
      console.log("User-Bot Interactions:", interactions); // Debugging log

      const response = await fetch("/api/download");

      if (!response.ok) {
        throw new Error("Failed to download the evaluation results.");
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);

      const link = document.createElement("a");
      link.href = url;
      link.download = "combined_eval.jsonl"; // File name for the downloaded file
      link.click();

      URL.revokeObjectURL(url); // Clean up the URL object
    } catch (error) {
      console.error("Error downloading evaluation results:", error);

      // Fallback: Save interactions as a .jsonl file
      const interactions = getUserBotInteractions(messages, metadata);
      const jsonlContent = interactions
        .map((interaction) => JSON.stringify(interaction))
        .join("\n");
      const blob = new Blob([jsonlContent], { type: "application/json" });
      const url = URL.createObjectURL(blob);

      const link = document.createElement("a");
      link.href = url;
      link.download = "chat.jsonl"; // Fallback file name
      link.click();

      URL.revokeObjectURL(url); // Clean up the URL object
    }
  };

  const getUserBotInteractions = (messages, metadata) => {
    const interactions = [];
    let currentQuery = null;

    messages.forEach((message) => {
      if (message.sender === "user") {
        // Start a new interaction with the user's query
        currentQuery = {
          query: message.text,
          response: null,
          retrieved_documents: [],
        };
      } else if (message.sender === "bot" && currentQuery) {
        // Add the bot's response to the current interaction
        currentQuery.response = message.text;

        // Add retrieved documents (fileName) from metadata
        if (Array.isArray(metadata)) {
          currentQuery.retrieved_documents = metadata.map((item) => item.fileName);
        }

        // Push the completed interaction to the list
        interactions.push(currentQuery);
        currentQuery = null; // Reset for the next interaction
      }
    });

    return interactions;
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      if (lastMessage.sender === "bot" && metadata.length > 0) {
        const allSubjects = metadata
          .map((item) => item.metadata.subjects)
          .filter((subjects) => subjects)
          .join(", ")
          .split(", ")
          .map((subject) => subject.trim());
        fetchHighlightedSubjects(lastMessage.text, allSubjects);
      }
    }
  }, [messages, metadata]);

  useEffect(() => {
    const fetchSearchDataset = async () => {
      try {
        const response = await fetch("/api/config/search_dataset");
        if (response.ok) {
          const data = await response.json();
          setSearchDataset(data.search_dataset);
        } else {
          console.error("Failed to fetch SEARCH_DATASET configuration.");
        }
      } catch (error) {
        console.error("Error fetching SEARCH_DATASET:", error);
      }
    };

    fetchSearchDataset();
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100vh",
          width: "100%",
          px: 2,
          backgroundColor: theme.palette.background.default,
        }}
      >
        {/* Left Sidebar for PDF Viewer */}
        <LeftBar
          isLeftSidebarOpen={isLeftSidebarOpen}
          toggleLeftSidebar={toggleLeftSidebar}
          selectedPdf={selectedPdf}
        />

        <Box
          sx={{
            width: "100%",
            maxWidth: "1200px",
            height: "90vh",
            display: "flex",
            flexDirection: "column",
            py: 2,
            px: 3,
            borderRadius: 2,
            boxShadow: 3,
            bgcolor: theme.palette.background.paper,
            border: mode === "dark" ? "1px solid #666" : "none",
          }}
        >
          <Box
            sx={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              borderBottom: `1px solid ${theme.palette.divider}`,
              pb: 2,
              mb: 2,
            }}
          >
            <Box display="flex" alignItems="center" gap={2}>
              <img
                src={mode === "dark" ? uzhLogoLight : uzhLogoDark}
                alt="UZH Logo"
                style={{ height: 80 }}
              />
              <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />
              <Box>
                <Typography variant="h6" fontWeight={600}>
                  Linguistic Research Infrastructure
                </Typography>
                <Typography variant="h6" fontWeight={600}>
                  URPP Digital Religion(s)
                </Typography>
                <Typography variant="h6" fontWeight={600}>
                  Institute of Education
                </Typography>
              </Box>
            </Box>
            <Box display="flex" alignItems="center" gap={1}>
              <Tooltip title="SpiritRAG" arrow>
                <img
                src={mode === "dark" ? logoDark : logoLight}
                alt="SpiritRAG Logo"
                style={{ height: 80, marginRight: "10px" }}
                onClick={() => {}}
                />  
              </Tooltip>
              
              <IconButton
                onClick={toggleColorMode}
                color="inherit"
                size="large"
              >
                {mode === "dark" ? (
                  <Tooltip title="Switch to Light Mode" arrow>
                  <Brightness7 fontSize="large" />
                  </Tooltip>
                ) : (
                  <Tooltip title="Switch to Dark Mode" arrow>
                    <Brightness4 fontSize="large" />
                  </Tooltip>
                )}
              </IconButton>
              <IconButton
                onClick={downloadChatHistory}
                color="inherit"
                size="large"
                sx={{
                  "&:hover": {
                    backgroundColor: mode === "dark" ? "#333" : "#f0f0f0", // Subtle hover effect
                  },
                }}
              >
                <Tooltip title="Download Chat Log" arrow>
                  <Download fontSize="large" sx={{ color: mode === "dark" ? "white" : "black" }} />
                </Tooltip>
              </IconButton>
            </Box>
          </Box>

          <Box
            sx={{
              flexGrow: 1,
              overflowY: "auto",
              display: "flex",
              flexDirection: "column",
            }}
          >
            {messages.map((msg, i) => (
              <Message
                key={i}
                text={msg.text}
                sender={msg.sender}
                isTyping={i === messages.length - 1 && msg.sender === "bot"}
                scrollRef={i === messages.length - 1 ? messagesEndRef : null}
              />
            ))}
            {loading && (
              <CircularProgress sx={{ alignSelf: "center", my: 2 }} />
            )}
            <div ref={messagesEndRef} />
          </Box>

          <Box sx={{ display: "flex", alignItems: "center", mt: 2 }}>
            <TextField
              fullWidth
              placeholder="Ask a question..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton onClick={() => fileInputRef.current.click()}>
                      <UploadFile />
                    </IconButton>
                    <IconButton onClick={handleSend}>
                      <Send />
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />
            <input
              type="file"
              accept="application/pdf"
              hidden
              ref={fileInputRef}
              onChange={handleFileChange}
            />
          </Box>
          {file && (
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
              Uploaded: {file.name}
            </Typography>
          )}
          <Typography
            variant="subtitle1"
            color="text.secondary"
            sx={{ mt: 2, textAlign: "center" }}
          >
            LLMs may produce incorrect, outdated, or incomplete information.  Always verify with trusted sources.          </Typography>
        </Box>
      </Box>

      {/* Floating Action Button for Sources */}
      <Box
        sx={{
          position: "fixed",
          bottom: 16,
          right: 16,
          zIndex: theme.zIndex.drawer + 1,
        }}
      >
        <Tooltip title="Source" arrow>
          <Fab
            color="primary"
            aria-label="sources"
            onClick={toggleSidebar}
            sx={{
              backgroundColor: mode === "dark" ? "#1565c0" : "#1C8D96",
              "&:hover": {
                backgroundColor: mode === "dark" ? "#0d47a1" : "#b2ebf2",
              },
            }}
          >
            <Source />
          </Fab>
        </Tooltip>
      </Box>

      {/* Floating Action Button for Left Sidebar */}
      <Box
        sx={{
          position: "fixed",
          bottom: 16,
          left: 16,
          zIndex: theme.zIndex.drawer + 1,
        }}
      >
        <Tooltip title="PDF Viewer" arrow>
          <Fab
            color="primary"
            aria-label="pdf-viewer"
            onClick={toggleLeftSidebar}
            sx={{
              backgroundColor: mode === "dark" ? "#1565c0" : "#1C8D96",
              "&:hover": {
                backgroundColor: mode === "dark" ? "#0d47a1" : "#b2ebf2",
              },
            }}
          >
            <MenuBookIcon />
          </Fab>
        </Tooltip>
      </Box>

      {/* Sidebar for displaying sources */}
      <RightBar
        isSidebarOpen={isSidebarOpen}
        toggleSidebar={toggleSidebar}
        metadata={metadata}
        showEval={showEval}
        setShowEval={setShowEval}
        renderSubjects={renderSubjects}
        renderLanguages={renderLanguages}
        toCamelCase={toCamelCase}
        userQuery={getLastMessageBySender(messages, "user")} // Extract the last user query
        generatedAnswer={getLastMessageBySender(messages, "bot")} // Extract the last bot answer
      />
    </ThemeProvider>
  );
};

export default Chat;
