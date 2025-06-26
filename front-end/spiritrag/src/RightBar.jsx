import React, { useState } from "react";
import {
  Box,
  Typography,
  Drawer,
  Fab,
  Tooltip,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Snackbar,
  Alert,
} from "@mui/material";
import DocEval from "./DocEval";
import AnsEval from "./AnsEval"; // Import the AnsEval component

const RightBar = ({
  isSidebarOpen,
  toggleSidebar,
  metadata,
  showEval,
  setShowEval,
  renderSubjects,
  renderLanguages,
  toCamelCase,
  userQuery, // Receive userQuery as a prop
  generatedAnswer, // Receive generatedAnswer as a prop
}) => {
  const [docFeedback, setDocFeedback] = useState({});
  const [ansFeedback, setAnsFeedback] = useState({});
  const [snackbarOpen, setSnackbarOpen] = useState(false); // State for Snackbar

  const handleDocFeedbackSubmit = (docId, ratings) => {
    setDocFeedback((prev) => ({ ...prev, [docId]: ratings }));
  };

  const handleAnsFeedbackSubmit = (ratings) => {
    setAnsFeedback(ratings);
  };

  const handleSendEvaluation = () => {
    // Map metadata to retrievedDocuments if retrievedDocuments is not already defined
    const retrievedDocuments = Array.isArray(metadata)
      ? metadata.map((item) => ({
          fileName: item.fileName || "", // Ensure fileName is correctly accessed
          title: item.metadata.title || "Unknown Title", // Add title from metadata
          publication_date: item.metadata.publication_date || "Unknown Date", // Add publication_date from metadata
          subjects: item.metadata.subjects || "Unknown Subjects", // Add subjects from metadata
          eval: docFeedback[item.fileName] || {}, // Use feedback if available, fallback to an empty object
        }))
      : [];

    // Get the current time in Zurich time
    const zurichTime = new Intl.DateTimeFormat("en-CH", {
      timeZone: "Europe/Zurich",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    })
      .format(new Date())
      .replace(",", "") // Remove the comma between date and time
      .replace(/(\d{2})\.(\d{2})\.(\d{4})/, "$3-$2-$1"); // Convert DD.MM.YYYY to YYYY-MM-DD

    // Prepare the evaluation data
    const evaluationData = {
      query: userQuery || "", // Use the userQuery prop
      answer: {
        text: generatedAnswer || "", // Use the generatedAnswer prop
        eval: ansFeedback || {}, // Fallback to an empty object if ansFeedback is undefined
      },
      docs: retrievedDocuments, // Use the mapped retrievedDocuments
      time: zurichTime, // Current timestamp in Zurich time
    };

    console.log("Sending evaluation data:", evaluationData); // Log the payload for debugging

    // Send the evaluation data to the backend
    fetch("/api/save-eval", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(evaluationData),
    })
      .then((response) => {
        if (response.ok) {
          console.log("Evaluation results sent successfully!");
          setSnackbarOpen(true); // Show Snackbar on success
        } else {
          console.error("Failed to send evaluation results:", response.statusText);
        }
      })
      .catch((error) => {
        console.error("Error sending evaluation results:", error);
      });

    console.log(metadata);
  };

  const handleSnackbarClose = () => {
    setSnackbarOpen(false); // Close the Snackbar
  };

  return (
    <Drawer
      anchor="right"
      open={isSidebarOpen}
      onClose={toggleSidebar}
      sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}
    >
      <Box
        sx={{
          width: 600,
          p: 2,
          display: "flex",
          flexDirection: "column",
          height: "100%",
          pb: showEval ? 8 : 4,
        }}
      >
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            mb: 2,
          }}
        >
          <Typography variant="h6" fontWeight={600}>
            Sources
          </Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            {showEval && (
              <Tooltip title="Submit Feedback" arrow>
                <Fab
                  size="small"
                  color="primary"
                  onClick={handleSendEvaluation} // Bind the function here
                  sx={{
                    backgroundColor: "#ffb74d",
                    "&:hover": {
                      backgroundColor: "#ff9800",
                    },
                  }}
                >
                  <Typography
                    variant="button"
                    sx={{ fontSize: "0.8rem", color: "white" }}
                  >
                    Send
                  </Typography>
                </Fab>
              </Tooltip>
            )}
            <Tooltip title="Start Evaluation" arrow>
              <Fab
                size="small"
                onClick={() => setShowEval((prev) => !prev)} // Toggle Eval visibility
                sx={{
                  backgroundColor: showEval ? "#1C8D96" : "grey", // Use #1C8D96 for ON and grey for OFF
                  "&:hover": {
                    backgroundColor: showEval ? "#0e6b6f" : "#a9a9a9", // Darker shades for hover
                  },
                }}
              >
                <Typography
                  variant="button"
                  sx={{ fontSize: "0.8rem", color: "white" }}
                >
                  Eval
                </Typography>
              </Fab>
            </Tooltip>
          </Box>
        </Box>
        <Grid container spacing={2}>
          {metadata.length > 0 ? (
            metadata.map((item, index) => (
              <Grid
                key={index}
                sx={{
                  mb: index === metadata.length - 1 && !showEval ? "20px" : 0, // Adjust margin-bottom dynamically
                }}
              >
                <Card
                  elevation={3}
                  sx={{
                    width: "100%",
                    maxWidth: "600px",
                    margin: "0 auto",
                    display: "flex",
                    flexDirection: "column",
                  }}
                >
                  <CardHeader
                    disableTypography
                    title={
                      <span
                        style={{
                          fontWeight: "bold",
                          color: "#1C8D96",
                          whiteSpace: "normal",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          display: "block",
                        }}
                      >
                        {item.metadata.title === item.metadata.symbol
                          ? item.metadata.title
                          : toCamelCase(item.metadata.title)}
                      </span>
                    }
                    sx={{
                      width: "100%",
                      boxSizing: "border-box",
                    }}
                  />
                  <CardContent sx={{ pt: 0 }}>
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{ mb: 0.5 }}
                    >
                      <strong>Subjects:</strong>{" "}
                      {item.metadata.subjects
                        ? renderSubjects(item.metadata.subjects)
                        : "Missing"}
                    </Typography>
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{ mb: 0.5 }}
                    >
                      <strong>Publication Date:</strong>{" "}
                      {item.metadata.publication_date || "Missing"}
                    </Typography>
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{
                        display: "flex",
                        flexWrap: "wrap",
                        alignItems: "center",
                        gap: "8px",
                      }}
                    >
                      <strong>Languages:</strong>{" "}
                      {renderLanguages(
                        item.metadata.languages,
                        item.fileName.split("-parsed.json")[0]
                      )}
                    </Typography>
                    {showEval && (
                      <DocEval
                        onFeedbackSubmit={(ratings) =>
                          handleDocFeedbackSubmit(item.fileName, ratings)
                        }
                      />
                    )}
                  </CardContent>
                </Card>
                
              </Grid> 
            ))
          ) : (
            <Typography variant="body2" color="text.secondary">
              No metadata available.
            </Typography>
          )}
        </Grid> 
        {metadata.length > 0 && showEval && (
          <AnsEval
            sx={{ mb: 8, mt: 8 }}
            onFeedbackSubmit={handleAnsFeedbackSubmit}
          />
        )}
        {metadata.length > 0 && showEval && (
            <Typography style={{ color: 'transparent' }}>
                This text is invisible
            </Typography>
        )}
        <Snackbar
          open={snackbarOpen}
          autoHideDuration={2000} // 2 seconds
          onClose={handleSnackbarClose}
        >
          <Alert onClose={handleSnackbarClose} severity="success" sx={{ width: "100%" }}>
            Evaluation sent successfully!
          </Alert>
        </Snackbar>
      </Box>
    </Drawer>
  );
};

export default RightBar;