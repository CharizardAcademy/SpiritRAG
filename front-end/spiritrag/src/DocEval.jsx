import React, { useState } from "react";
import { Box, IconButton, Typography, Tooltip } from "@mui/material";
import { Star, StarBorder } from "@mui/icons-material";

const DocEval = ({ onFeedbackSubmit }) => {
  const [ratings, setRatings] = useState({
    relevance: 0,
    accuracy: 0,
    usefulness: 0,
    temporality: 0,
    actionability: 0,
  });

  const handleRating = (perspective, value) => {
    const updatedRatings = { ...ratings, [perspective]: value };
    setRatings(updatedRatings);
    if (onFeedbackSubmit) {
      onFeedbackSubmit(updatedRatings); // Pass updated ratings to the parent
    }
  };

  const perspectives = [
    { key: "relevance", label: "Relevance", description: "How relevant is the document to your question?" },
    { key: "accuracy", label: "Accuracy", description: "How trustworthy is the document for answering your question?" },
    { key: "usefulness", label: "Usefulness", description: "How useful is the document for answering your question?" },
    { key: "temporality", label: "Temporality", description: "Is the document related to the time period implied by your question?" },
    { key: "actionability", label: "Actionability", description: "How many actionable insights does the document contain?" },
  ];

  const starColor = "#1C8D96"; // Define the color for the stars

  return (
    <Box
      sx={{
        mt: 3, // Add space above the Eval component
        p: 0.3, // Padding inside the Eval component
        border: "1px solid",
        borderColor: "divider",
        borderRadius: 2,
        backgroundColor: "background.paper",
        boxShadow: 1,
        display: "flex",
        flexDirection: "column",
        alignItems: "center", // Center the content horizontally
        justifyContent: "center", // Center the content vertically
      }}
    >
      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: "repeat(2, 1fr)", // Two columns for compact layout
          gap: 0, // Space between rows and columns
          width: "100%", // Ensure the grid spans the full width
        }}
      >
        {perspectives.map(({ key, label, description }) => (
          <Box
            key={key}
            display="flex"
            alignItems="center"
            justifyContent="space-between"
            sx={{
              padding: "0.5rem 1rem", // Add padding for better spacing
            }}
          >
            <Tooltip title={description} arrow>
              <Typography
                variant="body2"
                color="text.primary"
                sx={{ fontWeight: 500, flex: 1, cursor: "help" }}
              >
                {label}:
              </Typography>
            </Tooltip>
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                gap: 0.2, // Reduce gap between stars
                ml: 0, // Increase space between text and the first star
              }}
            >
              {[1, 2, 3, 4, 5].map((value) => (
                <IconButton
                  key={value}
                  onClick={() => handleRating(key, value)}
                  sx={{ p: 0.3 }} // Reduce padding around the stars
                >
                  {value <= ratings[key] ? (
                    <Star fontSize="small" sx={{ color: starColor }} />
                  ) : (
                    <StarBorder fontSize="small" sx={{ color: starColor }} />
                  )}
                </IconButton>
              ))}
            </Box>
          </Box>
        ))}
      </Box>
    </Box>
  );
};

export default DocEval;