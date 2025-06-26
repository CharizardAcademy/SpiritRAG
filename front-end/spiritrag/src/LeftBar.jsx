import React from "react";
import { Box, Typography, Drawer } from "@mui/material";

const LeftBar = ({ isLeftSidebarOpen, toggleLeftSidebar, selectedPdf }) => {
  return (
    <Drawer
      anchor="left"
      open={isLeftSidebarOpen}
      onClose={toggleLeftSidebar}
      sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}
    >
      <Box
        sx={{
          width: 600,
          p: 2,
          display: "flex",
          flexDirection: "column",
          height: "100%",
        }}
      >
        <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>
          PDF Viewer
        </Typography>
        {selectedPdf ? (
          <iframe
            src={selectedPdf}
            // src={"http://localhost:5000/api/pdf/health/crawled_data/A-RES-76-69/A-RES-76-69-en.pdf"}
            title="PDF Viewer"
            style={{ width: "100%", height: "100%" }}
          />
        ) : (
          <Typography variant="body2" color="text.secondary">
            No PDF selected.
          </Typography>
        )}
      </Box>
    </Drawer>
  );
};

export default LeftBar;