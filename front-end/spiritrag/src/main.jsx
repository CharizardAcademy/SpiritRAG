import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import React from 'react';
import "./index.css";
import Chat from "./Chat.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <Chat />
  </StrictMode>
);
