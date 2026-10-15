import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { trackOpenedProjects } from "./api";
import "./styles.css";

trackOpenedProjects();

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
