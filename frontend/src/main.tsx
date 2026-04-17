import { createRoot } from "react-dom/client";

import { AppProviders } from "./app/providers";
import { AppRouter } from "./app/router";
import "./styles/tokens.css";
import "./styles/globals.css";

const container = document.getElementById("root");

if (!container) {
  throw new Error("Root container #root was not found.");
}

createRoot(container).render(
  <AppProviders>
    <AppRouter />
  </AppProviders>,
);
