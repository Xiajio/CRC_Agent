import { BrowserRouter, Route, Routes } from "react-router-dom";

import { DatabasePage } from "../pages/database-page";
import { WorkspacePage } from "../pages/workspace-page";

export function AppRouter() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<WorkspacePage />} />
        <Route path="/database" element={<DatabasePage />} />
      </Routes>
    </BrowserRouter>
  );
}
