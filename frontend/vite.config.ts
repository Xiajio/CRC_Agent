import path from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const workspaceRoot = path.resolve(__dirname, "..");
const testsRoot = path.resolve(workspaceRoot, "tests", "frontend");
const nodeModulesRoot = path.resolve(__dirname, "node_modules");

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      react: path.resolve(nodeModulesRoot, "react"),
      "react/jsx-runtime": path.resolve(nodeModulesRoot, "react/jsx-runtime.js"),
      "react/jsx-dev-runtime": path.resolve(nodeModulesRoot, "react/jsx-dev-runtime.js"),
      "react-dom": path.resolve(nodeModulesRoot, "react-dom"),
      vitest: path.resolve(nodeModulesRoot, "vitest"),
      "@testing-library/react": path.resolve(nodeModulesRoot, "@testing-library/react"),
      "@testing-library/jest-dom/vitest": path.resolve(
        nodeModulesRoot,
        "@testing-library/jest-dom/vitest.js",
      ),
    },
  },
  server: {
    fs: {
      allow: [workspaceRoot],
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: [path.resolve(testsRoot, "setup.ts")],
    include: ["../tests/frontend/**/*.test.ts", "../tests/frontend/**/*.test.tsx"],
  },
});
