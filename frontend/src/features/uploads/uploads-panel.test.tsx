import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { UploadsPanel } from "./uploads-panel";

describe("UploadsPanel", () => {
  it("renders the current upload empty state with UTF-8 Chinese copy", () => {
    render(<UploadsPanel uploadedAssets={{}} onUpload={vi.fn()} />);

    expect(screen.getByRole("heading", { name: "资料上传" })).toBeInTheDocument();
    expect(screen.getByText("已恢复资料：0")).toBeInTheDocument();
    expect(screen.getByText("当前暂无已上传资料")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "上传资料" })).toBeInTheDocument();
  });
});
