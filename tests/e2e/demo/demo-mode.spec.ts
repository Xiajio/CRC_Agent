import { expect, test } from "@playwright/test";
import path from "node:path";

const repoRoot = path.basename(process.cwd()) === "frontend"
  ? path.resolve(process.cwd(), "..")
  : process.cwd();

test("replay demo walks patient and doctor flows", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("button", { name: "填写患者信息" }).click();
  await page.getByLabel("患者名称").fill("演示患者");
  await page.getByLabel("患者编号").fill("DEMO-001");
  await page.getByRole("button", { name: "保存" }).click();

  await page.getByTestId("conversation-input").fill("最近两个月大便带血，有时左下腹隐痛，大便变细，人也瘦了五斤。");
  await page.getByTestId("conversation-input").press("Enter");
  await expect(page.getByText("门诊分诊追问")).toBeVisible();

  await page.getByRole("button", { name: "1个月以上" }).click();
  await expect(page.getByText("建议尽快")).toBeVisible();

  await page.getByRole("button", { name: "上传资料" }).click();
  await page
    .getByTestId("upload-input")
    .setInputFiles(path.join(repoRoot, "tests", "fixtures", "demo_uploads", "demo_colonoscopy_report.pdf"));
  await expect(page.getByText("demo_colonoscopy_report.pdf")).toBeVisible();

  await page.getByRole("button", { name: "医生场景" }).click();
  await expect(page.getByTestId("doctor-scene")).toBeVisible();

  await page.getByRole("button", { name: "患者数据库" }).click();
  await page.getByRole("button", { name: "历史病例" }).click();
  await expect(page.getByTestId("database-workbench")).toBeVisible();
  await page.getByRole("button", { name: "查看 93" }).click();
  await page.getByTestId("database-case-093-bring-in").click();

  await page.getByRole("button", { name: "会诊" }).click();
  await page.getByTestId("conversation-input").fill("请基于当前患者信息生成临床评估、证据依据和治疗建议。");
  await page.getByTestId("conversation-input").press("Enter");
  await expect(page.getByText("需人工复核").first()).toBeVisible();
  await expect(page.getByText("执行计划")).toBeVisible();
  await expect(page.getByText("工作流路线图")).toBeVisible();

  await page.getByRole("button", { name: "多模态" }).click();
  await expect(page.getByTestId("doctor-multimodal-view")).toBeVisible();
});
