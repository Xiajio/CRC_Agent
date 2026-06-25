from __future__ import annotations

import math
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "medical-research-assistant-integration-report-2026-06-24.pdf"
SCREENSHOT = Path("C:/Users/msi/AppData/Local/Temp/codex-clipboard-7c5a9d09-a443-4777-94fe-853a0dc71912.png")

PAGE_W, PAGE_H = A4
MARGIN = 42
CONTENT_W = PAGE_W - MARGIN * 2

RED = colors.HexColor("#c9142f")
DARK_RED = colors.HexColor("#8d1021")
PINK = colors.HexColor("#fff2f3")
PINK_2 = colors.HexColor("#fae2e5")
INK = colors.HexColor("#1f2328")
MUTED = colors.HexColor("#5d6673")
LIGHT = colors.HexColor("#f7f8fa")
BORDER = colors.HexColor("#e3e7ee")
BLUE = colors.HexColor("#2f6fbd")
GREEN = colors.HexColor("#2f9e68")
ORANGE = colors.HexColor("#e7902e")
PURPLE = colors.HexColor("#8250df")
TEAL = colors.HexColor("#1f8a8a")


def register_fonts() -> tuple[str, str]:
    regular_candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    bold_candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
    ]

    regular = next((p for p in regular_candidates if p.exists()), None)
    bold = next((p for p in bold_candidates if p.exists()), None)
    if regular is None or bold is None:
        raise RuntimeError("No usable Chinese font found under C:/Windows/Fonts")

    pdfmetrics.registerFont(TTFont("BodyCN", str(regular)))
    pdfmetrics.registerFont(TTFont("BoldCN", str(bold)))
    return "BodyCN", "BoldCN"


BODY, BOLD = register_fonts()


class Report:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.c = canvas.Canvas(str(path), pagesize=A4)
        self.page_num = 0
        self.section = ""

    def page(self, section: str):
        if self.page_num > 0:
            self.footer()
            self.c.showPage()
        self.page_num += 1
        self.section = section
        self.header()

    def save(self):
        self.footer()
        self.c.save()

    def header(self):
        if self.page_num == 1:
            return
        self.c.setFillColor(RED)
        self.c.roundRect(MARGIN, PAGE_H - 34, 4, 14, 2, fill=1, stroke=0)
        self.c.setFillColor(INK)
        self.c.setFont(BOLD, 9)
        self.c.drawString(MARGIN + 12, PAGE_H - 28, "小亿灵析科研分析助手 - 医疗结合分析")
        self.c.setFillColor(MUTED)
        self.c.setFont(BODY, 8)
        self.c.drawRightString(PAGE_W - MARGIN, PAGE_H - 28, self.section)
        self.c.setStrokeColor(BORDER)
        self.c.line(MARGIN, PAGE_H - 40, PAGE_W - MARGIN, PAGE_H - 40)

    def footer(self):
        if self.page_num == 1:
            return
        self.c.setStrokeColor(BORDER)
        self.c.line(MARGIN, 30, PAGE_W - MARGIN, 30)
        self.c.setFillColor(MUTED)
        self.c.setFont(BODY, 8)
        self.c.drawString(MARGIN, 18, "基于当前 LangG 工作区代码、架构文档与科研助手草案截图生成")
        self.c.drawRightString(PAGE_W - MARGIN, 18, f"{self.page_num}")


def text_width(text: str, font: str, size: float) -> float:
    return pdfmetrics.stringWidth(text, font, size)


def wrap_text(text: str, font: str, size: float, width: float) -> list[str]:
    lines: list[str] = []
    for raw in str(text).split("\n"):
        if not raw:
            lines.append("")
            continue
        current = ""
        for ch in raw:
            candidate = current + ch
            if text_width(candidate, font, size) <= width or not current:
                current = candidate
            else:
                lines.append(current)
                current = ch
        if current:
            lines.append(current)
    return lines


def draw_text(c: canvas.Canvas, x: float, y: float, text: str, width: float, *,
              font: str = BODY, size: float = 10, leading: float = 14,
              color=INK, max_lines: int | None = None) -> float:
    c.setFillColor(color)
    c.setFont(font, size)
    lines = wrap_text(text, font, size, width)
    if max_lines is not None:
        lines = lines[:max_lines]
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def draw_title(c: canvas.Canvas, y: float, title: str, subtitle: str | None = None) -> float:
    c.setFillColor(INK)
    c.setFont(BOLD, 22)
    c.drawString(MARGIN, y, title)
    y -= 16
    c.setStrokeColor(RED)
    c.setLineWidth(3)
    c.line(MARGIN, y, MARGIN + 80, y)
    y -= 18
    if subtitle:
        y = draw_text(c, MARGIN, y, subtitle, CONTENT_W, size=10.5, leading=15, color=MUTED)
        y -= 8
    return y


def rounded_card(c: canvas.Canvas, x: float, y: float, w: float, h: float,
                 fill=colors.white, stroke=BORDER, radius: float = 8):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(0.8)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1)


def badge(c: canvas.Canvas, x: float, y: float, text: str, fill=PINK, color=RED) -> float:
    pad_x = 8
    w = text_width(text, BOLD, 8.5) + pad_x * 2
    c.setFillColor(fill)
    c.setStrokeColor(fill)
    c.roundRect(x, y - 3, w, 16, 8, fill=1, stroke=0)
    c.setFillColor(color)
    c.setFont(BOLD, 8.5)
    c.drawString(x + pad_x, y, text)
    return x + w + 6


def metric_card(c: canvas.Canvas, x: float, y: float, w: float, h: float,
                value: str, label: str, detail: str, color=RED):
    rounded_card(c, x, y, w, h, colors.white, BORDER, 8)
    c.setFillColor(color)
    c.setFont(BOLD, 18)
    c.drawString(x + 12, y + h - 28, value)
    c.setFillColor(INK)
    c.setFont(BOLD, 9)
    c.drawString(x + 12, y + h - 45, label)
    draw_text(c, x + 12, y + h - 61, detail, w - 24, size=7.5, leading=10, color=MUTED, max_lines=2)


def draw_bullets(c: canvas.Canvas, x: float, y: float, width: float,
                 items: list[str], *, size: float = 9.2, leading: float = 13,
                 bullet_color=RED) -> float:
    for item in items:
        c.setFillColor(bullet_color)
        c.circle(x + 3, y + 3.5, 2.2, fill=1, stroke=0)
        y = draw_text(c, x + 12, y, item, width - 12, size=size, leading=leading, color=INK)
        y -= 3
    return y


def draw_table(c: canvas.Canvas, x: float, y: float, col_widths: list[float],
               headers: list[str], rows: list[list[str]], *,
               row_h: float = 24, header_h: float = 25, font_size: float = 7.6,
               fills: list | None = None) -> float:
    total_w = sum(col_widths)
    c.setFillColor(DARK_RED)
    c.setStrokeColor(DARK_RED)
    c.roundRect(x, y - header_h, total_w, header_h, 6, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont(BOLD, font_size)
    cx = x
    for i, h in enumerate(headers):
        c.drawString(cx + 6, y - 16, h)
        cx += col_widths[i]
    y -= header_h
    for idx, row in enumerate(rows):
        fill = fills[idx] if fills and idx < len(fills) else (colors.white if idx % 2 == 0 else LIGHT)
        c.setFillColor(fill)
        c.setStrokeColor(BORDER)
        c.rect(x, y - row_h, total_w, row_h, fill=1, stroke=1)
        cx = x
        for i, cell in enumerate(row):
            draw_text(c, cx + 6, y - 14, cell, col_widths[i] - 12, size=font_size, leading=9, color=INK, max_lines=2)
            cx += col_widths[i]
        y -= row_h
    return y


def draw_arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float, color=RED):
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(1.5)
    c.line(x1, y1, x2, y2)
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 6
    pts = [
        (x2, y2),
        (x2 - size * math.cos(angle - math.pi / 6), y2 - size * math.sin(angle - math.pi / 6)),
        (x2 - size * math.cos(angle + math.pi / 6), y2 - size * math.sin(angle + math.pi / 6)),
    ]
    c.line(pts[0][0], pts[0][1], pts[1][0], pts[1][1])
    c.line(pts[0][0], pts[0][1], pts[2][0], pts[2][1])


def draw_screenshot(c: canvas.Canvas, x: float, y: float, w: float, h: float):
    if not SCREENSHOT.exists():
        rounded_card(c, x, y, w, h, LIGHT, BORDER, 8)
        draw_text(c, x + 12, y + h - 24, "草案截图素材未找到", w - 24, size=9, color=MUTED)
        return
    img = Image.open(SCREENSHOT)
    img.thumbnail((int(w * 2), int(h * 2)))
    reader = ImageReader(img)
    iw, ih = img.size
    scale = min(w / iw, h / ih)
    draw_w = iw * scale
    draw_h = ih * scale
    rounded_card(c, x, y, w, h, colors.white, BORDER, 8)
    c.drawImage(reader, x + (w - draw_w) / 2, y + (h - draw_h) / 2, draw_w, draw_h, preserveAspectRatio=True, mask="auto")


def page_cover(r: Report):
    c = r.c
    r.page("封面")
    c.setFillColor(PINK)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(RED)
    c.circle(PAGE_W - 80, PAGE_H - 88, 92, fill=1, stroke=0)
    c.setFillColor(DARK_RED)
    c.circle(PAGE_W - 45, PAGE_H - 48, 45, fill=1, stroke=0)
    c.setFillColor(INK)
    c.setFont(BOLD, 31)
    c.drawString(MARGIN, PAGE_H - 118, "科研分析助手")
    c.drawString(MARGIN, PAGE_H - 156, "与医疗结合分析报告")
    c.setFont(BODY, 13)
    c.setFillColor(MUTED)
    c.drawString(MARGIN, PAGE_H - 184, "面向 CRC 临床科研的能力盘点、机会判断与 MVP 路线图")
    x = MARGIN
    for text in ["医疗科研雷达", "真实世界队列", "多模态肿瘤研究", "证据链护栏"]:
        x = badge(c, x, PAGE_H - 218, text)
    rounded_card(c, MARGIN, 86, 230, 210, colors.white, BORDER, 10)
    c.setFillColor(RED)
    c.setFont(BOLD, 18)
    c.drawString(MARGIN + 18, 264, "核心判断")
    bullets = [
        "最优切入点是医学科研雷达，而不是大而全平台。",
        "现有 LangG 已具备医疗垂直多智能体底座。",
        "先做可审核证据发现，再做自动调度和写入知识库。",
        "队列分析与多模态样本集是第二增长点。",
    ]
    draw_bullets(c, MARGIN + 18, 238, 196, bullets, size=8.7, leading=12)
    draw_screenshot(c, MARGIN + 255, 76, 275, 410)
    c.setFillColor(MUTED)
    c.setFont(BODY, 9)
    c.drawString(MARGIN, 52, "生成日期：2026-06-24")
    c.drawRightString(PAGE_W - MARGIN, 52, "输出位置：output/pdf")


def page_summary(r: Report):
    c = r.c
    r.page("结论摘要")
    y = draw_title(c, PAGE_H - 72, "1. 结论摘要", "当前系统最适合演进为医疗科研证据智能平台，先以低风险、可审核的科研雷达作为第一阶段。")
    cards = [
        ("高复用", "RAG、Web Search、病例库、多模态工具、证据链均可直接复用。", RED),
        ("低风险", "MVP 定位为科研信息发现，不直接生成患者诊疗结论。", GREEN),
        ("强差异", "把最新证据、本地队列和多模态 AI 放在同一个研究工作台。", BLUE),
    ]
    x = MARGIN
    for value, label, color in cards:
        metric_card(c, x, y - 78, 160, 68, value, "战略判断", label, color)
        x += 176
    y -= 104
    rounded_card(c, MARGIN, y - 150, CONTENT_W, 138, colors.white, BORDER, 10)
    c.setFillColor(RED)
    c.setFont(BOLD, 14)
    c.drawString(MARGIN + 16, y - 36, "五条关键结论")
    y2 = y - 58
    draw_bullets(c, MARGIN + 18, y2, CONTENT_W - 36, [
        "不要把当前系统包装成已经完整落地的通用科研助手。它的真实优势是 CRC 医疗垂直多智能体底座。",
        "草案中的论文深读、证据追踪、知识资产和 Agent Chat 最适合第一批医疗科研功能。",
        "论文写作、专利/IP、趋势预测和实验沙盒要等项目实体、审核流和证据库成熟后再做。",
        "建议新增 Research Workspace，与 Patient、Doctor、Agent Admin 分层。",
        "第一阶段交付应是手动触发、人工审核、可追溯来源的医学科研雷达。",
    ], size=9.2, leading=13)
    y -= 182
    rounded_card(c, MARGIN, y - 170, CONTENT_W, 155, PINK, PINK_2, 10)
    c.setFillColor(DARK_RED)
    c.setFont(BOLD, 15)
    c.drawString(MARGIN + 16, y - 40, "推荐 MVP：小亿灵析医疗科研雷达")
    draw_text(c, MARGIN + 16, y - 64,
              "医生或 PI 创建研究主题，系统手动触发最新论文和指南更新搜索，生成论文/指南卡片、结构化摘要、证据强度和局限性说明。用户审核后加入项目证据池，并结合本地病例库形成初步可行性提示。",
              CONTENT_W - 32, size=10, leading=15, color=INK)
    x = MARGIN + 16
    for step in ["主题", "搜索", "去重", "摘要", "审核", "证据池"]:
        rounded_card(c, x, y - 137, 70, 30, colors.white, PINK_2, 8)
        c.setFillColor(RED)
        c.setFont(BOLD, 9)
        c.drawCentredString(x + 35, y - 124, step)
        x += 84


def page_current_capabilities(r: Report):
    c = r.c
    r.page("现有能力")
    y = draw_title(c, PAGE_H - 72, "2. 当前 LangG 医疗底座", "现有代码已经实现患者端、医生端、后台观测、RAG、联网搜索、病例库、文档解析、影像/病理 AI 和证据链。")
    layer_y = y - 60
    layers = [
        ("Frontend", "Patient / Doctor / Agent Admin 工作台\nChat、卡片、多模态视图、数据库工作台、证据池", RED),
        ("Backend BFF", "sessions、chat stream、database、uploads、assets、patient-registry、admin tools", BLUE),
        ("Agent Core", "Intent -> Planner -> Knowledge/Case/Rad/Path -> Assessment -> Decision -> Critic -> Citation -> Evaluator", PURPLE),
        ("RAG + Tools", "Chroma + BM25 + rerank\n临床工具、Web Search、YOLO、U-Net、PyRadiomics、CLAM", GREEN),
    ]
    x = MARGIN
    for idx, (title, body, color) in enumerate(layers):
        rounded_card(c, x, layer_y - 95, CONTENT_W, 78, colors.white, BORDER, 10)
        c.setFillColor(color)
        c.roundRect(x + 12, layer_y - 45, 72, 26, 6, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont(BOLD, 10)
        c.drawCentredString(x + 48, layer_y - 34, title)
        draw_text(c, x + 100, layer_y - 34, body, CONTENT_W - 120, size=8.9, leading=12, color=INK)
        if idx < len(layers) - 1:
            draw_arrow(c, PAGE_W / 2, layer_y - 105, PAGE_W / 2, layer_y - 123, RED)
        layer_y -= 112
    y2 = 146
    headers = ["能力", "当前状态", "科研价值"]
    rows = [
        ["RAG 检索", "指南目录、章节、治疗/分期/药物专项检索", "扩展到论文、共识、SOP、方案文档"],
        ["病例库", "统计、自然语言筛选、详情、写回", "真实世界队列和课题可行性"],
        ["多模态 AI", "YOLO、U-Net、PyRadiomics、CLAM", "影像组学/病理组学研究样本集"],
        ["证据链", "Claim -> Evidence -> Reference", "科研输出可信度护栏"],
    ]
    draw_table(c, MARGIN, y2, [92, 205, 205], headers, rows, row_h=29, font_size=7.4)


def page_alignment_heatmap(r: Report):
    c = r.c
    r.page("草案能力对齐")
    y = draw_title(c, PAGE_H - 72, "3. 草案 10 类能力与医疗结合", "对齐结果显示：科研核心引擎、Agent Chat、推荐追踪和知识资产最适合优先医疗化。")
    headers = ["草案能力", "匹配度", "医疗结合点", "优先级"]
    data = [
        ("自适应用户画像", "中", "医生/PI/课题组画像、研究偏好、队列画像", "P1"),
        ("科研核心智能引擎", "高", "论文深读、指南差异、试验方案解析、假设生成", "P0"),
        ("科研产出辅助", "中", "论文段落、研究方案、SOP、专利交底草稿", "P2"),
        ("推荐与追踪", "中高", "指南更新、临床证据、试验情报、竞品政策", "P0/P1"),
        ("多维数据可视化", "中", "队列地图、证据地图、技术成熟度地图", "P1/P2"),
        ("实验沙盒与仿真", "中", "队列可行性、算法验证、研究路径推演", "P1"),
        ("全域 Agent Chat", "高", "研究项目 Chat、虚拟专家、数据联动", "P0/P1"),
        ("知识资产统计", "中高", "项目知识库、决策日志、专家纪要", "P1"),
        ("趋势预测与专家共识", "低中", "成熟度曲线、专家倾向、临床采纳预测", "P2"),
        ("多维交叉分析矩阵", "中", "成熟度、证据等级、临床价值、本地可行性", "P1/P2"),
    ]
    fills = []
    for _, match, _, _ in data:
        if match == "高":
            fills.append(colors.HexColor("#edf8f1"))
        elif match == "中高":
            fills.append(colors.HexColor("#f1f7ff"))
        elif match == "中":
            fills.append(colors.HexColor("#fff8e8"))
        else:
            fills.append(colors.HexColor("#f8f0ff"))
    draw_table(c, MARGIN, y - 5, [108, 55, 252, 78], headers, [list(row) for row in data], row_h=33, font_size=7.4, fills=fills)
    rounded_card(c, MARGIN, 70, CONTENT_W, 54, PINK, PINK_2, 8)
    c.setFillColor(DARK_RED)
    c.setFont(BOLD, 10)
    c.drawString(MARGIN + 14, 104, "解读")
    draw_text(c, MARGIN + 14, 89,
              "`search_latest_research` 当前是候选/执行器层能力，不应描述为已上线的自动学习闭环。第一阶段应做手动运行和人工审核。",
              CONTENT_W - 28, size=8.2, leading=11, color=INK)


def page_priority_chart(r: Report):
    c = r.c
    r.page("机会优先级")
    y = draw_title(c, PAGE_H - 72, "4. 最值得优先做的医疗结合方向", "优先选择复用度高、临床风险可控、能快速形成差异化的功能。")
    chart_x, chart_y, chart_w, chart_h = MARGIN + 18, y - 300, 350, 245
    c.setStrokeColor(BORDER)
    c.setFillColor(colors.white)
    c.roundRect(chart_x - 12, chart_y - 28, chart_w + 42, chart_h + 64, 10, fill=1, stroke=1)
    c.setStrokeColor(MUTED)
    c.line(chart_x, chart_y, chart_x + chart_w, chart_y)
    c.line(chart_x, chart_y, chart_x, chart_y + chart_h)
    c.setFillColor(MUTED)
    c.setFont(BODY, 8)
    c.drawString(chart_x + chart_w - 88, chart_y - 18, "实施难度 ->")
    c.saveState()
    c.translate(chart_x - 28, chart_y + chart_h - 88)
    c.rotate(90)
    c.drawString(0, 0, "业务价值 ->")
    c.restoreState()
    opportunities = [
        ("医学科研雷达", 2.0, 4.6, RED),
        ("队列可行性", 2.4, 4.3, GREEN),
        ("指南变更监控", 2.2, 4.0, BLUE),
        ("多模态样本集", 3.6, 4.2, PURPLE),
        ("知识资产管理", 3.0, 3.5, TEAL),
        ("论文产出辅助", 3.4, 3.2, ORANGE),
        ("专利/IP 辅助", 4.3, 2.8, colors.HexColor("#8a63d2")),
        ("趋势/共识预测", 4.5, 3.1, colors.HexColor("#6f7680")),
    ]
    for label, difficulty, value, color in opportunities:
        px = chart_x + (difficulty - 1) / 4 * chart_w
        py = chart_y + (value - 1) / 4 * chart_h
        c.setFillColor(color)
        c.circle(px, py, 6.5, fill=1, stroke=0)
        draw_text(c, px + 8, py + 3, label, 90, size=7.2, leading=8, color=INK, max_lines=2)
    rounded_card(c, MARGIN + 405, chart_y - 28, 132, chart_h + 64, PINK, PINK_2, 10)
    c.setFillColor(DARK_RED)
    c.setFont(BOLD, 12)
    c.drawString(MARGIN + 420, chart_y + chart_h + 18, "优先顺序")
    draw_bullets(c, MARGIN + 420, chart_y + chart_h - 8, 103, [
        "P0 医学科研雷达",
        "P0/P1 队列分析",
        "P0/P1 指南监控",
        "P1 多模态样本集",
        "P1 知识资产",
        "P2 论文/专利/趋势",
    ], size=8, leading=11)
    y2 = chart_y - 60
    rounded_card(c, MARGIN, y2 - 110, CONTENT_W, 92, colors.white, BORDER, 8)
    c.setFillColor(INK)
    c.setFont(BOLD, 12)
    c.drawString(MARGIN + 14, y2 - 38, "为什么科研雷达排第一")
    draw_text(c, MARGIN + 14, y2 - 58,
              "它复用现有联网搜索、RAG、证据池和学习准备页面；输出定位为科研信息发现，不直接改变临床路径；并且能快速连接指南、论文、本地病例和专家审核。",
              CONTENT_W - 28, size=9.2, leading=13, color=INK)


def page_mvp_flow(r: Report):
    c = r.c
    r.page("MVP 流程")
    y = draw_title(c, PAGE_H - 72, "5. 医学科研雷达 MVP", "MVP 以手动触发、人工审核、可追溯证据为边界，避免直接自动入库或生成诊疗结论。")
    steps = [
        ("研究主题", "疾病、标志物、治疗线、关键词"),
        ("最新搜索", "PubMed / Cochrane / 指南更新"),
        ("去重打分", "来源、年份、研究类型、可信度"),
        ("结构化摘要", "PICO、样本、终点、结论、局限"),
        ("人工审核", "医生/PI 确认后进入项目证据池"),
        ("证据池", "支持后续队列分析和草稿输出"),
    ]
    box_w = 128
    box_h = 78
    gap_x = (CONTENT_W - box_w * 3) / 2
    row_gap = 30
    base_y = y - 118
    positions = []
    for i in range(len(steps)):
        row = i // 3
        col = i % 3
        positions.append((MARGIN + col * (box_w + gap_x), base_y - row * (box_h + row_gap)))

    for i, (title, body) in enumerate(steps):
        x, box_y = positions[i]
        rounded_card(c, x, box_y, box_w, box_h, colors.white, BORDER, 9)
        c.setFillColor(RED if i == 0 else BLUE if i < 4 else GREEN)
        c.circle(x + 16, box_y + box_h - 18, 10, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont(BOLD, 8)
        c.drawCentredString(x + 16, box_y + box_h - 21, str(i + 1))
        c.setFillColor(INK)
        c.setFont(BOLD, 9.2)
        c.drawString(x + 10, box_y + box_h - 42, title)
        draw_text(c, x + 10, box_y + box_h - 58, body, box_w - 20, size=7.2, leading=9, color=MUTED, max_lines=3)

    for i in range(len(steps) - 1):
        x, box_y = positions[i]
        next_x, next_y = positions[i + 1]
        if i in (0, 1, 3, 4):
            draw_arrow(c, x + box_w + 8, box_y + box_h / 2, next_x - 10, next_y + box_h / 2, RED)
        else:
            draw_arrow(c, x + box_w / 2, box_y - 6, next_x + box_w / 2, next_y + box_h + 6, RED)

    y2 = positions[-1][1] - 42
    headers = ["MVP 做什么", "明确不做什么"]
    rows = [
        ["研究主题队列、手动运行、论文/指南卡片、证据强度、人工审核", "不自动定时运行、不直接写入临床知识库"],
        ["结构化深读：PICO、样本量、终点、结果、局限性", "不自动生成完整论文、不给患者诊疗建议"],
        ["项目证据池和可追溯来源", "不默认导出患者级明细"],
    ]
    draw_table(c, MARGIN, y2, [248, 248], headers, rows, row_h=43, font_size=8.1)
    rounded_card(c, MARGIN, 80, CONTENT_W, 90, PINK, PINK_2, 10)
    c.setFillColor(DARK_RED)
    c.setFont(BOLD, 12)
    c.drawString(MARGIN + 14, 140, "对应代码落点")
    draw_text(c, MARGIN + 14, 120,
              "src/tools/web_search_tools.py 的 LatestResearchSearchTool、src/services/web_search_service.py 的 DeepResearchService、src/rag/evidence.py 的证据规范化、frontend/src/features/agent-admin 的学习准备和证据池页面。",
              CONTENT_W - 28, size=8.3, leading=12, color=INK)


def page_cohort(r: Report):
    c = r.c
    r.page("队列分析")
    y = draw_title(c, PAGE_H - 72, "6. 真实世界队列与假设生成", "本地病例库和患者登记处可以回答：这个课题有没有数据、有多少病例、字段缺不缺。")
    rounded_card(c, MARGIN, y - 250, 255, 225, colors.white, BORDER, 10)
    c.setFillColor(INK)
    c.setFont(BOLD, 13)
    c.drawString(MARGIN + 14, y - 44, "示例：队列可行性卡")
    metric_card(c, MARGIN + 14, y - 116, 68, 54, "126", "候选病例", "符合初筛", RED)
    metric_card(c, MARGIN + 92, y - 116, 68, 54, "72%", "字段完整", "关键字段", GREEN)
    metric_card(c, MARGIN + 170, y - 116, 68, 54, "48", "多模态", "影像+病理", BLUE)
    draw_bullets(c, MARGIN + 16, y - 145, 220, [
        "自然语言筛选研究队列",
        "输出样本量、缺失率、模态可用性",
        "识别偏倚和不可用字段",
        "生成可检验假设和方法学建议",
    ], size=8.4, leading=11)
    chart_x, chart_y = MARGIN + 304, y - 235
    rounded_card(c, chart_x - 14, chart_y - 16, 220, 210, colors.white, BORDER, 10)
    c.setFillColor(INK)
    c.setFont(BOLD, 12)
    c.drawString(chart_x, chart_y + 170, "字段完整度示意")
    bars = [("年龄/性别", 0.96, GREEN), ("分期", 0.88, GREEN), ("MMR/MSI", 0.74, BLUE), ("影像", 0.55, ORANGE), ("病理切片", 0.38, RED)]
    yy = chart_y + 138
    for label, ratio, color in bars:
        c.setFillColor(MUTED)
        c.setFont(BODY, 8)
        c.drawString(chart_x, yy + 3, label)
        c.setFillColor(LIGHT)
        c.rect(chart_x + 62, yy, 110, 9, fill=1, stroke=0)
        c.setFillColor(color)
        c.rect(chart_x + 62, yy, 110 * ratio, 9, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont(BOLD, 7.5)
        c.drawRightString(chart_x + 196, yy + 1, f"{int(ratio * 100)}%")
        yy -= 26
    y2 = 135
    rounded_card(c, MARGIN, y2 - 68, CONTENT_W, 52, PINK, PINK_2, 8)
    draw_text(c, MARGIN + 14, y2 - 38,
              "现有 `/api/database/stats`、`/api/database/cases/search`、`/api/database/query-intent` 可作为起点。需要新增研究字段解析、缺失率统计、脱敏报告和研究假设对象。",
              CONTENT_W - 28, size=8.5, leading=12, color=INK)


def page_multimodal(r: Report):
    c = r.c
    r.page("多模态研究")
    y = draw_title(c, PAGE_H - 72, "7. 多模态肿瘤研究工作台", "将影像、病理、影像组学和临床字段组织成可复核的研究样本视图。")
    lanes = [
        ("病例样本集", "筛选结果 -> 样本集", RED),
        ("影像链路", "YOLO 检测 -> U-Net 分割", BLUE),
        ("放射组学", "PyRadiomics -> LASSO 特征", GREEN),
        ("病理链路", "WSI -> CLAM -> 热力图", PURPLE),
        ("人工复核", "低置信度 -> 审核队列", ORANGE),
    ]
    y0 = y - 56
    for i, (title, body, color) in enumerate(lanes):
        x = MARGIN + (i % 2) * 266
        yy = y0 - (i // 2) * 112
        rounded_card(c, x, yy - 78, 238, 76, colors.white, BORDER, 10)
        c.setFillColor(color)
        c.roundRect(x + 12, yy - 30, 42, 20, 6, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont(BOLD, 8)
        c.drawCentredString(x + 33, yy - 23, f"M{i + 1}")
        c.setFillColor(INK)
        c.setFont(BOLD, 11)
        c.drawString(x + 66, yy - 23, title)
        draw_text(c, x + 66, yy - 43, body, 152, size=8.2, leading=11, color=MUTED)
    matrix_y = 170
    headers = ["样本", "临床字段", "影像", "病理", "组学", "复核"]
    rows = [
        ["P-093", "完整", "可用", "可用", "已提取", "通过"],
        ["P-112", "缺 MMR", "可用", "无切片", "待运行", "待补充"],
        ["P-128", "完整", "低置信", "可用", "已提取", "需复核"],
    ]
    draw_table(c, MARGIN, matrix_y, [72, 86, 78, 78, 86, 96], headers, rows, row_h=30, font_size=7.8)


def page_architecture(r: Report):
    c = r.c
    r.page("研究工作台架构")
    y = draw_title(c, PAGE_H - 72, "8. 建议新增 Research Workspace", "不要把科研能力塞进 Doctor tab。Research 应作为独立工作台，管理项目、论文、证据、队列、实验和产出。")
    workspaces = [
        ("Patient", "患者自述、上传、身份、分诊", RED),
        ("Doctor", "患者、诊疗问题、多模态会诊", BLUE),
        ("Research", "项目、论文、证据、队列、实验、输出", GREEN),
        ("Agent Admin", "会话、工具、规则、Trace、学习准备", PURPLE),
    ]
    x = MARGIN
    for title, body, color in workspaces:
        rounded_card(c, x, y - 88, 120, 74, colors.white, BORDER, 10)
        c.setFillColor(color)
        c.setFont(BOLD, 12)
        c.drawCentredString(x + 60, y - 38, title)
        draw_text(c, x + 12, y - 56, body, 96, size=7.3, leading=9, color=MUTED, max_lines=2)
        x += 128
    y -= 135
    c.setFillColor(INK)
    c.setFont(BOLD, 13)
    c.drawString(MARGIN, y, "建议新增核心实体")
    y -= 20
    entities = [
        ("ResearchProject", "项目空间、成员、状态、标签"),
        ("ResearchTopic", "研究主题、关键词、优先级"),
        ("Paper", "标题、作者、期刊、年份、DOI"),
        ("StudySummary", "PICO、设计、样本、终点、局限"),
        ("CohortQuery", "自然语言查询、过滤器、样本量"),
        ("Hypothesis", "假设、依据、所需数据、可行性"),
        ("ExpertOpinion", "专家角色、立场、置信度"),
        ("ResearchArtifact", "论文/SOP/报告/专利草稿"),
    ]
    for i, (title, body) in enumerate(entities):
        x = MARGIN + (i % 2) * 258
        yy = y - (i // 2) * 52
        rounded_card(c, x, yy - 38, 238, 38, LIGHT, BORDER, 7)
        c.setFillColor(RED if i < 4 else BLUE)
        c.setFont(BOLD, 8.5)
        c.drawString(x + 10, yy - 14, title)
        draw_text(c, x + 104, yy - 14, body, 122, size=7.2, leading=9, color=INK, max_lines=2)
    rounded_card(c, MARGIN, 72, CONTENT_W, 72, PINK, PINK_2, 8)
    draw_text(c, MARGIN + 14, 119,
              "编排建议：先不要新建巨大科研总图。按 Research Radar Agent、Cohort Analyst Agent、Paper Reader Agent、Evidence Mapper Agent、Research Writer Agent 逐步扩展。",
              CONTENT_W - 28, size=8.8, leading=13, color=INK)


def page_roadmap(r: Report):
    c = r.c
    r.page("路线图")
    y = draw_title(c, PAGE_H - 72, "9. 分阶段路线图", "从产品边界到科研雷达，再到队列、多模态、自动调度，逐步增加自动化能力。")
    phases = [
        ("Phase 0", "1 周", "产品定义与边界清理", "定义 Research Workspace、数据边界、安全说明"),
        ("Phase 1", "2-4 周", "医学科研雷达 MVP", "主题队列、手动搜索、去重打分、摘要、人工审核"),
        ("Phase 2", "4-8 周", "队列可行性与假设生成", "样本量、字段缺失、模态可用性、研究设计建议"),
        ("Phase 3", "8-12 周", "多模态研究样本集", "批量工具、模型输出审计、复核队列、研究矩阵"),
        ("Phase 4", "12 周以后", "自动调度与知识库入库", "scheduler、订阅、审核后入库、知识库版本管理"),
    ]
    x0 = MARGIN + 40
    y0 = y - 42
    c.setStrokeColor(PINK_2)
    c.setLineWidth(5)
    c.line(x0, y0 - 18, x0, y0 - 420)
    for i, (phase, dur, title, body) in enumerate(phases):
        yy = y0 - i * 94
        c.setFillColor(RED if i <= 1 else BLUE if i <= 3 else GREEN)
        c.circle(x0, yy - 18, 12, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont(BOLD, 8)
        c.drawCentredString(x0, yy - 21, str(i))
        rounded_card(c, x0 + 28, yy - 54, 420, 66, colors.white, BORDER, 9)
        c.setFillColor(RED)
        c.setFont(BOLD, 10)
        c.drawString(x0 + 44, yy - 14, f"{phase} / {dur}")
        c.setFillColor(INK)
        c.setFont(BOLD, 11.5)
        c.drawString(x0 + 44, yy - 32, title)
        draw_text(c, x0 + 200, yy - 14, body, 230, size=8.2, leading=11, color=MUTED, max_lines=3)


def page_risks(r: Report):
    c = r.c
    r.page("风险护栏")
    y = draw_title(c, PAGE_H - 72, "10. 风险与护栏", "医疗科研助手必须分清科研信息、临床决策、患者数据和自动化写入的边界。")
    headers = ["风险", "等级", "护栏"]
    rows = [
        ["医疗安全", "高", "科研输出不直接变成患者诊疗建议；低置信结果进入人工复核"],
        ["数据合规", "高", "默认脱敏统计；患者级明细需要权限、审计和用途记录"],
        ["证据质量", "高", "论文摘要保留 DOI/URL/片段；结论拆成 claim 并挂 evidence"],
        ["法务/IP", "中高", "专利模块只生成交底草稿，必须由法务或代理人审核"],
        ["自动化边界", "中高", "Phase 1 不自动写规则、不自动写知识库、不运行 scheduler"],
    ]
    fills = [colors.HexColor("#fff2f3"), colors.HexColor("#fff2f3"), colors.HexColor("#fff2f3"), colors.HexColor("#fff8e8"), colors.HexColor("#fff8e8")]
    draw_table(c, MARGIN, y - 5, [116, 60, 320], headers, rows, row_h=52, font_size=8, fills=fills)
    y2 = 185
    rounded_card(c, MARGIN, y2 - 100, CONTENT_W, 82, PINK, PINK_2, 10)
    c.setFillColor(DARK_RED)
    c.setFont(BOLD, 12)
    c.drawString(MARGIN + 14, y2 - 42, "必须保留的产品原则")
    draw_text(c, MARGIN + 14, y2 - 62,
              "先证据发现，后人工审核；先项目证据池，后知识库写入；先脱敏统计，后患者级明细；先手动运行，后自动调度。",
              CONTENT_W - 28, size=9.2, leading=13, color=INK)


def page_next_steps(r: Report):
    c = r.c
    r.page("落地建议")
    y = draw_title(c, PAGE_H - 72, "11. 代码改造落点与下一步", "以最小侵入方式复用现有模块，先构建 Research Radar 的可审核链路。")
    headers = ["层级", "建议落点", "说明"]
    rows = [
        ["后端", "`backend/api/routes/admin.py`", "扩展 research readiness 和 research jobs 的只读端点"],
        ["工具", "`src/tools/web_search_tools.py`", "将 LatestResearchSearchTool 接入手动科研雷达流程"],
        ["服务", "`src/services/web_search_service.py`", "复用 DeepResearchService 的拆解、来源结构化和可信度评分"],
        ["证据", "`src/rag/evidence.py`", "新增论文/研究证据 profile，保持可追溯"],
        ["数据库", "`backend/api/services/database_service.py`", "增加队列可行性和字段缺失统计"],
        ["前端", "`frontend/src/features/agent-admin`", "学习页作为科研雷达后台观测入口"],
        ["前端", "`frontend/src/features/cards`", "新增 paper_card、study_summary_card、hypothesis_card"],
    ]
    draw_table(c, MARGIN, y - 5, [62, 205, 230], headers, rows, row_h=37, font_size=7.8)
    y2 = 220
    rounded_card(c, MARGIN, y2 - 135, CONTENT_W, 118, colors.white, BORDER, 10)
    c.setFillColor(RED)
    c.setFont(BOLD, 14)
    c.drawString(MARGIN + 14, y2 - 43, "下一步建议")
    draw_bullets(c, MARGIN + 18, y2 - 65, CONTENT_W - 36, [
        "新增 Research Workspace 信息架构和实体模型设计文档。",
        "把 `search_latest_research` 产品化为手动触发的科研雷达任务。",
        "增加 paper_card / study_summary_card / evidence_delta_card。",
        "把病例库自然语言查询扩展成队列可行性报告。",
        "所有入库和产出动作先保持人工审核。",
    ], size=9.1, leading=13)


def page_sources(r: Report):
    c = r.c
    r.page("依据索引")
    y = draw_title(c, PAGE_H - 72, "12. 本地依据索引", "本报告从当前工作区代码、架构文档和用户提供草案截图形成，未使用外部网页材料。")
    sources = [
        "README.md - 当前系统总览与核心功能",
        "docs/current-architecture-map.md - 当前运行态架构",
        "src/nodes/README.md - Agent 节点说明",
        "src/tools/README.md - 工具说明",
        "src/rag/README.md - RAG 说明",
        "src/tools/manifest.py - 工具 manifest 与 search_latest_research 候选状态",
        "src/services/web_search_service.py - 联网搜索与 Deep Research 服务",
        "src/tools/web_search_tools.py - 临床证据、指南更新、最新研究搜索工具",
        "backend/api/routes/database.py - 数据库 API",
        "backend/api/services/database_service.py - 病例库统计与筛选服务",
        "frontend/src/features/agent-admin - 后台学习准备、证据池、Trace 页面",
        "用户提供草案截图 - 小亿灵析科研分析助手内部草案",
    ]
    draw_bullets(c, MARGIN + 5, y - 5, CONTENT_W - 10, sources, size=8.8, leading=13)
    draw_screenshot(c, MARGIN + 45, 78, CONTENT_W - 90, 240)


def build_pdf():
    r = Report(OUTPUT)
    page_cover(r)
    page_summary(r)
    page_current_capabilities(r)
    page_alignment_heatmap(r)
    page_priority_chart(r)
    page_mvp_flow(r)
    page_cohort(r)
    page_multimodal(r)
    page_architecture(r)
    page_roadmap(r)
    page_risks(r)
    page_next_steps(r)
    page_sources(r)
    r.save()
    print(OUTPUT)


if __name__ == "__main__":
    build_pdf()
