from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "langg-agent-development-validation-plan-2026-06-29.pdf"

PAGE_W, PAGE_H = A4

PALETTE = {
    "red": colors.HexColor("#c9142f"),
    "dark_red": colors.HexColor("#8d1021"),
    "red_soft": colors.HexColor("#fff1f3"),
    "ink": colors.HexColor("#1f2328"),
    "muted": colors.HexColor("#5d6673"),
    "line": colors.HexColor("#e3e7ee"),
    "panel": colors.HexColor("#f7f8fa"),
    "blue": colors.HexColor("#2f6fbd"),
    "blue_soft": colors.HexColor("#eef5ff"),
    "green": colors.HexColor("#2f9e68"),
    "green_soft": colors.HexColor("#ecf8f2"),
    "orange": colors.HexColor("#e7902e"),
    "orange_soft": colors.HexColor("#fff6e8"),
    "purple": colors.HexColor("#8250df"),
    "purple_soft": colors.HexColor("#f4efff"),
    "teal": colors.HexColor("#1f8a8a"),
    "teal_soft": colors.HexColor("#e9f7f7"),
}


def register_fonts() -> tuple[str, str]:
    regular_candidates = [
        ("MicrosoftYaHei", Path("C:/Windows/Fonts/msyh.ttc"), {"subfontIndex": 0}),
        ("NotoSansSC", Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"), {}),
        ("SimHei", Path("C:/Windows/Fonts/simhei.ttf"), {}),
        ("SimSun", Path("C:/Windows/Fonts/simsun.ttc"), {"subfontIndex": 0}),
    ]
    bold_candidates = [
        ("MicrosoftYaHeiBold", Path("C:/Windows/Fonts/msyhbd.ttc"), {"subfontIndex": 0}),
        ("SimHei", Path("C:/Windows/Fonts/simhei.ttf"), {}),
        ("NotoSansSCBold", Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"), {}),
    ]
    regular_name = "Helvetica"
    bold_name = "Helvetica-Bold"
    for name, path, kwargs in regular_candidates:
        if path.exists():
            pdfmetrics.registerFont(TTFont(name, str(path), **kwargs))
            regular_name = name
            break
    for name, path, kwargs in bold_candidates:
        if path.exists():
            pdfmetrics.registerFont(TTFont(name, str(path), **kwargs))
            bold_name = name
            break
    return regular_name, bold_name


BODY_FONT, BOLD_FONT = register_fonts()


def make_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=sample["Title"],
            fontName=BOLD_FONT,
            fontSize=24,
            leading=32,
            textColor=colors.white,
            alignment=TA_LEFT,
            wordWrap="CJK",
            spaceAfter=14,
        ),
        "cover_subtitle": ParagraphStyle(
            "cover_subtitle",
            parent=sample["BodyText"],
            fontName=BODY_FONT,
            fontSize=10.5,
            leading=17,
            textColor=colors.HexColor("#ffe9ec"),
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=sample["Heading1"],
            fontName=BOLD_FONT,
            fontSize=17,
            leading=23,
            textColor=PALETTE["dark_red"],
            alignment=TA_LEFT,
            wordWrap="CJK",
            spaceBefore=8,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=sample["Heading2"],
            fontName=BOLD_FONT,
            fontSize=13,
            leading=18,
            textColor=PALETTE["ink"],
            alignment=TA_LEFT,
            wordWrap="CJK",
            spaceBefore=8,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=sample["BodyText"],
            fontName=BODY_FONT,
            fontSize=9.2,
            leading=14.2,
            textColor=PALETTE["ink"],
            alignment=TA_LEFT,
            wordWrap="CJK",
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "small",
            parent=sample["BodyText"],
            fontName=BODY_FONT,
            fontSize=7.8,
            leading=11.5,
            textColor=PALETTE["muted"],
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "table_header",
            parent=sample["BodyText"],
            fontName=BOLD_FONT,
            fontSize=8.0,
            leading=10.5,
            textColor=colors.white,
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "table_cell": ParagraphStyle(
            "table_cell",
            parent=sample["BodyText"],
            fontName=BODY_FONT,
            fontSize=7.4,
            leading=10.5,
            textColor=PALETTE["ink"],
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "table_cell_bold": ParagraphStyle(
            "table_cell_bold",
            parent=sample["BodyText"],
            fontName=BOLD_FONT,
            fontSize=7.5,
            leading=10.5,
            textColor=PALETTE["ink"],
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "card_title": ParagraphStyle(
            "card_title",
            parent=sample["BodyText"],
            fontName=BOLD_FONT,
            fontSize=10,
            leading=14,
            textColor=PALETTE["dark_red"],
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "card_body": ParagraphStyle(
            "card_body",
            parent=sample["BodyText"],
            fontName=BODY_FONT,
            fontSize=8.2,
            leading=12.3,
            textColor=PALETTE["ink"],
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "badge": ParagraphStyle(
            "badge",
            parent=sample["BodyText"],
            fontName=BOLD_FONT,
            fontSize=8,
            leading=10,
            textColor=PALETTE["dark_red"],
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
    }


STYLES = make_styles()


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), STYLES[style])


def bullets(items: Iterable[str]) -> list[Paragraph]:
    return [p(f"• {item}") for item in items]


def table(
    headers: list[str],
    rows: list[list[str]],
    widths: list[float],
    *,
    header_color=PALETTE["dark_red"],
    zebra=True,
    repeat=1,
) -> LongTable:
    data = [[p(h, "table_header") for h in headers]]
    for row in rows:
        data.append([p(cell, "table_cell") for cell in row])
    t = LongTable(data, colWidths=widths, repeatRows=repeat, splitByRow=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("BOX", (0, 0), (-1, -1), 0.6, PALETTE["line"]),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, PALETTE["line"]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if zebra:
        for i in range(1, len(rows) + 1):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), PALETTE["panel"]))
    t.setStyle(TableStyle(style))
    return t


def card(title: str, body: str, *, accent=PALETTE["red_soft"]) -> Table:
    t = Table(
        [[p(title, "card_title")], [p(body, "card_body")]],
        colWidths=[162],
        rowHeights=None,
    )
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), accent),
                ("BOX", (0, 0), (-1, -1), 0.7, PALETTE["line"]),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return t


def card_grid(cards: list[Table], cols: int = 3) -> Table:
    rows = []
    for i in range(0, len(cards), cols):
        chunk = cards[i : i + cols]
        while len(chunk) < cols:
            chunk.append(Spacer(1, 1))
        rows.append(chunk)
    t = Table(rows, colWidths=[166] * cols, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return t


def section(story: list, title: str, intro: str | None = None) -> None:
    story.append(p(title, "h1"))
    if intro:
        story.append(p(intro))


def subsection(story: list, title: str) -> None:
    story.append(p(title, "h2"))


def on_first_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(PALETTE["dark_red"])
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(PALETTE["red"])
    canvas.rect(0, PAGE_H - 36 * mm, PAGE_W, 36 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont(BOLD_FONT, 10)
    canvas.drawString(20 * mm, PAGE_H - 20 * mm, "LangG CRC Agent - Strategy PDF")
    canvas.setFont(BODY_FONT, 8)
    canvas.drawRightString(PAGE_W - 20 * mm, PAGE_H - 20 * mm, "2026-06-29")
    canvas.restoreState()


def on_later_pages(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(PALETTE["red"])
    canvas.roundRect(15 * mm, PAGE_H - 14 * mm, 4 * mm, 8 * mm, 1.5 * mm, fill=1, stroke=0)
    canvas.setFillColor(PALETTE["ink"])
    canvas.setFont(BOLD_FONT, 8.5)
    canvas.drawString(21 * mm, PAGE_H - 10 * mm, "医疗软件功能增加与智能体改进方案")
    canvas.setFillColor(PALETTE["muted"])
    canvas.setFont(BODY_FONT, 7.5)
    canvas.drawRightString(PAGE_W - 15 * mm, PAGE_H - 10 * mm, "结合当前 git 状态、架构文档与实际代码")
    canvas.setStrokeColor(PALETTE["line"])
    canvas.line(15 * mm, 13 * mm, PAGE_W - 15 * mm, 13 * mm)
    canvas.setFillColor(PALETTE["muted"])
    canvas.setFont(BODY_FONT, 7)
    canvas.drawString(15 * mm, 8 * mm, "内部初步方案 - 不替代医疗、法律或合规审查")
    canvas.drawRightString(PAGE_W - 15 * mm, 8 * mm, str(doc.page))
    canvas.restoreState()


def build_story() -> list:
    story: list = []

    story.append(Spacer(1, 64 * mm))
    story.append(p("医疗软件功能增加与智能体改进详细方案", "cover_title"))
    story.append(
        p(
            "基于当前 LangG 仓库、crc-client-integration-verification 分支、CRC triage WIP、"
            "FastAPI + LangGraph + React 架构、Agent Admin 观测面、RAG/联网搜索/验收体系，"
            "提出医疗软件功能扩展与智能体改进的阶段化方案。",
            "cover_subtitle",
        )
    )
    story.append(Spacer(1, 12 * mm))
    cover_cards = [
        card("产品主线", "以 CRC 专项问诊、患者记录、医生复核、报告草稿和多模态工作台为第一条纵向功能切片。", accent=colors.HexColor("#fff7f7")),
        card("智能体主线", "围绕医生蒸馏、自动文献 harness、评测 harness、RAG 证据链和工具治理逐步增强。", accent=colors.HexColor("#fff7f7")),
        card("验证主线", "每个增量必须留下产品行为、状态持久化、评测证据和可观测 hook。", accent=colors.HexColor("#fff7f7")),
    ]
    story.append(card_grid(cover_cards))
    story.append(Spacer(1, 8 * mm))
    story.append(
        p(
            "输出文件: output/pdf/langg-agent-development-validation-plan-2026-06-29.pdf<br/>"
            "生成脚本: scripts/generate_agent_development_validation_plan_pdf.py",
            "cover_subtitle",
        )
    )
    story.append(PageBreak())

    section(
        story,
        "1. 执行摘要",
        "本方案的核心判断是: 当前项目不应把功能开发、agent 评测和前沿实验拆成三条互不相干的路线。"
        "最稳妥的做法是选择一个真实产品切片作为共同载体。基于当前 git 状态，CRC 专项问诊是第一条合适的纵向切片。",
    )
    story.extend(
        bullets(
            [
                "医疗软件功能增加优先围绕患者侧 CRC triage、患者记录/随访卡、医生侧复核和报告草稿、研究工作台入口展开。",
                "智能体改进优先做医生蒸馏数据闭环、自动文献 harness、评测 harness、RAG 证据链、工具 manifest 和 Agent Admin 观测。",
                "所有前沿能力先以 shadow mode 和人工审核方式接入，不直接改动患者或医生主路径。",
                "当前阶段建议投入比例为 55% 功能闭环、30% 验证 harness、15% 前沿实验；CRC 纵向切片稳定后调整为 45%/35%/20%。",
                "方案不把 `CRC-client` 作为运行时子应用接入，而是吸收协议和体验，把数据写回 LangG session、patient registry 和 PatientCommandService。",
            ]
        )
    )
    story.append(
        table(
            ["判断项", "当前证据", "方案决策"],
            [
                ["仓库主线", "`crc-client-integration-verification` 分支和未提交 WIP 集中在 CRC triage、patient records、care cards、routing。", "先稳定 CRC 纵向切片，避免同时铺开研究平台、生产部署和大规模 UI 改造。"],
                ["agent 基础", "`src/graph_builder.py` 已有 doctor/patient 双图、critic、citation、evaluator、node_timings。", "把评测和观测接入真实 graph 输出，而不是新建孤立 benchmark。"],
                ["自动文献基础", "`search_latest_research` 已存在但在 manifest 中是 candidate/executor-only。", "第一阶段做手动触发 + 人工审核 + evidence staging，不直接自动入库。"],
                ["后台观察", "Agent Admin 已有工具 manifest、trace/evidence/memory/rules 设计和部分实现。", "作为观测面和学习准备面，不阻塞 CRC MVP。"],
            ],
            [72, 216, 210],
        )
    )
    story.append(PageBreak())

    section(
        story,
        "2. 当前项目事实和代码依据",
        "本节只列与方案设计直接相关的事实。未提交工作树中的 CRC triage 相关文件被视为当前 WIP，而不是已稳定发布能力。",
    )
    story.append(
        table(
            ["层级", "关键文件或模块", "对方案的含义"],
            [
                ["BFF/运行时", "`backend/app.py`, `backend/api/services/graph_service.py`, `session_store*`, `payload_builder.py`", "已有 FastAPI runtime、patient/doctor graph service、POST SSE、session lock 和 snapshot 边界。新功能应复用这些边界。"],
                ["患者侧 CRC", "`frontend/src/features/patient-crc-triage/*`, `src/services/crc_triage_flow.py`, `backend/api/routes/crc_triage.py`", "CRC 专项问诊已经进入实现阶段，适合作为功能和验证共同载体。"],
                ["患者数据", "`PatientCommandService`, `patient_registry_service.py`, `patient_care_cards.py`, `frontend/src/features/patient-records/*`", "completed assessment 应写入 patient records，并派生 care cards。不能另开 CSV/localStorage 源。"],
                ["医生侧智能体", "`src/graph_builder.py`, `src/nodes/*`, `frontend/src/features/doctor/*`", "doctor graph 是复杂临床推理、RAG、工具调用、critic/evaluator 的主要场景，适合做医生蒸馏和复核闭环。"],
                ["RAG/文献", "`src/rag/*`, `src/tools/rag_tools.py`, `src/tools/web_search_tools.py`, `src/services/web_search_service.py`", "已有指南 RAG、联网搜索、DeepResearchService 雏形和 candidate 文献搜索工具。"],
                ["可观测/后台", "`src/tools/manifest.py`, `frontend/src/features/agent-admin/*`, `backend/api/routes/admin.py`", "工具 manifest 和 Agent Admin 可承载工具可达性、学习准备、trace、evidence、memory 状态。"],
                ["验证体系", "`tests/backend/*`, `frontend/src/**/*.test.tsx`, `tests/e2e/*`, `scripts/run_*playwright*`", "已有 pytest/Vitest/Playwright 基础，新增功能必须进入 case pack，而不是只靠人工演示。"],
            ],
            [68, 178, 252],
        )
    )
    subsection(story, "2.1 当前 git 状态对方案的约束")
    story.extend(
        bullets(
            [
                "当前分支为 `crc-client-integration-verification`，HEAD 为 `25ee455 Add CRC triage assessment flow`。",
                "工作树有大量 CRC triage、patient records、general routing、frontend style 和 backend test WIP。方案不建议在此时叠加 Redis run lock、OIDC、SSE resume 等生产化大改。",
                "已有 `docs/superpowers/specs/2026-06-29-agent-development-validation-balance-design.md`，本 PDF 在其基础上扩展医生蒸馏、自动文献 harness 和医疗软件功能路线。",
            ]
        )
    )
    story.append(PageBreak())

    section(story, "3. 总体架构方向", "新增能力分成四个工作面: Patient、Doctor、Research、Agent Admin。每个工作面复用当前 runtime，但边界不同。")
    story.append(
        table(
            ["工作面", "目标用户", "主要对象", "优先能力"],
            [
                ["Patient", "患者", "session、identity、uploads、CRC triage assessment", "CRC 专项问诊、资料上传、个人记录、随访卡、PDF 材料导出。"],
                ["Doctor", "医生", "bound patient、clinical plan、decision、report draft", "患者上下文复核、医生报告草稿、多模态结果、指南证据、人工修正采集。"],
                ["Research", "医生/PI/研发", "project、paper、cohort、hypothesis、evidence delta", "自动文献 harness、队列可行性、假设生成、多模态研究样本集。"],
                ["Agent Admin", "工程/运营/管理", "tools、rules、trace、memory、learning readiness", "工具 manifest、运行 trace、证据池、学习任务状态、评测报告入口。"],
            ],
            [58, 62, 150, 228],
            header_color=PALETTE["red"],
        )
    )
    subsection(story, "3.1 技术原则")
    story.extend(
        bullets(
            [
                "保持 patient/doctor scene 和未来 research workspace 的边界清晰，不把 Agent Admin 当第三个 graph scene。",
                "患者事实只通过 session、patient registry、PatientCommandService、records/care cards 传播。",
                "医生修正和评分先作为蒸馏数据候选，不直接覆盖模型或永久规则。",
                "自动文献只进入候选证据池，人工审核后才能进入项目知识库或 RAG 候选区。",
                "每个新能力必须附带 case pack、snapshot/API 证据和可观测字段。",
            ]
        )
    )
    story.append(PageBreak())

    section(story, "4. 医疗软件功能增加路线", "功能扩展以低风险、高复用为优先，先完成患者侧闭环，再加强医生侧复核，最后形成研究工作台。")
    story.append(
        table(
            ["优先级", "功能模块", "范围", "落地文件或接口", "验证证据"],
            [
                ["P0", "CRC 专项问诊闭环", "患者开始问诊、结构化问题、风险分层、缺失信息、完成摘要、保存 assessment。", "`patient-crc-triage/*`, `crc_triage.py`, `crc_triage_flow.py`, `patient_triage_protocol.py`", "协议单测、API 保存测试、前端 completed state 测试、session snapshot。"],
                ["P0", "患者记录和 care cards", "将 `crc_triage_assessment` 写入 records，并派生重点关注、周期检查、日常行动卡。", "`patient_commands.py`, `patient_care_cards.py`, `frontend/src/features/patient-records/*`", "record id/event id、projection version、care card payload 和 UI 一致性。"],
                ["P1", "医生复核和报告草稿", "医生绑定患者后读取 CRC assessment、生成报告草稿、标记需复核字段。", "`DoctorGraphService`, `doctor-report-draft-*`, `sessions.py`", "doctor session 注入患者摘要、报告草稿测试、人工修正记录。"],
                ["P1", "上传报告和问诊联动", "患者上传肠镜/病理/影像报告后补全 triage missing info。", "`uploads.py`, `upload_service.py`, `document_converter.py`, `payload_builder.py`", "上传 asset、processed file、context message、triage summary 更新。"],
                ["P1", "指南变更和证据提示", "医生侧显示与当前患者或项目相关的指南更新、证据变化。", "`rag_tools.py`, `web_search_tools.py`, `retrieved_evidence`, `references`", "RAG trace、citation coverage、人工审核状态。"],
                ["P2", "多模态肿瘤研究工作台", "从病例筛选生成样本集，运行影像/病理/组学工具，形成研究矩阵。", "`tumor_*`, `radiomics_tools.py`, `pathology_clam_tools.py`, doctor multimodal view", "样本集版本、工具版本、输出卡片、人工复核记录。"],
            ],
            [38, 74, 130, 138, 118],
        )
    )
    subsection(story, "4.1 CRC triage 的第一条验收切片")
    story.extend(
        bullets(
            [
                "低危常规问诊: 问题顺序、final assessment、suggested tests、无急诊 disposition。",
                "高危红旗: fatal/red flag 触发 urgent 或 emergency，不被 routine flow 覆盖。",
                "中危补问: 单个红旗或不确定信号触发 backfill，不提前归档。",
                "肠镜信息缺失: 提到肠镜但无关键结果时保留 missing_information，并引导上传或补充。",
                "问诊中切换话题: CRC state 可恢复，普通 patient assistant 不被污染。",
                "保存至患者记录: API 返回 patient/version/event/record，records 和 care cards 同步。",
            ]
        )
    )
    story.append(PageBreak())

    section(story, "5. 医生蒸馏方案", "医生蒸馏不是把医生文字直接喂给模型，而是把医生的复核、改写、驳回、补充证据和最终采纳记录转成可审计训练/评测数据。")
    story.append(
        table(
            ["阶段", "目标", "输入", "输出", "安全边界"],
            [
                ["D0 采集", "记录医生对 agent 输出的人工修正和采纳情况。", "doctor graph 输出、report draft、CRC assessment、医生编辑、critic/evaluator 信号。", "ReviewEvent、CorrectionPair、AcceptedDraft、RejectedReason。", "默认脱敏，不记录隐藏推理，不直接训练。"],
                ["D1 标注", "把医生修正转成结构化偏好和 rubric。", "修正文案、字段变更、证据缺失、风险提示。", "标签: factuality、safety、completeness、citation、tone、workflow fit。", "医生可撤回，敏感字段最小化。"],
                ["D2 蒸馏", "用高质量医生偏好训练 prompt、routing policy 或小模型适配器。", "已审核 CorrectionPair、golden cases、失败案例。", "Prompt patch、judge rubric、routing rule、可选 SFT/DPO 数据包。", "先离线评测，不能直接上线覆盖主路径。"],
                ["D3 验证", "证明蒸馏改进优于 baseline。", "CRC case pack、doctor decision pack、RAG evidence pack。", "胜率、红旗召回、引用覆盖、医生编辑距离、延迟、失败率。", "未通过阈值只保留为实验结果。"],
                ["D4 发布", "把通过验证的改动小步进入生产路径。", "评测报告、人工 sign-off、回滚点。", "feature flag、版本号、release note。", "保留回滚和审计链。"],
            ],
            [44, 94, 122, 122, 116],
            header_color=PALETTE["purple"],
        )
    )
    subsection(story, "5.1 可复用的当前代码基础")
    story.extend(
        bullets(
            [
                "`src/graph_builder.py` 中 doctor graph 已包含 `critic`, `citation`, `evaluator`, `node_timings`，可作为蒸馏候选数据的上下文。",
                "`frontend/src/features/doctor/doctor-report-draft-*` 可成为医生编辑和采纳事件的入口。",
                "`backend/api/services/patient_registry_service.py` 和 patient records 可提供患者事实上下文，但训练数据必须先脱敏。",
                "`frontend/src/features/agent-admin` 可展示蒸馏数据准备度、样本数量、失败类型和最近评测结果。",
            ]
        )
    )
    subsection(story, "5.2 第一版医生蒸馏数据结构")
    story.append(
        table(
            ["对象", "关键字段", "用途"],
            [
                ["DoctorReviewEvent", "session_id, patient_id, source_node, original_output, edited_output, action, reviewer_id, timestamp", "记录医生采纳、修改、拒绝 agent 输出的事实。"],
                ["CorrectionPair", "input_context_hash, original_answer, corrected_answer, correction_type, safety_tags, evidence_tags", "形成 SFT/DPO 或 prompt-rubric 数据候选。"],
                ["RubricScore", "factuality, completeness, safety, citation, patient_specificity, doctor_workflow_fit", "评价候选输出是否优于 baseline。"],
                ["DistillationRun", "dataset_version, model_or_prompt_version, metrics, failed_cases, approved_by", "跟踪一次蒸馏实验和发布门槛。"],
            ],
            [90, 214, 194],
            header_color=PALETTE["purple"],
        )
    )
    story.append(PageBreak())

    section(story, "6. Harness 自动文献方案", "自动文献 harness 的目标是持续发现和筛选医学证据，而不是自动把未审核内容写入临床 RAG。")
    story.append(
        table(
            ["阶段", "能力", "代码基础", "新增对象", "人工审核点"],
            [
                ["H0 手动雷达", "研究主题手动触发 `search_latest_research` 和 guideline update。", "`LatestResearchSearchTool`, `WebSearchService.search_research`, `src/tools/manifest.py` candidate 状态。", "ResearchTopic, LiteratureRun, PaperCandidate", "用户确认主题和来源范围。"],
                ["H1 结构化抽取", "从论文/指南结果提取 PICO、样本量、终点、结论、局限性。", "`ResearchResult`, `SourceItem`, `document_converter`, RAG evidence helpers。", "PaperSummary, EvidenceDelta, StudyDesignCard", "人工审核摘要和证据等级。"],
                ["H2 证据池", "审核通过后进入项目证据池，不直接进入全局 RAG。", "Agent Admin evidence page, `retrieved_evidence`, `references`。", "ProjectEvidenceItem, EvidenceDecision", "PI/医生确认 include/exclude。"],
                ["H3 入库预览", "对候选证据生成 RAG chunk 预览和冲突检查。", "`src/rag/ingest.py`, `bm25_index.py`, `retriever.py`。", "IngestPreview, ConflictReport", "管理员批准后才构建索引。"],
                ["H4 定时任务", "主题订阅、定时运行、失败重试、结果通知。", "未来 `agent_learning_jobs` 或 scheduler。", "LearningJob, RunLog, Notification", "默认人工审核，禁止静默影响临床建议。"],
            ],
            [42, 104, 142, 116, 94],
            header_color=PALETTE["teal"],
        )
    )
    subsection(story, "6.1 Harness 输出卡片")
    story.extend(
        bullets(
            [
                "paper_card: 标题、来源、年份、研究类型、样本量、主要终点、可信度。",
                "evidence_delta_card: 与本地指南/RAG 版本相比新增、变化、冲突或待确认的结论。",
                "study_summary_card: PICO、方法、结果、局限性、适用人群、是否与本地 CRC 队列相关。",
                "ingest_preview_card: 预计 chunk 数、来源 hash、可能重复、冲突项、人工审核状态。",
            ]
        )
    )
    subsection(story, "6.2 安全规则")
    story.extend(
        bullets(
            [
                "自动文献结果不得直接出现在患者端建议中。",
                "未审核 evidence 只能显示为候选或研究信息，不能标记为指南事实。",
                "所有外部来源必须保留 URL、来源类型、年份、摘要和可信度评分。",
                "需要保留 negative evidence 和 conflicting evidence，不能只保存支持性证据。",
                "每次入库必须可回滚，并记录审核人、时间和索引版本。",
            ]
        )
    )
    story.append(PageBreak())

    section(story, "7. 评测 Harness 方案", "评测 harness 负责回答两个问题: 功能是否能跑通，以及智能体是否真的更好。")
    story.append(
        table(
            ["评测包", "覆盖范围", "关键指标", "执行方式"],
            [
                ["CRC triage pack", "低危、高危、中危补问、肠镜缺失、话题切换、保存 records。", "完成率、红旗召回、missing info、state 一致性。", "pytest + Vitest + Playwright fixture。"],
                ["Doctor decision pack", "医生绑定患者、RAG、治疗决策、critic、citation、evaluator、report draft。", "引用覆盖、医生编辑距离、人工采纳率、关键事实缺失。", "fixture graph + snapshot + manual review。"],
                ["RAG evidence pack", "指南检索、章节读取、treatment/staging/drug profiles、web fallback。", "hit rate、citation accuracy、retrieval latency、source freshness。", "golden query + `rag_trace` inspection。"],
                ["Literature harness pack", "search_latest_research、paper summary、evidence delta、人工审核。", "去重率、结构化字段完整率、无来源率、审核通过率。", "offline fixture + controlled live run。"],
                ["Distillation pack", "医生修正前后输出比较、rubric judge、shadow model。", "胜率、安全不退化、延迟、失败 case 数。", "baseline vs candidate replay。"],
            ],
            [74, 148, 136, 140],
            header_color=PALETTE["blue"],
        )
    )
    subsection(story, "7.1 Gate 设计")
    story.append(
        table(
            ["Gate", "必须证明", "证据类型"],
            [
                ["产品行为", "用户能完成目标流程，错误状态可恢复。", "UI 截图、Playwright trace、最终可见文本。"],
                ["状态边界", "UI、session snapshot、patient records、cards 或 evidence pool 一致。", "API response、snapshot、数据库投影、reducer state。"],
                ["智能体质量", "路由、证据、风险提示、问诊追问、输出结构满足 case 预期。", "golden case、judge rubric、人工复核。"],
                ["可观测性", "能解释为什么通过或失败。", "trace、node_timings、rag_trace、event log、review events。"],
            ],
            [64, 252, 182],
            header_color=PALETTE["blue"],
        )
    )
    story.append(PageBreak())

    section(story, "8. RAG、工具和记忆改进路线", "智能体前沿性不只来自更强模型，也来自更清晰的工具边界、证据链、记忆治理和可回滚实验。")
    story.append(
        table(
            ["方向", "当前基础", "改进动作", "风险控制"],
            [
                ["RAG 证据链", "`retrieved_references`, `rag_trace`, RAG evidence contract。", "补齐 `retrieved_evidence` 到 admin/API 输出，按 general/treatment/staging/drug/research profile 评估。", "引用必须可追溯，不依赖模型手写来源。"],
                ["工具 manifest", "`src/tools/manifest.py`, `/api/admin/tools`。", "增加 health/readiness 状态，但仍不在 manifest endpoint 实例化重型工具。", "防止模型权重加载、网络调用和敏感路径泄露。"],
                ["路由策略", "`routing_policy.py`, `tool_targets.py`, graph router。", "为 CRC triage、doctor decision、research workspace 定义独立 route facts 和 golden fixtures。", "先 shadow，对主路径保守发布。"],
                ["记忆治理", "summary_memory、structured_summary、context maintenance。", "区分患者事实、医生偏好、科研项目记忆、蒸馏候选数据。", "不把 session memory 当长期机构知识库。"],
                ["模型前沿", "LLMService、provider_capabilities。", "引入 provider capability matrix，按任务选择模型: triage、judge、summary、literature extraction。", "每个模型版本必须过 case pack。"],
            ],
            [72, 126, 184, 116],
            header_color=PALETTE["green"],
        )
    )
    subsection(story, "8.1 推荐的智能体版本化对象")
    story.extend(
        bullets(
            [
                "AgentPolicyVersion: routing、prompt、tool scope、review policy 的版本组合。",
                "EvidenceIndexVersion: RAG collection、BM25 index、文献候选区、审核状态。",
                "JudgeRubricVersion: evaluator/critic 的评分维度、阈值和降级策略。",
                "DistillationDatasetVersion: 医生修正样本、脱敏策略、适用场景、排除样本。",
                "HarnessRun: 输入 case pack、agent version、输出指标、失败明细和人工 sign-off。",
            ]
        )
    )
    story.append(PageBreak())

    section(story, "9. 分阶段实施路线图", "路线图按当前 WIP 收敛、验证强化、研究工作台、前沿产品化四个阶段推进。")
    story.append(
        table(
            ["阶段", "时间窗口", "功能交付", "智能体交付", "退出标准"],
            [
                ["Phase 0", "1-2 周", "收敛 CRC triage WIP: tab、flow、assessment save、records/care cards。", "CRC case pack v0、协议单测、API 保存测试、前端 completed state 测试。", "完成 assessment 保存并可在 records/care cards 中复现。"],
                ["Phase 1", "2-4 周", "医生侧读取 CRC assessment，生成报告草稿，上传报告补全 triage。", "doctor review events、基础医生蒸馏数据 schema、decision pack v0。", "医生修正可记录，且不影响现有 doctor flow。"],
                ["Phase 2", "4-8 周", "Research workspace MVP: 研究主题、手动文献雷达、paper/evidence cards。", "literature harness v0、paper summary rubric、evidence staging。", "手动运行文献搜索并生成审核卡片，不自动入临床 RAG。"],
                ["Phase 3", "8-12 周", "队列可行性、研究样本集、多模态研究矩阵。", "distillation shadow eval、RAG profile eval、tool health/readiness。", "有 baseline vs candidate 评测报告，前沿能力可按 feature flag 小步发布。"],
                ["Phase 4", "12 周后", "定时文献任务、知识资产库、论文/基金/专利草稿辅助。", "自动学习 job、index versioning、医生蒸馏模型或 prompt 发布流程。", "具备人工审核、回滚、合规和长期维护机制。"],
            ],
            [44, 54, 150, 150, 100],
            header_color=PALETTE["dark_red"],
        )
    )
    subsection(story, "9.1 近期不建议做")
    story.extend(
        bullets(
            [
                "不要直接把 `CRC-client` 作为子应用嵌入主 React 应用。",
                "不要让自动文献结果直接进入患者建议或临床决策主路径。",
                "不要在 CRC WIP 未收敛前同时实施 Redis 分布式锁、SSE resume、OIDC 等生产化大改。",
                "不要把医生修正直接等同于可训练数据，必须先脱敏、标注、审核和版本化。",
                "不要只用 LLM judge 判断医疗质量，必须保留人工复核和状态一致性证据。",
            ]
        )
    )
    story.append(PageBreak())

    section(story, "10. 治理、安全和合规边界", "医疗软件和智能体改进必须把安全边界前置。下面是第一阶段就要遵守的规则。")
    story.append(
        table(
            ["边界", "规则", "落地动作"],
            [
                ["患者安全", "患者端输出是分诊和下一步建议，不是最终诊断或治疗方案。", "红旗症状强制提示线下就医，missing_information 保留在 summary 和 records。"],
                ["医生责任", "医生侧输出是辅助草稿，最终采纳、修改、签署由医生完成。", "记录 accepted/rejected/edited 事件，作为蒸馏候选而非自动真值。"],
                ["数据脱敏", "蒸馏、文献、评测数据不能默认包含可识别患者信息。", "使用 context hash、字段最小化、导出审批和审计日志。"],
                ["文献入库", "外部文献必须人工审核后才进入项目证据池或 RAG。", "EvidenceDecision 记录 reviewer、decision、reason、version。"],
                ["隐藏推理", "后台和报告不得泄露 hidden chain-of-thought、prompt secrets、API key。", "admin state sanitizer、测试扫描、review gate。"],
                ["版本回滚", "prompt、policy、RAG index、distillation dataset、judge rubric 都要可追溯。", "引入 version object 和 harness run 报告。"],
            ],
            [62, 230, 206],
            header_color=PALETTE["orange"],
        )
    )
    story.append(PageBreak())

    section(story, "11. 详细交付清单", "下面的清单把功能、智能体、验证和文档交付物合并管理，便于后续写 implementation plan。")
    story.append(
        table(
            ["类别", "交付物", "优先级", "完成证据"],
            [
                ["功能", "CRC triage 完成态和保存态", "P0", "UI completed state、save API response、patient record、care card。"],
                ["功能", "Patient records 面板接入 CRC assessment", "P0", "records API 和 frontend test。"],
                ["功能", "Doctor 复核和 report draft 读取 CRC assessment", "P1", "doctor session snapshot、draft UI、医生编辑事件。"],
                ["智能体", "医生蒸馏数据 schema 和采集入口", "P1", "DoctorReviewEvent/CorrectionPair 测试和脱敏检查。"],
                ["智能体", "文献 harness 手动雷达", "P1", "ResearchTopic/LiteratureRun/PaperCandidate 和 paper_card。"],
                ["智能体", "RAG evidence/admin 输出", "P1", "`retrieved_evidence`, `rag_trace`, citation report 可见。"],
                ["验证", "CRC case pack v0", "P0", "pytest/Vitest/Playwright 或 API snapshot 覆盖 6 个场景。"],
                ["验证", "doctor decision pack v0", "P1", "baseline replay 和人工复核清单。"],
                ["验证", "distillation shadow report", "P2", "baseline vs candidate 指标报告。"],
                ["治理", "版本化对象定义", "P2", "AgentPolicyVersion、EvidenceIndexVersion、JudgeRubricVersion 草案。"],
                ["文档", "本 PDF 和后续 implementation plan", "P0", "PDF 产物、计划文档、review gate。"],
            ],
            [58, 206, 42, 192],
            header_color=PALETTE["dark_red"],
        )
    )
    story.append(PageBreak())

    section(story, "12. 附录: 建议参考的当前文件", "这些路径是后续实现计划和代码审查时最应该优先阅读的文件。")
    story.append(
        table(
            ["主题", "文件"],
            [
                ["当前平衡设计", "`docs/superpowers/specs/2026-06-29-agent-development-validation-balance-design.md`"],
                ["CRC triage 设计", "`docs/superpowers/specs/2026-06-24-patient-crc-triage-subpage-design.md`"],
                ["Agent Admin", "`docs/superpowers/specs/2026-06-14-agent-admin-phase-one-design.md`"],
                ["E2E 验收", "`docs/superpowers/specs/2026-04-11-e2e-full-acceptance-design.md`"],
                ["RAG evidence", "`docs/superpowers/specs/2026-04-29-rag-evidence-contract-design.md`"],
                ["运行时入口", "`backend/app.py`, `backend/api/services/graph_service.py`, `backend/api/services/graph_factory.py`"],
                ["Graph 构建", "`src/graph_builder.py`, `src/state.py`, `src/nodes/*`, `src/policies/*`"],
                ["患者数据", "`backend/api/services/patient_commands.py`, `patient_registry_service.py`, `patient_care_cards.py`"],
                ["文献和联网搜索", "`src/tools/web_search_tools.py`, `src/services/web_search_service.py`, `src/tools/manifest.py`"],
                ["多模态工具", "`src/tools/tumor_screening_tools.py`, `tumor_localization_tools.py`, `radiomics_tools.py`, `pathology_clam_tools.py`"],
                ["前端工作台", "`frontend/src/pages/workspace-page.tsx`, `frontend/src/features/patient-crc-triage/*`, `frontend/src/features/doctor/*`, `frontend/src/features/agent-admin/*`"],
            ],
            [90, 408],
            header_color=PALETTE["red"],
        )
    )
    story.append(Spacer(1, 8 * mm))
    story.append(
        p(
            "结论: 当前最稳妥的路线不是单独做科研助手、单独做评测平台或单独堆前沿模型，"
            "而是以 CRC triage 为第一条纵向产品切片，逐步接入医生蒸馏、自动文献 harness、评测 harness 和 Agent Admin 观测治理。"
        )
    )
    return story


def build_pdf() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="医疗软件功能增加与智能体改进详细方案",
        author="Codex",
        subject="LangG CRC Agent development and validation strategy",
    )
    doc.build(build_story(), onFirstPage=on_first_page, onLaterPages=on_later_pages)


if __name__ == "__main__":
    build_pdf()
    print(OUTPUT)
