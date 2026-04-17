from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import registerFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "output" / "pdf" / "e2e-acceptance-exec-summary-2026-04-13.pdf"


def register_cn_font() -> str:
    candidates = [
        ("MicrosoftYaHei", Path(r"C:\Windows\Fonts\msyh.ttc"), 0),
        ("SimSun", Path(r"C:\Windows\Fonts\simsun.ttc"), 0),
    ]
    for name, path, subfont_index in candidates:
        if not path.exists():
            continue
        registerFont(TTFont(name, str(path), subfontIndex=subfont_index))
        return name
    raise FileNotFoundError("No supported Chinese font file found.")


def build_styles(font_name: str):
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ExecTitle",
            parent=styles["Title"],
            fontName=font_name,
            fontSize=23,
            leading=29,
            textColor=colors.HexColor("#17324D"),
            alignment=TA_LEFT,
            spaceAfter=4,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="ExecSubtitle",
            parent=styles["Normal"],
            fontName=font_name,
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#5E6B78"),
            spaceAfter=10,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontName=font_name,
            fontSize=12.5,
            leading=16,
            textColor=colors.HexColor("#17324D"),
            spaceAfter=4,
            spaceBefore=0,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyCN",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#243746"),
            spaceAfter=3,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="MetricLabel",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=8.5,
            leading=10,
            textColor=colors.HexColor("#5E6B78"),
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="MetricValue",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#17324D"),
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="CalloutTitle",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=11.5,
            leading=15,
            textColor=colors.white,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="CalloutBody",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=9.5,
            leading=13,
            textColor=colors.white,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="FooterCN",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#6E7C8A"),
            wordWrap="CJK",
        )
    )
    return styles


def metric_card(label: str, value: str, styles) -> Table:
    table = Table(
        [
            [Paragraph(label, styles["MetricLabel"])],
            [Paragraph(value, styles["MetricValue"])],
        ],
        colWidths=[42 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F7FA")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D7E0E8")),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def section_table(title: str, bullets: list[str], styles) -> Table:
    rows = [[Paragraph(title, styles["SectionTitle"])]]
    rows.extend([[Paragraph(f"{index}. {text}", styles["BodyCN"])] for index, text in enumerate(bullets, start=1)])

    table = Table(rows, colWidths=[84 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D7E0E8")),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def build_document() -> None:
    font_name = register_cn_font()
    styles = build_styles(font_name)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=13 * mm,
        bottomMargin=12 * mm,
        title="智能体 E2E 全量验收简版汇报",
        author="Codex",
    )

    story = [
        Paragraph("智能体 E2E 全量验收简版汇报", styles["ExecTitle"]),
        Paragraph("面向领导/客户 | 2026-04-13 | 项目：LangG_New", styles["ExecSubtitle"]),
    ]

    callout = Table(
        [
            [
                Paragraph("自动化验收结论：PASS", styles["CalloutTitle"]),
                Paragraph("发布建议：PASS WITH CONDITIONS", styles["CalloutTitle"]),
            ],
            [
                Paragraph(
                    "后端 acceptance-support 34 passed，前端 Playwright E2E 14 passed，当前无 skipped、无 conditional-pass 例外项。",
                    styles["CalloutBody"],
                ),
                Paragraph(
                    "上线前仍建议补齐人工签字项，覆盖医疗文案、视觉呈现、引用可信度与 Trust & Safety 结果复核。",
                    styles["CalloutBody"],
                ),
            ],
        ],
        colWidths=[84 * mm, 84 * mm],
    )
    callout.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1D5C4E")),
                ("LEFTPADDING", (0, 0), (-1, -1), 11),
                ("RIGHTPADDING", (0, 0), (-1, -1), 11),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#2D7A68")),
            ]
        )
    )
    story.extend([callout, Spacer(1, 6 * mm)])

    metrics = Table(
        [
            [
                metric_card("后端自动化", "34 passed", styles),
                metric_card("前端 E2E", "14 passed", styles),
                metric_card("计划任务", "10 / 10 完成", styles),
                metric_card("阻断缺陷", "0", styles),
            ]
        ],
        colWidths=[42 * mm, 42 * mm, 42 * mm, 42 * mm],
    )
    metrics.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.extend([metrics, Spacer(1, 6 * mm)])

    left_col = [
        section_table(
            "本次目标",
            [
                "在前后端真实联调条件下，用受控 fixture 与测试库完成一轮可复现的全量验收。",
                "覆盖工作区主对话、数据库工作台、文件上传、卡片交互、会话恢复与消息持久化一致性。",
                "形成可重复执行的脚本、证据目录、运行手册和发布报告模板，支持后续版本复用。",
            ],
            styles,
        ),
        Spacer(1, 4 * mm),
        section_table(
            "关键完善",
            [
                "完成 Task 1 到 Task 10，打通 fixture 透传、受控数据库、上传卡片、Playwright acceptance harness 与全量 runner。",
                "将不稳定的 graph fixtures 收敛为可信的混合模式：数据库场景保留 live capture，其余高风险场景改为 deterministic template。",
                "解锁并跑通工作区、数据库、上传三组 E2E 套件，补齐此前被 skip 或 fixme 的主链路场景。",
            ],
            styles,
        ),
    ]

    right_col = [
        section_table(
            "验收结果",
            [
                "后端验收支持用例 34 passed；Playwright 端到端用例 14 passed；核心链路全部通过。",
                "工作区、数据库、上传与卡片动作均已形成证据，统一归档到 output/acceptance。",
                "当前自动化结果可作为版本放行依据，说明系统在受控环境下已具备稳定复测能力。",
            ],
            styles,
        ),
        Spacer(1, 4 * mm),
        section_table(
            "剩余事项",
            [
                "仍需按人工清单完成医疗合理性、视觉体验、引用可信度和 Trust & Safety 的签字确认。",
                "存在一项非阻断技术尾项：document_converter.py 仍有 Pydantic deprecation warning。",
                "当前工作区不是 git repo，因此本次未形成 commit 或 branch checkpoint，但不影响验收结论与证据完整性。",
            ],
            styles,
        ),
    ]

    summary_grid = Table(
        [[left_col[0], right_col[0]], [left_col[2], right_col[2]]],
        colWidths=[84 * mm, 84 * mm],
        rowHeights=[60 * mm, 60 * mm],
    )
    summary_grid.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.extend([summary_grid, Spacer(1, 4 * mm)])

    story.append(
        Paragraph(
            "证据目录：output/acceptance。详细版技术报告：e2e-release-report-2026-04-13.md。",
            styles["FooterCN"],
        )
    )

    doc.build(story)


if __name__ == "__main__":
    build_document()
