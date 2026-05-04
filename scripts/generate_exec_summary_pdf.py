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
OUTPUT_PATH = ROOT / "output" / "pdf" / "real-case-human-review-exec-summary-2026-05-03.pdf"


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
        title="真实病例人工复核验收简版汇报",
        author="Codex",
    )

    story = [
        Paragraph("真实病例人工复核验收简版汇报", styles["ExecTitle"]),
        Paragraph("面向领导/客户 | 2026-05-03 | 项目：LangG | 仓库：D:\\YiZhu_Agnet\\LangG", styles["ExecSubtitle"]),
    ]

    callout = Table(
        [
            [
                Paragraph("自动化验收结论：PASS", styles["CalloutTitle"]),
                Paragraph("发布建议：PASS WITH HUMAN REVIEW REQUIRED", styles["CalloutTitle"]),
            ],
            [
                Paragraph(
                    "浏览器验收已通过：ok=true，planRows=3，roadmapSteps=4，blockedRoadmapSteps=1，warningCount=5，failedResponses=[]。",
                    styles["CalloutBody"],
                ),
                Paragraph(
                    "没有医学与安全人工签署前，该病例建议不得被解释为无需复核的最终治疗方案。",
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
                metric_card("Plan rows", "3", styles),
                metric_card("Roadmap", "4 steps", styles),
                metric_card("Blocked", "1 step", styles),
                metric_card("Warnings", "5 visible", styles),
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
                "使用真实病例 fixture 验证缺少直接 guideline references 时，系统是否明确进入人工肿瘤专科复核。",
                "覆盖 built frontend、fixture backend、headless browser、执行计划、roadmap 和 clinical event stream 的可见性。",
                "形成可交付的 runbook、人工复核清单、Markdown 报告和 PDF 简版汇报模板。",
            ],
            styles,
        ),
        Spacer(1, 4 * mm),
        section_table(
            "关键完善",
            [
                "将 handoff 从旧 LangG_New 路径与旧一键全量验收口径，修正为当前仓库 D:\\YiZhu_Agnet\\LangG。",
                "把 scripts/run_real_case_browser_acceptance.cjs 纳入交付说明，并统一输出 JSON、截图和后端日志证据。",
                "明确旧 full-pack runner 仅为历史入口；当前 active handoff 是真实病例浏览器验收。",
            ],
            styles,
        ),
    ]

    right_col = [
        section_table(
            "验收检查点",
            [
                "浏览器验收应产生 real-case-human-review-acceptance.json、截图和 backend stdout/stderr 日志。",
                "JSON 需记录 ok=true、fixtureCase=real_case_human_review，以及非零 plan/roadmap/event 计数。",
                "截图需显示 HUMAN_REVIEW_REQUIRED、建议保留、无直接引用披露和 blocked review step。",
            ],
            styles,
        ),
        Spacer(1, 4 * mm),
        section_table(
            "剩余事项",
            [
                "本次使用已有 frontend dist 完成浏览器验收；当前环境重新构建被 esbuild spawn EPERM / Access is denied 阻断。",
                "由医学、产品/测试、安全复核人员完成签署，确认治疗文案和安全披露没有误导。",
                "如需恢复旧 full-pack E2E，先恢复 tests/e2e/acceptance 再运行 run_e2e_full_acceptance.ps1。",
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
            "证据目录：output/browser-acceptance/real_case_human_review。详细报告：real-case-human-review-acceptance-report-2026-05-03.md。",
            styles["FooterCN"],
        )
    )

    doc.build(story)


if __name__ == "__main__":
    build_document()
