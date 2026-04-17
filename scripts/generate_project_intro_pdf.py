from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 42

PALETTE = {
    "navy": colors.HexColor("#0F3D56"),
    "teal": colors.HexColor("#1D7A85"),
    "mint": colors.HexColor("#2FB6A2"),
    "sky": colors.HexColor("#DFF4F1"),
    "ice": colors.HexColor("#F5FAFB"),
    "sand": colors.HexColor("#F6C67A"),
    "text": colors.HexColor("#183B4A"),
    "muted": colors.HexColor("#5A7380"),
    "line": colors.HexColor("#C7DDE3"),
    "white": colors.white,
}


def _register_fonts() -> tuple[str, str]:
    regular_candidates = [
        ("MicrosoftYaHei", Path("C:/Windows/Fonts/msyh.ttc"), {"subfontIndex": 0}),
        ("SimHei", Path("C:/Windows/Fonts/simhei.ttf"), {}),
        ("SimSun", Path("C:/Windows/Fonts/simsun.ttc"), {"subfontIndex": 0}),
    ]
    bold_candidates = [
        ("MicrosoftYaHeiBold", Path("C:/Windows/Fonts/msyhbd.ttc"), {"subfontIndex": 0}),
        ("SimHei", Path("C:/Windows/Fonts/simhei.ttf"), {}),
    ]

    regular_name = "STSong-Light"
    bold_name = "STSong-Light"

    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    except KeyError:
        pass

    for font_name, font_path, extra in regular_candidates:
        if font_path.exists():
            pdfmetrics.registerFont(TTFont(font_name, str(font_path), **extra))
            regular_name = font_name
            break

    for font_name, font_path, extra in bold_candidates:
        if font_path.exists():
            pdfmetrics.registerFont(TTFont(font_name, str(font_path), **extra))
            bold_name = font_name
            break

    return regular_name, bold_name


REGULAR_FONT, BOLD_FONT = _register_fonts()

STYLES = {
    "hero_title": ParagraphStyle(
        "hero_title",
        fontName=BOLD_FONT,
        fontSize=27,
        leading=36,
        textColor=PALETTE["white"],
        alignment=TA_LEFT,
        wordWrap="CJK",
    ),
    "hero_subtitle": ParagraphStyle(
        "hero_subtitle",
        fontName=REGULAR_FONT,
        fontSize=11.5,
        leading=18,
        textColor=colors.HexColor("#E7F6F5"),
        alignment=TA_LEFT,
        wordWrap="CJK",
    ),
    "chapter_title": ParagraphStyle(
        "chapter_title",
        fontName=BOLD_FONT,
        fontSize=21,
        leading=26,
        textColor=PALETTE["navy"],
        alignment=TA_LEFT,
        wordWrap="CJK",
    ),
    "section_text": ParagraphStyle(
        "section_text",
        fontName=REGULAR_FONT,
        fontSize=10.2,
        leading=16,
        textColor=PALETTE["text"],
        alignment=TA_LEFT,
        wordWrap="CJK",
    ),
    "small_text": ParagraphStyle(
        "small_text",
        fontName=REGULAR_FONT,
        fontSize=9,
        leading=14,
        textColor=PALETTE["muted"],
        alignment=TA_LEFT,
        wordWrap="CJK",
    ),
    "card_title": ParagraphStyle(
        "card_title",
        fontName=BOLD_FONT,
        fontSize=12,
        leading=18,
        textColor=PALETTE["navy"],
        alignment=TA_LEFT,
        wordWrap="CJK",
    ),
    "card_body": ParagraphStyle(
        "card_body",
        fontName=REGULAR_FONT,
        fontSize=9.5,
        leading=14,
        textColor=PALETTE["text"],
        alignment=TA_LEFT,
        wordWrap="CJK",
    ),
    "center_stat": ParagraphStyle(
        "center_stat",
        fontName=BOLD_FONT,
        fontSize=13,
        leading=18,
        textColor=PALETTE["navy"],
        alignment=TA_CENTER,
        wordWrap="CJK",
    ),
}


def draw_paragraph(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y_top: float,
    width: float,
    style: ParagraphStyle,
) -> float:
    paragraph = Paragraph(text, style)
    _, height = paragraph.wrap(width, PAGE_HEIGHT)
    paragraph.drawOn(pdf, x, y_top - height)
    return height


def draw_round_card(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    accent: colors.Color,
    badge: str | None = None,
) -> None:
    pdf.setFillColor(PALETTE["white"])
    pdf.setStrokeColor(PALETTE["line"])
    pdf.setLineWidth(1)
    pdf.roundRect(x, y, width, height, 14, fill=1, stroke=1)

    pdf.setFillColor(accent)
    pdf.roundRect(x + 14, y + height - 18, 70, 8, 4, fill=1, stroke=0)

    if badge:
        pdf.setFillColor(PALETTE["sky"])
        pdf.roundRect(x + width - 90, y + height - 30, 72, 18, 9, fill=1, stroke=0)
        pdf.setFillColor(PALETTE["teal"])
        pdf.setFont("Helvetica-Bold", 8.5)
        pdf.drawCentredString(x + width - 54, y + height - 24, badge)

    draw_paragraph(pdf, title, x + 16, y + height - 28, width - 32, STYLES["card_title"])
    draw_paragraph(pdf, body, x + 16, y + height - 58, width - 32, STYLES["card_body"])


def draw_chip(pdf: canvas.Canvas, x: float, y: float, text: str, fill: colors.Color) -> None:
    width = 22 + len(text) * 7.2
    pdf.setFillColor(fill)
    pdf.roundRect(x, y, width, 20, 10, fill=1, stroke=0)
    pdf.setFillColor(PALETTE["white"])
    pdf.setFont("Helvetica-Bold", 8)
    pdf.drawString(x + 10, y + 6.5, text)


def draw_page_frame(pdf: canvas.Canvas, page_no: int, cn_title: str, en_title: str) -> None:
    pdf.setFillColor(PALETTE["ice"])
    pdf.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)

    pdf.setFillColor(PALETTE["navy"])
    pdf.roundRect(MARGIN, PAGE_HEIGHT - 58, PAGE_WIDTH - 2 * MARGIN, 24, 12, fill=1, stroke=0)
    pdf.setFillColor(PALETTE["white"])
    pdf.setFont(BOLD_FONT, 11.5)
    pdf.drawString(MARGIN + 16, PAGE_HEIGHT - 51, cn_title)
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawRightString(PAGE_WIDTH - MARGIN - 16, PAGE_HEIGHT - 50.5, en_title)

    pdf.setStrokeColor(PALETTE["line"])
    pdf.setLineWidth(0.8)
    pdf.line(MARGIN, 28, PAGE_WIDTH - MARGIN, 28)
    pdf.setFillColor(PALETTE["muted"])
    pdf.setFont("Helvetica", 8.5)
    pdf.drawString(MARGIN, 16, "Intelligent Agent Product Introduction")
    pdf.drawRightString(PAGE_WIDTH - MARGIN, 16, f"{page_no:02d}")


def draw_cover(pdf: canvas.Canvas) -> None:
    pdf.setFillColor(PALETTE["navy"])
    pdf.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)

    pdf.setFillColor(PALETTE["teal"])
    pdf.circle(PAGE_WIDTH - 72, PAGE_HEIGHT - 78, 118, fill=1, stroke=0)
    pdf.setFillColor(PALETTE["mint"])
    pdf.circle(PAGE_WIDTH - 8, PAGE_HEIGHT - 156, 76, fill=1, stroke=0)
    pdf.setFillColor(colors.HexColor("#114C6D"))
    pdf.roundRect(0, 0, PAGE_WIDTH, 138, 0, fill=1, stroke=0)

    pdf.setStrokeColor(colors.HexColor("#65CFC4"))
    pdf.setLineWidth(3)
    pdf.line(MARGIN, PAGE_HEIGHT - 145, MARGIN + 130, PAGE_HEIGHT - 145)

    draw_paragraph(
        pdf,
        "智能体产品介绍",
        MARGIN,
        PAGE_HEIGHT - 182,
        240,
        STYLES["hero_title"],
    )
    draw_paragraph(
        pdf,
        "面向医疗服务全流程的智能协同平台，覆盖就诊前采集、就诊中交互、就诊后工具支持与管理协同，为机构提供可扩展、可治理、可落地的 AI 能力底座。",
        MARGIN,
        PAGE_HEIGHT - 246,
        280,
        STYLES["hero_subtitle"],
    )

    draw_chip(pdf, MARGIN, PAGE_HEIGHT - 318, "Pre-Visit", PALETTE["teal"])
    draw_chip(pdf, MARGIN + 90, PAGE_HEIGHT - 318, "In-Visit", PALETTE["mint"])
    draw_chip(pdf, MARGIN + 174, PAGE_HEIGHT - 318, "Post-Visit", PALETTE["sand"])
    draw_chip(pdf, MARGIN + 266, PAGE_HEIGHT - 318, "Governance", colors.HexColor("#4B8CA8"))

    quote_y = 182
    pdf.setFillColor(colors.HexColor("#123A50"))
    pdf.roundRect(MARGIN, quote_y, PAGE_WIDTH - 2 * MARGIN, 114, 18, fill=1, stroke=0)
    draw_paragraph(
        pdf,
        "核心定位：将多角色医疗服务流程沉淀为“可感知、可推理、可执行、可追溯”的智能体工作流，帮助医院、专科中心与互联网医疗平台提升服务效率与患者体验。",
        MARGIN + 20,
        quote_y + 86,
        PAGE_WIDTH - 2 * MARGIN - 40,
        ParagraphStyle(
            "cover_box",
            parent=STYLES["hero_subtitle"],
            textColor=colors.HexColor("#D9F1EE"),
            fontSize=12.5,
            leading=20,
        ),
    )

    pdf.setFillColor(PALETTE["white"])
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(MARGIN, 78, "Intelligent Agent Product Introduction")
    pdf.setFont(REGULAR_FONT, 10.5)
    pdf.drawString(MARGIN, 58, "适用对象：医疗机构管理者、信息化负责人、业务运营负责人、临床协同团队")


def draw_contents(pdf: canvas.Canvas) -> None:
    draw_page_frame(pdf, 2, "章节导航", "Contents")

    draw_paragraph(pdf, "项目内容围绕产品价值、架构、能力模块和合规落地展开。", MARGIN, PAGE_HEIGHT - 92, 250, STYLES["section_text"])

    sections = [
        ("01", "产品概述", "Product Overview"),
        ("02", "产品架构", "Product Architecture"),
        ("03", "核心功能模块", "Core Modules"),
        ("04", "扩展功能", "Future Roadmap"),
        ("05", "典型应用场景", "Application Scenarios"),
        ("06", "技术亮点", "Technology Highlights"),
        ("07", "数据安全与合规", "Data Security"),
        ("08", "总结与展望", "Summary & Outlook"),
    ]

    start_y = PAGE_HEIGHT - 168
    for index, (num, cn, en) in enumerate(sections):
        row_y = start_y - index * 58
        pdf.setFillColor(PALETTE["white"])
        pdf.setStrokeColor(PALETTE["line"])
        pdf.roundRect(MARGIN, row_y, PAGE_WIDTH - 2 * MARGIN, 42, 12, fill=1, stroke=1)
        pdf.setFillColor(PALETTE["sky"] if index % 2 == 0 else colors.HexColor("#EAF7F3"))
        pdf.roundRect(MARGIN + 12, row_y + 8, 48, 26, 13, fill=1, stroke=0)
        pdf.setFillColor(PALETTE["teal"])
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawCentredString(MARGIN + 36, row_y + 17, num)
        pdf.setFillColor(PALETTE["navy"])
        pdf.setFont(BOLD_FONT, 11.5)
        pdf.drawString(MARGIN + 74, row_y + 17, cn)
        pdf.setFont("Helvetica", 9)
        pdf.setFillColor(PALETTE["muted"])
        pdf.drawString(PAGE_WIDTH - MARGIN - 170, row_y + 17, en)

    draw_round_card(
        pdf,
        PAGE_WIDTH - MARGIN - 210,
        72,
        210,
        120,
        "建议阅读路径",
        "先看产品概述与架构，建立整体认知；再聚焦四层模块能力与应用场景；最后结合技术亮点和合规能力判断落地可行性。",
        PALETTE["mint"],
        badge="Recommended",
    )


def draw_overview(pdf: canvas.Canvas) -> None:
    draw_page_frame(pdf, 3, "一、产品概述", "Product Overview")

    draw_paragraph(
        pdf,
        "本产品是一套面向医疗服务场景的智能体平台，以“患者旅程”为主线，将咨询、问诊、随访、运营与管理支持连接为一个统一闭环。",
        MARGIN,
        PAGE_HEIGHT - 100,
        310,
        STYLES["chapter_title"],
    )

    draw_paragraph(
        pdf,
        "产品不替代医生决策，而是作为医生、患者、护士、运营与管理人员之间的智能协同层，负责信息组织、知识召回、任务编排和服务承接。",
        MARGIN,
        PAGE_HEIGHT - 168,
        305,
        STYLES["section_text"],
    )

    stats = [
        ("服务对象", "患者、医生、护士、运营、管理者"),
        ("价值主张", "更快获取信息、更稳执行流程、更好沉淀数据"),
        ("部署方式", "可嵌入院内系统、互联网医院或专病服务平台"),
    ]

    stat_y = PAGE_HEIGHT - 292
    for idx, (title, body) in enumerate(stats):
        draw_round_card(
            pdf,
            MARGIN + idx * 170,
            stat_y,
            154,
            112,
            title,
            body,
            PALETTE["mint"] if idx != 1 else PALETTE["sand"],
        )

    draw_round_card(
        pdf,
        MARGIN,
        82,
        250,
        132,
        "产品目标",
        "围绕患者服务、临床协同和机构运营三条主线，打造一个可持续迭代的 AI 助手体系，让服务响应更及时、流程执行更标准、数据沉淀更完整。",
        PALETTE["teal"],
    )
    draw_round_card(
        pdf,
        MARGIN + 268,
        82,
        245,
        132,
        "适用价值",
        "适合专病管理、互联网问诊、院内导诊、术后康复、慢病随访等高频业务场景，帮助机构形成可复制的智能服务能力。",
        PALETTE["sand"],
    )


def draw_architecture(pdf: canvas.Canvas) -> None:
    draw_page_frame(pdf, 4, "二、产品架构", "Product Architecture")

    draw_paragraph(
        pdf,
        "架构按照“入口感知 - 智能交互 - 工具执行 - 管理协同 - 数据底座”分层设计，保证前台体验和后台治理同步建设。",
        MARGIN,
        PAGE_HEIGHT - 96,
        PAGE_WIDTH - 2 * MARGIN,
        STYLES["section_text"],
    )

    layers = [
        ("用户入口层", "患者小程序 / App / 导诊台 / 医生工作台 / 管理后台", PALETTE["sand"], 610),
        ("患者信息采集层", "预约问卷、病史补录、症状分诊、资料上传、意图识别", colors.HexColor("#CFEDEA"), 518),
        ("智能诊疗交互层", "问答交互、知识召回、辅助解释、对话总结、就诊建议提示", colors.HexColor("#D8EEF8"), 426),
        ("工具执行层", "随访任务、提醒通知、报告生成、健康教育、工单分发", colors.HexColor("#E3F5F0"), 334),
        ("服务协同管理层", "角色权限、流程编排、质控审计、运营监控、配置中心", colors.HexColor("#E7EEF7"), 242),
        ("数据与模型底座", "结构化数据、向量知识库、规则引擎、模型路由、日志与追踪", PALETTE["navy"], 146),
    ]

    x = MARGIN + 18
    width = PAGE_WIDTH - 2 * MARGIN - 36
    for title, body, fill, y in layers:
        pdf.setFillColor(fill)
        pdf.setStrokeColor(PALETTE["line"] if fill != PALETTE["navy"] else PALETTE["navy"])
        pdf.roundRect(x, y, width, 64, 16, fill=1, stroke=1)
        pdf.setFillColor(PALETTE["white"] if fill == PALETTE["navy"] else PALETTE["navy"])
        pdf.setFont(BOLD_FONT, 12)
        pdf.drawString(x + 18, y + 42, title)
        pdf.setFont(REGULAR_FONT, 9.6)
        pdf.drawString(x + 18, y + 22, body)
        if y > 146:
            pdf.setStrokeColor(PALETTE["teal"])
            pdf.setLineWidth(2)
            pdf.line(PAGE_WIDTH / 2, y - 6, PAGE_WIDTH / 2, y - 20)
            pdf.line(PAGE_WIDTH / 2, y - 20, PAGE_WIDTH / 2 - 6, y - 14)
            pdf.line(PAGE_WIDTH / 2, y - 20, PAGE_WIDTH / 2 + 6, y - 14)

    pdf.setFont("Helvetica-Bold", 8.5)
    pdf.setFillColor(PALETTE["teal"])
    pdf.drawString(MARGIN + 20, 96, "Product Architecture")


def draw_core_modules_upper(pdf: canvas.Canvas) -> None:
    draw_page_frame(pdf, 5, "三、核心功能模块", "Core Modules")
    draw_paragraph(
        pdf,
        "核心能力覆盖患者触达、问诊协同、工具支撑和组织管理四个层面，其中前两组直接影响就诊体验。",
        MARGIN,
        PAGE_HEIGHT - 96,
        PAGE_WIDTH - 2 * MARGIN,
        STYLES["section_text"],
    )

    draw_round_card(
        pdf,
        MARGIN,
        328,
        246,
        362,
        "第一组：患者信息采集层（就诊前）",
        "1. 智能预问诊：在患者到院前完成主诉、现病史、既往史和生活习惯信息补录。<br/><br/>2. 多模态资料接入：支持图片、检查报告、病历摘要上传并自动归档。<br/><br/>3. 风险预分层：基于规则与模型识别急重症风险、缺失材料和优先服务需求。<br/><br/>4. 预约协同：与挂号、检查预约、复诊登记联动，减少人工沟通成本。",
        PALETTE["mint"],
        badge="Pre-Visit",
    )
    draw_round_card(
        pdf,
        MARGIN + 266,
        328,
        246,
        362,
        "第二组：智能诊疗交互层（就诊中）",
        "1. 就诊对话助手：为患者提供清晰问答，为医生提供信息摘要与提醒。<br/><br/>2. 专病知识召回：结合院内规范、科室知识库和常见路径进行内容检索与解释。<br/><br/>3. 结构化总结：自动生成关键病情要点、用药提醒、检查建议草稿。<br/><br/>4. 协同触发：根据问诊结果触发复诊提醒、检查建议或后续服务。",
        PALETTE["sand"],
        badge="In-Visit",
    )


def draw_core_modules_lower(pdf: canvas.Canvas) -> None:
    draw_page_frame(pdf, 6, "三、核心功能模块", "Core Modules")
    draw_paragraph(
        pdf,
        "后两组能力面向就诊后和日常运营，重点解决长期服务连续性与规模化管理问题。",
        MARGIN,
        PAGE_HEIGHT - 96,
        PAGE_WIDTH - 2 * MARGIN,
        STYLES["section_text"],
    )

    draw_round_card(
        pdf,
        MARGIN,
        328,
        246,
        362,
        "第三组：工具层（就诊后 / 日常）",
        "1. 随访计划：针对术后、慢病、复诊患者生成周期性任务和提醒。<br/><br/>2. 健康教育：基于病种、治疗方案和阶段推送个性化宣教内容。<br/><br/>3. 服务工单：连接客服、护士站、药学咨询、康复指导等处理流程。<br/><br/>4. 文档输出：自动生成沟通记录、出院指导、随访纪要等标准化材料。",
        PALETTE["teal"],
        badge="Post-Visit",
    )
    draw_round_card(
        pdf,
        MARGIN + 266,
        328,
        246,
        362,
        "第四组：服务协同管理层（管理支撑）",
        "1. 流程配置：按科室、病种和机构策略配置服务模板。<br/><br/>2. 角色协同：支持医生、护士、客服、运营和管理员分工处理。<br/><br/>3. 质量监控：追踪响应时效、服务完成率、满意度和异常工单。<br/><br/>4. 运营分析：沉淀咨询量、转化率、活跃度、复诊率等经营指标。",
        colors.HexColor("#7AB7D6"),
        badge="Governance",
    )


def draw_roadmap_and_scenarios(pdf: canvas.Canvas) -> None:
    draw_page_frame(pdf, 7, "四、扩展功能 / 五、典型应用场景", "Future Roadmap & Application Scenarios")

    draw_round_card(
        pdf,
        MARGIN,
        344,
        220,
        344,
        "扩展功能（未来规划）",
        "1. 多智能体协作：让导诊、知识问答、随访和运营助手按角色分工协同。<br/><br/>2. 多模态理解：接入影像摘要、语音转写、检验报告智能解析。<br/><br/>3. 个性化推荐：结合用户画像与疗程阶段给出更精准的服务建议。<br/><br/>4. 生态连接：与 HIS、EMR、CRM、消息中心和支付系统进一步打通。",
        PALETTE["mint"],
        badge="Future Roadmap",
    )

    draw_round_card(
        pdf,
        MARGIN + 238,
        344,
        274,
        344,
        "典型应用场景",
        "1. 互联网医院预问诊与分诊。<br/><br/>2. 专病中心长期随访与康复管理。<br/><br/>3. 住院患者出院指导与复诊承接。<br/><br/>4. 高客单价项目的咨询转化与服务闭环。<br/><br/>5. 院内导诊及辅助服务台的智能化升级。",
        PALETTE["sand"],
        badge="Application Scenarios",
    )

    pdf.setStrokeColor(PALETTE["teal"])
    pdf.setLineWidth(2)
    timeline_y = 194
    pdf.line(MARGIN + 18, timeline_y, PAGE_WIDTH - MARGIN - 18, timeline_y)
    phases = [
        ("P1", "流程智能化", "标准问诊、随访和知识问答率先上线"),
        ("P2", "角色协同化", "支持跨岗位任务流转和运营监控"),
        ("P3", "平台生态化", "沉淀开放接口与多模型策略能力"),
    ]
    for idx, (phase, title, body) in enumerate(phases):
        cx = MARGIN + 72 + idx * 160
        pdf.setFillColor(PALETTE["teal"] if idx == 0 else PALETTE["mint"] if idx == 1 else PALETTE["sand"])
        pdf.circle(cx, timeline_y, 14, fill=1, stroke=0)
        pdf.setFillColor(PALETTE["white"])
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawCentredString(cx, timeline_y - 3, phase)
        draw_paragraph(pdf, title, cx - 40, 168, 80, STYLES["card_title"])
        draw_paragraph(pdf, body, cx - 58, 144, 116, STYLES["small_text"])


def draw_tech_highlights(pdf: canvas.Canvas) -> None:
    draw_page_frame(pdf, 8, "六、技术亮点", "Technology Highlights")

    draw_paragraph(
        pdf,
        "平台以工作流编排、知识增强和可观测性为关键能力，使智能体从“能回答”走向“能协作、能执行、能治理”。",
        MARGIN,
        PAGE_HEIGHT - 96,
        PAGE_WIDTH - 2 * MARGIN,
        STYLES["section_text"],
    )

    cards = [
        ("模型路由", "根据任务类型在问答、摘要、规则判断等能力间切换，兼顾效果与成本。"),
        ("知识增强", "结合向量检索、关键词召回和结构化规则，提升医疗场景回答稳定性。"),
        ("工作流编排", "将问诊、提醒、转派、记录等动作串联为可配置流程。"),
        ("可观测性", "记录提示词、模型响应、工具调用和人工介入痕迹，便于复盘。"),
        ("低耦合集成", "支持 API、消息、Webhook、表单等方式嵌入现有系统。"),
        ("持续优化", "通过反馈闭环持续更新知识库、规则策略和服务模板。"),
    ]

    for idx, (title, body) in enumerate(cards):
        row = idx // 2
        col = idx % 2
        draw_round_card(
            pdf,
            MARGIN + col * 252,
            472 - row * 142,
            230,
            118,
            title,
            body,
            PALETTE["mint"] if idx % 3 == 0 else PALETTE["sand"] if idx % 3 == 1 else PALETTE["teal"],
        )

    pdf.setFont("Helvetica-Bold", 9)
    pdf.setFillColor(PALETTE["teal"])
    pdf.drawString(MARGIN, 92, "Technology Highlights")


def draw_security(pdf: canvas.Canvas) -> None:
    draw_page_frame(pdf, 9, "七、数据安全与合规", "Data Security")

    draw_paragraph(
        pdf,
        "医疗场景对数据安全、内容审查和权限治理要求更高，平台从架构、流程和运营三方面建立合规防线。",
        MARGIN,
        PAGE_HEIGHT - 96,
        PAGE_WIDTH - 2 * MARGIN,
        STYLES["section_text"],
    )

    left_cards = [
        ("最小权限", "对患者数据、模型配置、服务模板和日志数据实施分级授权。"),
        ("敏感信息保护", "支持脱敏展示、传输加密、日志筛除和水印追踪。"),
        ("审计留痕", "保留关键问答、工具执行、人工接管和配置变更记录。"),
    ]
    for idx, (title, body) in enumerate(left_cards):
        draw_round_card(pdf, MARGIN, 460 - idx * 120, 220, 96, title, body, PALETTE["mint"])

    pdf.setFillColor(PALETTE["white"])
    pdf.setStrokeColor(PALETTE["line"])
    pdf.roundRect(MARGIN + 244, 236, 268, 320, 16, fill=1, stroke=1)
    draw_paragraph(pdf, "合规治理闭环", MARGIN + 262, 530, 120, STYLES["card_title"])

    loop_items = [
        ("接入前", "数据分类、接口评估、权限审批"),
        ("运行中", "内容审查、异常告警、人工兜底"),
        ("复盘后", "质量抽检、日志分析、策略修订"),
    ]
    for idx, (title, body) in enumerate(loop_items):
        y = 432 - idx * 88
        pdf.setFillColor(PALETTE["sky"] if idx % 2 == 0 else colors.HexColor("#EEF7F5"))
        pdf.roundRect(MARGIN + 262, y, 232, 62, 12, fill=1, stroke=0)
        pdf.setFillColor(PALETTE["navy"])
        pdf.setFont(BOLD_FONT, 11)
        pdf.drawString(MARGIN + 278, y + 40, title)
        pdf.setFont(REGULAR_FONT, 9.3)
        pdf.drawString(MARGIN + 278, y + 22, body)

    pdf.setFont("Helvetica-Bold", 9)
    pdf.setFillColor(PALETTE["teal"])
    pdf.drawString(MARGIN, 92, "Data Security")


def draw_summary(pdf: canvas.Canvas) -> None:
    draw_page_frame(pdf, 10, "八、总结与展望", "Summary & Outlook")

    draw_paragraph(
        pdf,
        "智能体产品的价值不在于单点问答，而在于把医疗服务中的“信息、任务、角色和知识”重新组织为一个可执行系统。",
        MARGIN,
        PAGE_HEIGHT - 96,
        PAGE_WIDTH - 2 * MARGIN,
        STYLES["chapter_title"],
    )

    draw_round_card(
        pdf,
        MARGIN,
        332,
        472,
        178,
        "总结",
        "通过四层能力建设，平台既能改善患者体验，也能为机构建立标准流程、数据资产和长期运营能力。它适合从专病、互联网医疗、术后随访等高价值场景切入，再逐步扩展到更广的服务链路。",
        PALETTE["teal"],
    )

    draw_round_card(
        pdf,
        MARGIN,
        140,
        226,
        146,
        "下一步建议",
        "优先选择一个高频、可量化、跨角色协作明显的业务场景试点，以便快速验证服务效率与满意度提升。",
        PALETTE["sand"],
        badge="Next Step",
    )
    draw_round_card(
        pdf,
        MARGIN + 246,
        140,
        226,
        146,
        "落地原则",
        "坚持“小场景切入、流程先行、人工兜底、持续迭代”的实施路线，降低部署风险并加快复制扩展。",
        PALETTE["mint"],
        badge="Principle",
    )

    pdf.setFillColor(PALETTE["navy"])
    pdf.roundRect(MARGIN, 58, PAGE_WIDTH - 2 * MARGIN, 52, 12, fill=1, stroke=0)
    pdf.setFillColor(PALETTE["white"])
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(MARGIN + 16, 89, "Summary & Outlook")
    pdf.setFont(REGULAR_FONT, 10)
    pdf.drawString(MARGIN + 16, 71, "建议将本 PDF 作为对外介绍或内部汇报底稿，再结合机构具体业务数据进行二次定制。")


def build_pdf(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output_path), pagesize=A4)
    pdf.setTitle("智能体产品介绍")
    pdf.setAuthor("OpenAI Codex")
    pdf.setSubject("医疗服务智能体产品介绍")

    draw_cover(pdf)
    pdf.showPage()
    draw_contents(pdf)
    pdf.showPage()
    draw_overview(pdf)
    pdf.showPage()
    draw_architecture(pdf)
    pdf.showPage()
    draw_core_modules_upper(pdf)
    pdf.showPage()
    draw_core_modules_lower(pdf)
    pdf.showPage()
    draw_roadmap_and_scenarios(pdf)
    pdf.showPage()
    draw_tech_highlights(pdf)
    pdf.showPage()
    draw_security(pdf)
    pdf.showPage()
    draw_summary(pdf)
    pdf.save()
    return output_path


def generate_pdf(output_path: str | Path | None = None) -> Path:
    if output_path is None:
        output_path = Path("output/pdf/intelligent-agent-product-introduction.pdf")
    return build_pdf(Path(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the intelligent agent product introduction PDF.")
    parser.add_argument(
        "--output",
        default="output/pdf/intelligent-agent-product-introduction.pdf",
        help="Output PDF path.",
    )
    args = parser.parse_args()
    output_path = generate_pdf(args.output)
    print(output_path)


if __name__ == "__main__":
    main()
