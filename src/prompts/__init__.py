"""
Prompt 管理模块

本模块集中管理所有 LLM Prompts，方便维护和调整。
非开发人员（如医生或产品经理）可以直接修改此模块中的 Prompt 措辞。

模块结构：
- assessment_prompts.py: 评估和诊断相关 Prompts
- intent_prompts.py: 意图分类相关 Prompts
- knowledge_prompts.py: 知识检索相关 Prompts
- decision_prompts.py: 决策和审核相关 Prompts
- database_prompts.py: 数据库查询相关 Prompts
- planner_prompts.py: 规划节点相关 Prompts
"""

from .assessment_prompts import (
    FAST_FORMAT_PROMPT,
    CASE_INTEGRITY_SYSTEM_PROMPT,
    ASSESSMENT_SYSTEM_PROMPT,
    DIAGNOSIS_SYSTEM_PROMPT,
    RADIOLOGY_REPORT_INTERPRETATION_PROMPT,
)

from .intent_prompts import (
    INTENT_CLASSIFIER_SYSTEM_PROMPT,
)

from .knowledge_prompts import (
    SEARCH_PLANNER_SYSTEM_PROMPT,
    SUFFICIENCY_EVALUATOR_SYSTEM_PROMPT,
    KNOWLEDGE_SYNTHESIS_SYSTEM_PROMPT,
    GENERAL_KNOWLEDGE_SYNTHESIS_SYSTEM_PROMPT,
)

from .general_prompts import (
    GENERAL_CHAT_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    INFO_ONLY_SYSTEM_PROMPT,
)

from .decision_prompts import (
    DECISION_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    QUERY_GENERATION_SYSTEM_PROMPT,
)

from .database_prompts import (
    DATABASE_QUERY_SYSTEM_PROMPT,
)

from .planner_prompts import (
    PLANNER_SYSTEM_PROMPT,
    SELF_CORRECTION_PROMPT_TEMPLATE,
    PLANNING_USER_PROMPT_TEMPLATE,
    MULTI_TASK_USER_PROMPT_TEMPLATE,
)

from .evaluation_prompts import (
    CITATION_CHECKER_SYSTEM_PROMPT,
    LLM_JUDGE_SYSTEM_PROMPT,
)

__all__ = [
    # Assessment Prompts
    'FAST_FORMAT_PROMPT',
    'CASE_INTEGRITY_SYSTEM_PROMPT',
    'ASSESSMENT_SYSTEM_PROMPT',
    'DIAGNOSIS_SYSTEM_PROMPT',
    # Intent Prompts
    'INTENT_CLASSIFIER_SYSTEM_PROMPT',
    'GENERAL_CHAT_SYSTEM_PROMPT',
    # Knowledge Prompts
    'SEARCH_PLANNER_SYSTEM_PROMPT',
    'SUFFICIENCY_EVALUATOR_SYSTEM_PROMPT',
    'KNOWLEDGE_SYNTHESIS_SYSTEM_PROMPT',
    'GENERAL_KNOWLEDGE_SYNTHESIS_SYSTEM_PROMPT',
    # Decision Prompts
    'DECISION_SYSTEM_PROMPT',
    'CRITIC_SYSTEM_PROMPT',
    'QUERY_GENERATION_SYSTEM_PROMPT',
    # Database Prompts
    'DATABASE_QUERY_SYSTEM_PROMPT',
    # Planner Prompts
    'PLANNER_SYSTEM_PROMPT',
    'SELF_CORRECTION_PROMPT_TEMPLATE',
    'PLANNING_USER_PROMPT_TEMPLATE',
    'MULTI_TASK_USER_PROMPT_TEMPLATE',
    # Evaluation Prompts
    'CITATION_CHECKER_SYSTEM_PROMPT',
    'LLM_JUDGE_SYSTEM_PROMPT',
]

__version__ = '1.0.0'
