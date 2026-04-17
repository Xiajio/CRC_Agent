"""
Clinical Nodes - 主模块（重构版）

本文件保持对外的统一接口，所有实现细节已拆分到子模块中。
"""

# 从子模块导入所有节点和函数
from .node_utils import *
from .intent_nodes import *
from .general_nodes import *
from .knowledge_nodes import *
from .assessment_nodes import *
from .staging_nodes import *
from .decision_nodes import *
from .router import route_after_intent

# 保持向后兼容：导出所有公共接口
# 这样外部代码（如 graph_builder.py）仍然可以像以前一样使用这些函数

__all__ = [
    # ========== 工具类和辅助函数 ==========
    'ThinkingColors',
    'ThinkingResult',
    '_clean_json_string',
    '_clean_and_validate_json',
    '_extract_and_update_references',
    '_calculate_text_similarity',
    '_parse_thinking_tags',
    '_extract_thinking_from_chunk',
    '_invoke_with_streaming',
    '_ensure_message',
    '_execute_tool_calls',
    '_execute_tool_calls_robust',
    '_build_fallback_search_query',
    '_user_text',
    '_latest_user_text',
    '_calculate_improvement',
    '_is_repeated_rejection',
    '_generate_fallback_plan',
    '_is_postop_context',
    '_extract_ct_text',
    '_extract_pathology_text',
    '_extract_mri_text',
    '_needs_full_decision',
    '_select_tools',
    
    # ========== 意图分类和路由 ==========
    'node_intent_classifier',
    'route_by_intent',
    'route_after_intent',
    'node_general_chat',
    '_rule_based_intent',
    
    # ========== 知识检索 ==========
    'node_knowledge_retrieval',
    'node_web_search_agent',
    '_extract_search_query',
    '_select_search_tool',
    
    # ========== 评估和诊断 ==========
    'node_assessment',
    'node_diagnosis',
    'node_staging_router',
    
    # ========== 分期 ==========
    'node_colon_staging',
    'node_rectal_staging',
    
    # ========== 决策和审核 ==========
    'node_decision',
    'node_critic',
    'route_by_critic_v2',
    'node_finalize',
    '_detect_feedback_type',
    '_format_final_response',
    '_calculate_text_similarity',  # 决策模块中的版本
]

# 版本信息
__version__ = '2.0.0'

# 模块文档
"""
本模块已被重构为多个子模块，以提高可维护性。

## 模块结构：

1. **node_utils.py** - 共享工具函数和辅助类
   - ThinkingColors: 终端输出颜色
   - ThinkingResult: 思考结果容器
   - JSON 处理函数
   - 引用提取函数
   - 文本相似度计算
   - 流式调用和工具执行
   - 用户文本提取
   - 数据提取（CT、MRI、病理）

2. **intent_nodes.py** - 意图分类和路由
   - node_intent_classifier: 意图分类节点
   - route_by_intent: 基于意图的路由函数
   - node_general_chat: 闲聊处理节点
   - _rule_based_intent: 规则意图检测

3. **knowledge_nodes.py** - 知识检索
   - node_knowledge_retrieval: 本地知识检索节点
   - node_web_search_agent: 联网搜索代理
   - _extract_search_query: 搜索查询提取
   - _select_search_tool: 搜索工具选择

4. **assessment_nodes.py** - 评估和诊断
   - node_assessment: 风险评估节点
   - node_diagnosis: 诊断节点
   - node_staging_router: 分期路由函数

5. **staging_nodes.py** - 分期
   - node_colon_staging: 结肠癌分期节点
   - node_rectal_staging: 直肠癌分期节点

6. **decision_nodes.py** - 决策和审核
   - node_decision: 决策节点
   - node_critic: 审核节点
   - route_by_critic_v2: 路由函数
   - node_finalize: 最终化节点
   - _detect_feedback_type: 反馈类型检测
   - _format_final_response: 响应格式化

## 向后兼容性：

所有从外部使用此模块的代码仍然可以正常工作，无需修改导入语句。
例如：
  - from src.nodes.clinical_nodes import node_assessment
  - from src.nodes.clinical_nodes import node_decision
  - from src.nodes.clinical_nodes import route_by_critic_v2

## 使用示例：

```python
from src.nodes import clinical_nodes

# 使用意图分类
intent_result = node_intent_classifier(model, streaming=True)

# 使用决策节点
decision_result = node_decision(model, tools, streaming=True)

# 使用审核节点
critic_result = node_critic(model, streaming=True)

# 使用路由
route = route_by_critic_v2(state)
```
"""
