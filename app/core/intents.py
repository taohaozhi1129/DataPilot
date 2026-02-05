from typing import List, Dict
from pydantic import BaseModel

class IntentDefinition(BaseModel):
    name: str
    description: str
    examples: List[str]

# -----------------------------------------------------------------------------
# 意图注册表 (Intent Registry)
# 在这里配置系统支持的所有意图，Router 会自动加载
# -----------------------------------------------------------------------------
INTENT_REGISTRY = [
    IntentDefinition(
        name="SQL_QUERY",
        description="用户想要从数据库中查询数据、进行统计分析、获取报表或查看具体的数据记录。",
        examples=[
            "查询上个月的销售额",
            "统计一下北京地区的用户数量",
            "帮我看看最近的订单情况",
            "分析一下2024年的增长趋势",
            "有多少个活跃用户？"
        ]
    ),
    IntentDefinition(
        name="TASK_CREATE",
        description="用户想要构建数据集成任务、进行数据同步、数据迁移、ETL 操作或创建新的数据作业。",
        examples=[
            "创建一个从 MySQL 到 Hive 的同步任务",
            "把 users 表的数据迁移到 clickhouse",
            "帮我配置一个每日抽取的任务",
            "新建数据同步作业",
            "从生产库导入数据到测试库"
        ]
    ),
    IntentDefinition(
        name="CHAT",
        description="用户进行的闲聊、问候、自我介绍，或者不涉及具体数据操作的一般性咨询。",
        examples=[
            "你好",
            "你是谁？",
            "你会做什么？",
            "今天的日期是多少",
            "谢谢你"
        ]
    )
]

def get_intent_options() -> str:
    """生成意图选项的格式化字符串，用于 Prompt"""
    options = []
    for intent in INTENT_REGISTRY:
        options.append(f"- {intent.name}: {intent.description}")
    return "\n".join(options)

def get_few_shot_examples() -> str:
    """生成 Few-Shot 示例，用于增强识别能力"""
    examples = []
    for intent in INTENT_REGISTRY:
        for ex in intent.examples[:3]: # 每个意图取前3个例子
            examples.append(f"User: {ex}\nIntent: {intent.name}")
    return "\n".join(examples)
