import random
import json
from faker import Faker

fake = Faker("zh_CN")

BUSINESS_DOMAINS = [
    "销售域", "用户域", "订单域", "库存域",
    "财务域", "营销域", "供应链域", "物流域"
]

DATA_SOURCES = [
    "ODS销售系统",
    "ODS订单系统",
    "ODS用户系统",
    "CRM系统",
    "ERP系统"
]

METRIC_TYPES = ["金额", "数量", "比例", "时间", "状态", "编码"]

COMMON_COLUMNS = [
    ("id", "主键ID", "bigint"),
    ("created_at", "创建时间", "datetime"),
    ("updated_at", "更新时间", "datetime"),
    ("deleted_flag", "删除标志", "int")
]


def generate_table_metadata(table_index: int):
    domain = random.choice(BUSINESS_DOMAINS)
    data_source = random.choice(DATA_SOURCES)

    table_name = f"{domain[:2]}_table_{table_index}"
    table_comment = f"{domain}核心业务表，记录{domain}相关业务数据"

    return {
        "table_name": table_name,
        "schema_name": "dw",
        "table_comment": table_comment,
        "business_domain": domain,
        "data_source": data_source,
        "is_fact_table": random.choice([True, False])
    }


def generate_column_metadata(table_name: str, column_index: int):
    column_name = f"col_{column_index}"
    column_comment = fake.word() + "字段"
    data_type = random.choice(["bigint", "int", "decimal(18,2)", "varchar(255)", "datetime"])
    metric_type = random.choice(METRIC_TYPES)

    return {
        "table_name": table_name,
        "column_name": column_name,
        "column_comment": column_comment,
        "data_type": data_type,
        "is_primary_key": column_name == "col_1",
        "is_foreign_key": random.choice([True, False]),
        "is_nullable": random.choice([True, False]),
        "business_meaning": f"用于表示{column_comment}的业务含义",
        "metric_type": metric_type,
        "enum_values": json.dumps({"A": "类型A", "B": "类型B"}, ensure_ascii=False)
    }


def build_embedding_text_table(table_meta: dict):
    return f"""
表名：{table_meta['table_name']}
数据库：{table_meta['schema_name']}
业务域：{table_meta['business_domain']}
数据源：{table_meta['data_source']}
是否事实表：{'是' if table_meta['is_fact_table'] else '否'}
描述：{table_meta['table_comment']}
"""


def build_embedding_text_column(column_meta: dict):
    return f"""
字段名：{column_meta['column_name']}
所属表：{column_meta['table_name']}
数据类型：{column_meta['data_type']}
是否主键：{'是' if column_meta['is_primary_key'] else '否'}
是否外键：{'是' if column_meta['is_foreign_key'] else '否'}
是否可空：{'是' if column_meta['is_nullable'] else '否'}
业务含义：{column_meta['business_meaning']}
指标类型：{column_meta['metric_type']}
枚举值：{column_meta['enum_values']}
"""


import os

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "tests", "data")

# 确保输出目录存在
os.makedirs(DATA_DIR, exist_ok=True)

def generate_all_metadata():
    tables = []
    columns = []

    for i in range(120):  # 120 张表
        table_meta = generate_table_metadata(i)
        table_meta["text"] = build_embedding_text_table(table_meta)
        tables.append(table_meta)

        column_count = random.randint(8, 20)

        for j in range(column_count):
            col_meta = generate_column_metadata(table_meta["table_name"], j + 1)
            col_meta["text"] = build_embedding_text_column(col_meta)
            columns.append(col_meta)

    print(f"生成表数量: {len(tables)}")
    print(f"生成字段数量: {len(columns)}")
    print(f"总数据量: {len(tables) + len(columns)}")
    print(f"数据输出目录: {DATA_DIR}")

    with open(os.path.join(DATA_DIR, "tables.json"), "w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False, indent=2)

    with open(os.path.join(DATA_DIR, "columns.json"), "w", encoding="utf-8") as f:
        json.dump(columns, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    generate_all_metadata()