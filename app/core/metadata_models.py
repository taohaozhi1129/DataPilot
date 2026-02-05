from enum import Enum
from typing import List, Optional, Any, Dict, Union
from pydantic import BaseModel, Field

# 1. 定义元数据类型枚举
class MetadataType(str, Enum):
    TABLE = "rdbms_table"       # 关系型数据库表
    FILE = "file"               # 文档/文件
    NOSQL = "nosql_collection"  # NoSQL 集合 (如 MongoDB, Redis)

# --- 基础组件 ---
class ColumnMetadata(BaseModel):
    """列元数据"""
    name: str
    type: str
    comment: Optional[str] = None
    is_primary_key: bool = False

# --- 具体元数据模型 ---

class TableMetadata(BaseModel):
    """关系型表元数据"""
    type: MetadataType = MetadataType.TABLE
    db_name: str
    table_name: str
    columns: List[ColumnMetadata]
    ddl: Optional[str] = None  # 可选：存储建表语句
    comment: Optional[str] = None

class FileMetadata(BaseModel):
    """文件/非结构化文档元数据"""
    type: MetadataType = MetadataType.FILE
    file_name: str
    file_path: str
    file_size: Optional[int] = None
    file_type: str = Field(..., description="pdf, docx, txt, md, etc.")
    last_modified: Optional[str] = None
    summary: Optional[str] = None  # 文件的摘要信息

class NoSQLMetadata(BaseModel):
    """NoSQL 集合/键值对元数据"""
    type: MetadataType = MetadataType.NOSQL
    db_type: str = Field(..., description="mongodb, redis, cassandra, etc.")
    collection_name: str
    schema_validation: Optional[Dict[str, Any]] = None  # MongoDB 的 $jsonSchema
    sample_document: Optional[Dict[str, Any]] = None    # 存储一个样例文档结构

# 联合类型，用于类型检查
MetadataUnion = Union[TableMetadata, FileMetadata, NoSQLMetadata]

# --- 任务模板模型 (用于 template_collection) ---

class TaskTemplatePayload(BaseModel):
    """任务模板 Payload"""
    task_type: str = Field(..., description="text_to_sql, summary, extraction, etc.")
    template: str  # Jinja2 风格的 Prompt 模板
    required_vars: List[str]  # 模板需要的变量名，如 ['schema', 'question']
    description: Optional[str] = None
    version: str = "1.0"
