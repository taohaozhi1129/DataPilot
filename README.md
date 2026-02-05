# DataPilot (Data Intelligence Agent)

> **赋能数据中台的下一代 AI 智能体基石**

> *Automating Data Operations with Reasoning & RAG*

## 🌟 项目愿景 (Vision)
DataPilot 旨在成为连接**业务人员**与**底层数据**的智能桥梁。它不仅仅是一个简单的问答机器人，而是一个具备**推理能力 (Reasoning)** 和 **执行能力 (Action)** 的智能体 (Agent)。

通过集成 RAG (检索增强生成) 和 LangGraph (工作流编排)，DataPilot 能够理解复杂的业务指令，自动完成从**数据查询 (Text-to-SQL)** 到**任务构建 (Text-to-Task)** 的全流程操作，显著降低数据中台的使用门槛。

## 🚀 核心能力 (Core Capabilities)

### 1. 🧠 智能意图识别 (Intent Reasoning)
不再依赖死板的关键词匹配。DataPilot 内置思维链 (Chain of Thought)，能够深度理解用户语境。
- **动态路由**: 自动区分闲聊、数据查询、任务构建等多种场景。
- **歧义处理**: 遇到模糊指令时，会结合历史上下文进行判断。

### 2. 📊 Text-to-SQL (RAG Enhanced)
让不懂 SQL 的业务人员也能自由查数。
- **Schema 混合检索**: 稠密向量 + BM25 关键词检索，兼顾语义召回与精确匹配（中英文）。
- **安全生成**: 严格的权限控制，只生成 SELECT 语句，拒绝危险操作。
- **智能纠错**: 生成的 SQL 符合标准 ANSI 语法或特定方言（MySQL/ClickHouse 等）。

### 3. ⚙️ Text-to-Task (Automation)
一句话创建复杂的数据集成任务。
- **模板匹配**: 根据描述自动检索相似的任务模板。
- **槽位填充 (Slot Filling)**: 自动提取源库、目标库、表名等关键参数。
- **自动配置**: 生成可直接执行的 JSON 任务配置。

### 4. 🔄 多轮交互 (Context Aware)
支持类似人类的连续对话。
- **状态记忆**: 记住上一轮的查询结果或任务状态。
- **主动追问**: 当关键信息缺失时（如未指定目标表），Agent 会主动发起追问，直到任务信息完整。

## 🛠️ 技术栈 (Tech Stack)

- **LLM Orchestration**: [LangChain](https://www.langchain.com/) / [LangGraph](https://www.langchain.com/langgraph)
- **Model**: OpenAI / DeepSeek / Compatible LLMs
- **Vector DB**: [Milvus 2.6+](https://milvus.io/) (Feature Vectors)
- **Embedding**: BAAI/bge-base-zh (State-of-the-art Chinese Embeddings)
- **Backend API**: [FastAPI](https://fastapi.tiangolo.com/)
- **State Persistence**: Redis (Production-grade memory persistence)
- **Validation**: Pydantic V2

## 📦 快速开始 (Quick Start)

### 1. 环境准备
```bash
# 推荐使用 Python 3.10+
pip install -r requirements.txt
```

### 2. 配置
所有配置集中在 `config.py`，建议通过环境变量覆盖。
至少需要设置：
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`（如使用 OpenAI-compatible 服务）
- `OPENAI_MODEL_NAME`
建议设置：
- `BM25_ANALYZER`（默认 `jieba`，适合中英文混合）
如已有旧集合，请重建 `metadata_collection` / `template_collection` 以启用 BM25 字段与索引。
并确保 Redis 与 Milvus 服务已启动。

### 3. 启动服务
```bash
uvicorn main:app --reload
```

### 4. 可选：本地接口测试
```bash
python test_api.py
```

## 📝 许可证
MIT License
