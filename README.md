# RAG Knowledge Assistant

一个面向初学者的最小可用 RAG（Retrieval-Augmented Generation，检索增强生成）项目。

这个项目解决的问题是：
- 你有一批本地文档（`txt/md/pdf/docx`）。
- 你希望上传后自动切分、向量化、建立索引。
- 你希望通过聊天接口基于文档内容问答，而不是“纯大模型胡说”。

## 1. 项目在做什么

核心能力：
- 文档解析：读取 `txt/md/pdf/docx`。
- 文本切块：把长文按规则切成 `Chunk`。
- 向量化：把 chunk 转成向量（默认可用哈希向量，生产可换 `sentence-transformers`）。
- 向量索引：使用 FAISS 做相似度检索。
- 持久化：使用 SQLite 保存文档、chunk、索引映射、聊天历史。
- 问答编排：先检索，再把上下文喂给 LLM（DeepSeek）生成答案。
- 前端调试页：提供 Streamlit 页面做上传、重建索引、问答、历史查看。

## 2. 技术栈

- 后端框架：FastAPI
- 数据存储：SQLite
- 向量索引：FAISS (`faiss-cpu`)
- 向量模型：`sentence-transformers`（可选）+ 内置 hash fallback
- LLM 调用：DeepSeek Chat Completions API
- 前端：Streamlit
- 测试：Pytest

## 3. 快速启动（本地）

### 3.1 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

### 3.2 配置环境变量

```bash
copy .env.example .env
```

必须关注的配置：
- `DEEPSEEK_API_KEY`：不填也能跑上传/检索流程，但聊天会走兜底提示。
- `EMBEDDING_PROVIDER`：
  - `hash`：零依赖、可快速跑通。
  - `sentence_transformers`：真实语义检索效果更好。

### 3.3 启动后端 API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

访问：
- Swagger 文档：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/api/v1/health`

### 3.4 启动 Streamlit 前端

```bash
streamlit run frontend/streamlit_app.py
```

## 4. 典型使用流程

1. 上传文档（`POST /api/v1/upload`）。
2. 后端会：解析文档 -> 切块 -> 写 SQLite -> 重建 FAISS 索引。
3. 发送问题（`POST /api/v1/chat`）。
4. 后端会：检索相关 chunk -> 拼接提示词 -> 调用 DeepSeek -> 返回答案。
5. 查询历史（`GET /api/v1/history/{session_id}`）。

## 5. API 一览

- `GET /api/v1/health`
  - 服务健康检查。
- `POST /api/v1/documents/parse`
  - 只解析文档，不入库不建索引。
- `POST /api/v1/upload`
  - 上传文档并重建索引。
- `POST /api/v1/index/rebuild`
  - 仅重建索引（从当前 SQLite 全量 chunks 重建）。
- `POST /api/v1/chat`
  - 问答接口（支持 `session_id`、`source_filter`）。
- `GET /api/v1/history/{session_id}`
  - 查看会话历史。

## 6. 目录结构（按职责）

```text
app/
  api/routes/         # HTTP 路由层（参数校验、响应结构）
  core/               # 配置、日志、依赖注入
  models/             # 领域模型（Document/Chunk/RetrievalHit/ChatResult）
  repositories/       # 持久化层（SQLite/FAISS）
  services/           # 业务逻辑层（解析、切块、向量化、检索、聊天编排）
  main.py             # FastAPI 应用入口

frontend/
  streamlit_app.py    # 本地调试用 UI

tests/
  test_*.py           # 单元与接口测试
```

## 7. 架构分层理解（给初学者）

- Route 层（`api/routes`）：
  - 只做“收请求、调服务、组响应”。
  - 不放复杂业务逻辑。
- Service 层（`services`）：
  - 真正的业务编排：解析、切块、索引、检索、问答。
- Repository 层（`repositories`）：
  - 和存储打交道（SQLite、FAISS），不关心 HTTP。
- Model 层（`models`）：
  - 统一数据结构，减少“字典乱飞”。

你可以把它理解成：`Controller -> Service -> Repository -> Storage`。

## 8. 常见开发命令

```bash
pytest
ruff check .
ruff format .
mypy app
```

## 9. 新手建议阅读顺序

推荐按这个顺序读代码：
1. `app/main.py`（应用入口）
2. `app/api/routes/rag.py`（核心接口）
3. `app/services/rag_chat_service.py`（RAG 编排）
4. `app/services/retriever_service.py` + `app/services/index_service.py`
5. `app/repositories/sqlite_store.py` + `app/repositories/faiss_store.py`
6. `tests/test_rag_api.py`（从测试看完整流程）
