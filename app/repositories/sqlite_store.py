"""SQLite 持久化实现。

本仓储负责存储：
1. 文档表（documents）；
2. 分块表（chunks）；
3. 向量位置映射表（index_entries）；
4. 聊天历史表（chat_messages）。
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.models import Chunk, Document


@dataclass(slots=True, frozen=True)
class StoredChunk:
    """从数据库读取出来的分块模型。"""

    chunk_id: str
    source: str
    doc_type: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, str]


class SQLiteStore:
    """SQLite 仓储类。"""

    def __init__(self, db_path: str) -> None:
        """初始化数据库连接信息并确保表结构存在。

        Args:
            db_path: SQLite 数据库文件路径。
        """
        self.db_path = Path(db_path)
        # 确保数据库目录存在。
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # 初始化表结构。
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """创建数据库连接。

        Returns:
            sqlite3.Connection: 行对象支持按列名访问。
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """创建所有必需表（若不存在）。"""
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL UNIQUE,
                    doc_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT NOT NULL UNIQUE,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS index_entries (
                    vector_id INTEGER PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def upsert_document_with_chunks(self, document: Document, chunks: list[Chunk]) -> None:
        """写入/更新文档，并替换该文档全部分块。

        Args:
            document: 文档模型。
            chunks: 文档对应的分块列表。
        """
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            # 先 upsert 文档主记录。
            conn.execute(
                """
                INSERT INTO documents(
                    source, doc_type, content, metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source) DO UPDATE SET
                    doc_type=excluded.doc_type,
                    content=excluded.content,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    document.source,
                    document.doc_type,
                    document.content,
                    json.dumps(document.metadata, ensure_ascii=False),
                    now,
                    now,
                ),
            )

            # 查出文档主键，用于 chunk 外键。
            row = conn.execute(
                "SELECT id FROM documents WHERE source = ?",
                (document.source,),
            ).fetchone()
            if row is None:
                raise RuntimeError("写入 documents 后未查询到文档 ID")

            document_id = int(row["id"])
            # 替换策略：先删旧分块，再插新分块。
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO chunks(
                        chunk_id, document_id, chunk_index, text,
                        start_char, end_char, metadata_json, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        document_id,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.start_char,
                        chunk.end_char,
                        json.dumps(chunk.metadata, ensure_ascii=False),
                        now,
                        now,
                    ),
                )

    def list_all_chunks(self) -> list[StoredChunk]:
        """查询所有分块（按文档和分块序号排序）。

        Returns:
            list[StoredChunk]: 全量分块列表。
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    c.chunk_id, d.source, d.doc_type, c.chunk_index, c.text,
                    c.start_char, c.end_char, c.metadata_json
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY d.id, c.chunk_index
                """
            ).fetchall()
        return [self._row_to_stored_chunk(row) for row in rows]

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[StoredChunk]:
        """按分块 ID 批量查询，且保持输入顺序。

        Args:
            chunk_ids: 分块 ID 列表。

        Returns:
            list[StoredChunk]: 按输入顺序返回的分块列表。
        """
        if not chunk_ids:
            return []

        placeholders = ",".join("?" for _ in chunk_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    c.chunk_id, d.source, d.doc_type, c.chunk_index, c.text,
                    c.start_char, c.end_char, c.metadata_json
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.chunk_id IN ({placeholders})
                """,
                tuple(chunk_ids),
            ).fetchall()

        # 临时字典用于恢复输入顺序。
        by_id = {row["chunk_id"]: self._row_to_stored_chunk(row) for row in rows}
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def replace_index_entries(self, chunk_ids: list[str]) -> None:
        """全量替换向量位置映射表。

        Args:
            chunk_ids: 与 FAISS 向量位置一一对应的分块 ID 列表。
        """
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            # 清空旧映射。
            conn.execute("DELETE FROM index_entries")
            # 从 0 开始重建 vector_id -> chunk_id 映射。
            conn.executemany(
                "INSERT INTO index_entries(vector_id, chunk_id, created_at) VALUES (?, ?, ?)",
                [(index, chunk_id, now) for index, chunk_id in enumerate(chunk_ids)],
            )

    def append_index_entries(self, chunk_ids: list[str]) -> None:
        """向映射表追加新向量位置。

        Args:
            chunk_ids: 需要追加映射的分块 ID 列表。
        """
        if not chunk_ids:
            return

        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            # 找到当前最大 vector_id。
            row = conn.execute(
                "SELECT COALESCE(MAX(vector_id), -1) AS max_id FROM index_entries"
            ).fetchone()
            if row is None:
                raise RuntimeError("读取 index_entries 最大 ID 失败")

            start = int(row["max_id"]) + 1
            conn.executemany(
                "INSERT INTO index_entries(vector_id, chunk_id, created_at) VALUES (?, ?, ?)",
                [(start + index, chunk_id, now) for index, chunk_id in enumerate(chunk_ids)],
            )

    def list_index_chunk_ids(self) -> list[str]:
        """按向量顺序返回分块 ID 列表。

        Returns:
            list[str]: 与 FAISS 检索返回下标一致的 chunk_id 顺序列表。
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM index_entries ORDER BY vector_id ASC"
            ).fetchall()
        return [str(row["chunk_id"]) for row in rows]

    def save_chat_message(self, session_id: str, role: str, content: str) -> None:
        """保存一条聊天消息。

        Args:
            session_id: 会话 ID。
            role: 消息角色（`user` 或 `assistant`）。
            content: 消息内容。
        """
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages(session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, now),
            )

    def list_chat_history(self, session_id: str, limit: int) -> list[dict[str, str]]:
        """查询会话历史（按时间正序返回）。

        Args:
            session_id: 会话 ID。
            limit: 最大返回条数。

        Returns:
            list[dict[str, str]]: 每条包含 `role` 和 `content`。
        """
        with self._connect() as conn:
            # 先按倒序取最近 N 条，再在 Python 中反转为正序。
            rows = conn.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        return [
            {"role": str(row["role"]), "content": str(row["content"])}
            for row in reversed(rows)
        ]

    @staticmethod
    def _row_to_stored_chunk(row: sqlite3.Row) -> StoredChunk:
        """把 SQLite 行对象转换为 `StoredChunk`。

        Args:
            row: SQLite 查询返回的一行。

        Returns:
            StoredChunk: 类型化后的分块对象。
        """
        return StoredChunk(
            chunk_id=str(row["chunk_id"]),
            source=str(row["source"]),
            doc_type=str(row["doc_type"]),
            chunk_index=int(row["chunk_index"]),
            text=str(row["text"]),
            start_char=int(row["start_char"]),
            end_char=int(row["end_char"]),
            metadata=json.loads(str(row["metadata_json"])),
        )
