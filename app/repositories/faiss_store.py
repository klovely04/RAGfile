"""FAISS 向量索引存储封装。"""

from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray

# 二维向量矩阵类型别名：shape=(n, d)。
FloatMatrix = NDArray[np.float32]
# 单条向量类型别名：shape=(d,)。
FloatVector = NDArray[np.float32]


class FaissStore:
    """FAISS 索引读写与检索操作封装。"""

    def __init__(self, index_path: str) -> None:
        """初始化索引文件路径。

        Args:
            index_path: FAISS 索引文件持久化路径。
        """
        self.index_path = Path(index_path)
        # 确保父目录存在，避免写入时报错。
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> faiss.Index | None:
        """加载磁盘中的索引。

        Returns:
            faiss.Index | None: 存在则返回索引对象，不存在返回 None。
        """
        if not self.index_path.exists():
            return None
        return faiss.read_index(str(self.index_path))

    def save(self, index: faiss.Index) -> None:
        """保存索引到磁盘。

        Args:
            index: 要保存的 FAISS 索引对象。
        """
        faiss.write_index(index, str(self.index_path))

    def rebuild(self, vectors: FloatMatrix) -> faiss.Index:
        """根据向量矩阵全量重建索引。

        Args:
            vectors: 形状为 `(n, d)` 的向量矩阵。

        Returns:
            faiss.Index: 新建并写盘后的索引对象。
        """
        if vectors.ndim != 2:
            raise ValueError("vectors 必须是二维矩阵")
        # 取向量维度 d。
        dimension = vectors.shape[1]
        # 使用内积索引（通常配合归一化向量近似余弦相似度）。
        index = faiss.IndexFlatIP(dimension)
        # 添加全部向量。
        index.add(vectors.astype(np.float32))
        # 持久化。
        self.save(index)
        return index

    def append(self, vectors: FloatMatrix) -> faiss.Index:
        """向现有索引追加向量。

        Args:
            vectors: 要追加的二维向量矩阵。

        Returns:
            faiss.Index: 追加后的索引对象（并已写盘）。
        """
        if vectors.ndim != 2:
            raise ValueError("vectors 必须是二维矩阵")
        # 先尝试加载现有索引。
        existing = self.load()
        # 若不存在则等价于全量重建。
        if existing is None:
            return self.rebuild(vectors)
        # 追加向量。
        existing.add(vectors.astype(np.float32))
        # 写回磁盘。
        self.save(existing)
        return existing

    @staticmethod
    def search(
        index: faiss.Index,
        query_vector: FloatVector,
        top_k: int,
    ) -> tuple[FloatVector, NDArray[np.int64]]:
        """执行单条查询向量的 Top-K 检索。

        Args:
            index: 已加载的 FAISS 索引对象。
            query_vector: 一维查询向量。
            top_k: 返回的候选数量。

        Returns:
            tuple[FloatVector, NDArray[np.int64]]:
                - 第一个元素是相似度分数数组；
                - 第二个元素是索引位置数组（对应 SQLite 映射表中的 vector_id）。
        """
        if query_vector.ndim != 1:
            raise ValueError("query_vector 必须是一维向量")
        # FAISS 需要二维输入，这里扩成 shape=(1, d)。
        scores, indices = index.search(query_vector[np.newaxis, :].astype(np.float32), top_k)
        # 只返回第一行结果（因为是单条查询）。
        return scores[0].astype(np.float32), indices[0].astype(np.int64)
