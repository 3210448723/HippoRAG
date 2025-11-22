# 嵌入向量存储模块 / Embedding Vector Storage Module
# 该模块提供嵌入向量的持久化存储和检索功能
# This module provides persistent storage and retrieval for embedding vectors

import numpy as np  # 数值计算 / Numerical computing
from tqdm import tqdm  # 进度条 / Progress bar
import os  # 操作系统接口 / Operating system interface
from typing import Union, Optional, List, Dict, Set, Any, Tuple, Literal  # 类型提示 / Type hints
import logging  # 日志记录 / Logging
from copy import deepcopy  # 深拷贝 / Deep copy
import pandas as pd  # 数据处理 / Data processing

from .utils.misc_utils import compute_mdhash_id, NerRawOutput, TripleRawOutput

# 获取日志记录器 / Get logger
logger = logging.getLogger(__name__)

class EmbeddingStore:
    """
    嵌入向量存储类：用于管理和持久化文本嵌入向量
    Embedding Store Class: Manages and persists text embedding vectors
    
    该类提供了插入、检索和删除文本嵌入向量的功能，使用 Parquet 文件格式
    进行持久化存储。支持批量操作和增量更新。
    This class provides insertion, retrieval, and deletion of text embedding vectors,
    using Parquet file format for persistent storage. Supports batch operations and
    incremental updates.
    """
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        """
        初始化嵌入向量存储实例
        Initialize the embedding store instance
        
        该方法配置必要的参数并设置工作目录。如果目录不存在，会自动创建。
        构建 Parquet 文件名并加载已有数据。
        This method configures necessary parameters and sets up the working directory.
        Creates directory if it doesn't exist. Constructs Parquet filename and loads
        existing data.

        参数 / Parameters:
            embedding_model: 用于生成嵌入向量的模型 / Model used for generating embeddings
            db_filename: 数据存储目录路径 / Directory path for data storage
            batch_size: 批处理大小 / Batch size for processing
            namespace: 数据隔离的唯一标识符（如 'chunk'、'entity'、'fact'）
                      Unique identifier for data segregation (e.g., 'chunk', 'entity', 'fact')

        功能说明 / Functionality:
        - 将参数赋值给实例变量 / Assigns parameters to instance variables
        - 检查并创建目录（如不存在）/ Checks and creates directory (if not exists)
        - 构建 Parquet 文件名 / Constructs Parquet filename
        - 加载已有数据 / Loads existing data
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace

        # 如果目录不存在，创建它 / Create directory if it doesn't exist
        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        # 构建 Parquet 文件的完整路径 / Construct full path for Parquet file
        self.filename = os.path.join(
            db_filename, f"vdb_{self.namespace}.parquet"
        )
        # 加载已有数据 / Load existing data
        self._load_data()

    def get_missing_string_hash_ids(self, texts: List[str]):
        """
        获取尚未存储的文本的哈希 ID
        Get hash IDs of texts that are not yet stored
        
        该方法计算文本的哈希 ID，并返回尚未在存储中的文本及其 ID。
        This method computes hash IDs for texts and returns texts and IDs not yet in storage.
        
        参数 / Parameters:
            texts: List[str] - 待检查的文本列表 / List of texts to check
            
        返回值 / Returns:
            Dict - 缺失的哈希 ID 到文本内容的映射 / Mapping from missing hash IDs to text content
        """
        nodes_dict = {}

        # 为每个文本计算哈希 ID / Compute hash ID for each text
        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # 从输入字典获取所有哈希 ID / Get all hash_ids from input dictionary
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  {}

        # 获取已存在的哈希 ID / Get existing hash IDs
        existing = self.hash_id_to_row.keys()

        # 过滤出缺失的哈希 ID / Filter out missing hash_ids
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        # 返回缺失的 ID 及其对应的文本 / Return missing IDs and corresponding texts
        return {h: {"hash_id": h, "content": t} for h, t in zip(missing_ids, texts_to_encode)}

    def insert_strings(self, texts: List[str]):
        """
        插入文本并生成其嵌入向量
        Insert texts and generate their embedding vectors
        
        该方法计算文本的哈希 ID，过滤出新文本，生成嵌入向量并存储。
        支持增量插入，已存在的文本会被跳过。
        This method computes hash IDs for texts, filters out new texts, generates
        embeddings and stores them. Supports incremental insertion, existing texts
        are skipped.
        
        参数 / Parameters:
            texts: List[str] - 待插入的文本列表 / List of texts to insert
        """
        nodes_dict = {}

        # 为每个文本计算哈希 ID / Compute hash ID for each text
        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # 从输入字典获取所有哈希 ID / Get all hash_ids from input dictionary
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  # 没有要插入的内容 / Nothing to insert

        # 获取已存在的哈希 ID / Get existing hash IDs
        existing = self.hash_id_to_row.keys()

        # 过滤出缺失的哈希 ID / Filter out missing hash_ids
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        # 记录插入统计信息 / Log insertion statistics
        logger.info(
            f"Inserting {len(missing_ids)} new records, {len(all_hash_ids) - len(missing_ids)} records already exist.")

        if not missing_ids:
            return  {}  # 所有记录都已存在 / All records already exist

        # 从 "content" 字段准备要编码的文本 / Prepare texts to encode from "content" field
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        # 使用嵌入模型批量生成嵌入向量 / Batch generate embeddings using embedding model
        missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)

        # 更新或插入数据 / Update or insert data
        self._upsert(missing_ids, texts_to_encode, missing_embeddings)

    def _load_data(self):
        """
        从 Parquet 文件加载已有数据
        Load existing data from Parquet file
        
        该方法从 Parquet 文件读取已保存的嵌入数据，并构建必要的索引结构。
        如果文件不存在，初始化为空数据结构。
        This method reads saved embedding data from Parquet file and constructs
        necessary index structures. Initializes empty structures if file doesn't exist.
        """
        # 如果 Parquet 文件存在，读取数据 / If Parquet file exists, read data
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            # 提取哈希 ID、文本和嵌入向量 / Extract hash IDs, texts, and embeddings
            self.hash_ids, self.texts, self.embeddings = df["hash_id"].values.tolist(), df["content"].values.tolist(), df["embedding"].values.tolist()
            # 构建哈希 ID 到索引的映射 / Build mapping from hash ID to index
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            # 构建哈希 ID 到行数据的映射 / Build mapping from hash ID to row data
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t}
                for h, t in zip(self.hash_ids, self.texts)
            }
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h  for idx, h in enumerate(self.hash_ids)}
            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logger.info(f"Loaded {len(self.hash_ids)} records from {self.filename}")
        else:
            self.hash_ids, self.texts, self.embeddings = [], [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}

    def _save_data(self):
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": self.texts,
            "embedding": self.embeddings
        })
        data_to_save.to_parquet(self.filename, index=False)
        self.hash_id_to_row = {h: {"hash_id": h, "content": t} for h, t, e in zip(self.hash_ids, self.texts, self.embeddings)}
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        logger.info(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, hash_ids, texts, embeddings):
        self.embeddings.extend(embeddings)
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)

        logger.info(f"Saving new records.")
        self._save_data()

    def delete(self, hash_ids):
        indices = []

        for hash in hash_ids:
            indices.append(self.hash_id_to_idx[hash])

        sorted_indices = np.sort(indices)[::-1]

        for idx in sorted_indices:
            self.hash_ids.pop(idx)
            self.texts.pop(idx)
            self.embeddings.pop(idx)

        logger.info(f"Saving record after deletion.")
        self._save_data()

    def get_row(self, hash_id):
        return self.hash_id_to_row[hash_id]

    def get_hash_id(self, text):
        return self.text_to_hash_id[text]

    def get_rows(self, hash_ids, dtype=np.float32):
        if not hash_ids:
            return {}

        results = {id : self.hash_id_to_row[id] for id in hash_ids}

        return results

    def get_all_ids(self):
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self):
        return deepcopy(self.hash_id_to_row)

    def get_all_texts(self):
        return set(row['content'] for row in self.hash_id_to_row.values())

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        return self.embeddings[self.hash_id_to_idx[hash_id]].astype(dtype)
    
    def get_embeddings(self, hash_ids, dtype=np.float32) -> list[np.ndarray]:
        if not hash_ids:
            return []

        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings, dtype=dtype)[indices]

        return embeddings