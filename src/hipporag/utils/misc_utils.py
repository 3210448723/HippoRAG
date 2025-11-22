# 杂项工具函数模块 / Miscellaneous Utility Functions Module
# 该模块包含各种辅助数据类和工具函数
# This module contains various helper data classes and utility functions

from argparse import ArgumentTypeError
from dataclasses import dataclass
from hashlib import md5
from typing import Dict, Any, List, Tuple, Literal, Union, Optional
import numpy as np
import re
import logging

from .typing import Triple
from .llm_utils import filter_invalid_triples

logger = logging.getLogger(__name__)

@dataclass
class NerRawOutput:
    """
    命名实体识别（NER）原始输出数据类
    Named Entity Recognition (NER) Raw Output Data Class
    
    用于存储从文档块中提取的命名实体信息
    Stores named entity information extracted from document chunks
    """
    chunk_id: str  # 文档块 ID / Chunk ID
    response: str  # LLM 的原始响应 / Raw LLM response
    unique_entities: List[str]  # 唯一实体列表 / List of unique entities
    metadata: Dict[str, Any]  # 元数据 / Metadata


@dataclass
class TripleRawOutput:
    """
    三元组提取原始输出数据类
    Triple Extraction Raw Output Data Class
    
    用于存储从文档块中提取的知识三元组信息
    Stores knowledge triple information extracted from document chunks
    """
    chunk_id: str  # 文档块 ID / Chunk ID
    response: str  # LLM 的原始响应 / Raw LLM response
    triples: List[List[str]]  # 三元组列表（每个三元组为 [主语, 关系, 宾语]）/ List of triples (each triple is [subject, relation, object])
    metadata: Dict[str, Any]  # 元数据 / Metadata

@dataclass
class LinkingOutput:
    """
    链接输出数据类
    Linking Output Data Class
    
    用于存储实体链接的分数和类型信息
    Stores entity linking score and type information
    """
    score: np.ndarray  # 链接分数数组 / Array of linking scores
    type: Literal['node', 'dpr']  # 链接类型：'node'（基于节点）或 'dpr'（密集段落检索）/ Linking type: 'node' (node-based) or 'dpr' (dense passage retrieval)

@dataclass
class QuerySolution:
    """
    查询解决方案数据类
    Query Solution Data Class
    
    用于存储查询的检索结果、答案和评估信息
    Stores retrieval results, answers and evaluation information for queries
    """
    question: str  # 查询问题 / Query question
    docs: List[str]  # 检索到的文档列表 / List of retrieved documents
    doc_scores: np.ndarray = None  # 文档分数 / Document scores
    answer: str = None  # 生成的答案 / Generated answer
    gold_answers: List[str] = None  # 黄金标准答案列表 / List of gold standard answers
    gold_docs: Optional[List[str]] = None  # 黄金标准文档列表 / List of gold standard documents


    def to_dict(self):
        """
        将 QuerySolution 转换为字典格式
        Convert QuerySolution to dictionary format
        
        用于序列化和日志记录，仅保留前 5 个文档及其分数
        Used for serialization and logging, keeps only top 5 documents and scores
        """
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs[:5],  # 仅保留前 5 个文档 / Keep only top 5 documents
            "doc_scores": [round(v, 4) for v in self.doc_scores.tolist()[:5]]  if self.doc_scores is not None else None,  # 四舍五入到 4 位小数 / Round to 4 decimal places
            "gold_docs": self.gold_docs,
        }

def text_processing(text):
    """
    文本预处理函数
    Text preprocessing function
    
    将文本标准化：转换为小写，仅保留字母和数字，移除特殊字符
    Normalizes text: converts to lowercase, keeps only letters and numbers, removes special characters
    
    参数 / Parameters:
        text: str 或 list - 待处理的文本或文本列表 / Text or list of texts to process
        
    返回值 / Returns:
        str 或 list - 处理后的文本 / Processed text(s)
    """
    # 如果是列表，递归处理每个元素 / If list, recursively process each element
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    # 确保是字符串类型 / Ensure it's a string
    if not isinstance(text, str):
        text = str(text)
    # 转换为小写，仅保留字母数字和空格 / Convert to lowercase, keep only alphanumeric and spaces
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()

def reformat_openie_results(corpus_openie_results) -> (Dict[str, NerRawOutput], Dict[str, TripleRawOutput]):
    """
    重新格式化 OpenIE 结果
    Reformat OpenIE results
    
    将 OpenIE 提取的原始结果转换为标准的 NerRawOutput 和 TripleRawOutput 格式
    Converts raw OpenIE extraction results to standard NerRawOutput and TripleRawOutput formats
    
    参数 / Parameters:
        corpus_openie_results: 原始 OpenIE 结果列表 / List of raw OpenIE results
        
    返回值 / Returns:
        Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]] - 
        格式化的 NER 结果和三元组结果字典 / Formatted NER and triple result dictionaries
    """
    # 构建 NER 输出字典 / Build NER output dictionary
    ner_output_dict = {
        chunk_item['idx']: NerRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            unique_entities=list(np.unique(chunk_item['extracted_entities']))  # 去重实体 / Deduplicate entities
        )
        for chunk_item in corpus_openie_results
    }
    # 构建三元组输出字典 / Build triple output dictionary
    triple_output_dict = {
        chunk_item['idx']: TripleRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            triples=filter_invalid_triples(triples=chunk_item['extracted_triples'])  # 过滤无效三元组 / Filter invalid triples
        )
        for chunk_item in corpus_openie_results
    }

    return ner_output_dict, triple_output_dict

def extract_entity_nodes(chunk_triples: List[List[Triple]]) -> (List[str], List[List[str]]):
    """
    从三元组中提取实体节点
    Extract entity nodes from triples
    
    从每个文档块的三元组中提取唯一的实体（主语和宾语），构建图节点列表
    Extracts unique entities (subjects and objects) from triples in each chunk to build graph nodes
    
    参数 / Parameters:
        chunk_triples: List[List[Triple]] - 每个文档块的三元组列表 / List of triples for each chunk
        
    返回值 / Returns:
        Tuple[List[str], List[List[str]]] - 
        (所有唯一实体列表, 每个块的实体列表) / (List of all unique entities, List of entities per chunk)
    """
    chunk_triple_entities = []  # 每个块的唯一实体列表 / List of unique entities from each chunk's triples
    for triples in chunk_triples:
        triple_entities = set()
        for t in triples:
            if len(t) == 3:
                # 添加主语和宾语 / Add subject and object
                triple_entities.update([t[0], t[2]])
            else:
                # 记录无效三元组 / Log invalid triple
                logger.warning(f"During graph construction, invalid triple is found: {t}")
        chunk_triple_entities.append(list(triple_entities))
    # 从所有块中提取唯一实体 / Extract unique entities from all chunks
    graph_nodes = list(np.unique([ent for ents in chunk_triple_entities for ent in ents]))
    return graph_nodes, chunk_triple_entities

def flatten_facts(chunk_triples: List[Triple]) -> List[Triple]:
    """
    展平事实三元组
    Flatten fact triples
    
    将所有文档块的三元组合并为一个单一列表
    Merges triples from all chunks into a single list
    
    参数 / Parameters:
        chunk_triples: List[Triple] - 每个文档块的三元组列表 / List of triples per chunk
        
    返回值 / Returns:
        List[Triple] - 展平后的三元组列表 / Flattened list of triples
    """
    graph_triples = []  # 所有块的唯一关系三元组列表 / List of unique relation triples from all chunks
    for triples in chunk_triples:
        graph_triples.extend([tuple(t) for t in triples])
    graph_triples = list(set(graph_triples))
    return graph_triples

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x
    
    return (x - min_val) / range_val

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    return prefix + md5(content.encode()).hexdigest()


def all_values_of_same_length(data: dict) -> bool:
    """
    Return True if all values in 'data' have the same length or data is an empty dict,
    otherwise return False.
    """
    # Get an iterator over the dictionary's values
    value_iter = iter(data.values())

    # Get the length of the first sequence (handle empty dict case safely)
    try:
        first_length = len(next(value_iter))
    except StopIteration:
        # If the dictionary is empty, treat it as all having "the same length"
        return True

    # Check that every remaining sequence has this same length
    return all(len(seq) == first_length for seq in value_iter)


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )
