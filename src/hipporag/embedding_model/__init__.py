# 嵌入模型（Embedding Model）模块初始化文件
# Embedding Model module initialization file

# 导入各种嵌入模型实现 / Import various embedding model implementations
from .Contriever import ContrieverModel  # Contriever 嵌入模型
from .base import EmbeddingConfig, BaseEmbeddingModel  # 基础配置和基类 / Base config and class
from .GritLM import GritLMEmbeddingModel  # GritLM 嵌入模型
from .NVEmbedV2 import NVEmbedV2EmbeddingModel  # NVIDIA Embed V2 嵌入模型
from .OpenAI import OpenAIEmbeddingModel  # OpenAI 嵌入模型
from .Cohere import CohereEmbeddingModel  # Cohere 嵌入模型
from .Transformers import TransformersEmbeddingModel  # Hugging Face Transformers 嵌入模型
from .VLLM import VLLMEmbeddingModel  # VLLM 嵌入模型

# 导入日志工具 / Import logging utility
from ..utils.logging_utils import get_logger

# 获取日志记录器 / Get logger
logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    """
    根据模型名称获取相应的嵌入模型类
    Get the appropriate embedding model class based on model name
    
    该函数根据嵌入模型的名称字符串，返回对应的嵌入模型类。
    支持多种嵌入模型：GritLM、NV-Embed-v2、Contriever、OpenAI、Cohere、
    Transformers 和 VLLM。
    This function returns the corresponding embedding model class based on the
    embedding model name string. Supports multiple models: GritLM, NV-Embed-v2,
    Contriever, OpenAI, Cohere, Transformers, and VLLM.
    
    参数 / Parameters:
        embedding_model_name: str - 嵌入模型名称，默认为 "nvidia/NV-Embed-v2"
                                   Embedding model name, defaults to "nvidia/NV-Embed-v2"
        
    返回值 / Returns:
        Type[BaseEmbeddingModel] - 嵌入模型类 / Embedding model class
        
    异常 / Raises:
        AssertionError - 如果模型名称未知 / If model name is unknown
    """
    # 根据模型名称返回相应的嵌入模型类 / Return corresponding embedding model class based on name
    if "GritLM" in embedding_model_name:
        return GritLMEmbeddingModel
    elif "NV-Embed-v2" in embedding_model_name:
        return NVEmbedV2EmbeddingModel
    elif "contriever" in embedding_model_name:
        return ContrieverModel
    elif "text-embedding" in embedding_model_name:  # OpenAI 的嵌入模型 / OpenAI embedding models
        return OpenAIEmbeddingModel
    elif "cohere" in embedding_model_name:
        return CohereEmbeddingModel
    elif embedding_model_name.startswith("Transformers/"):  # Transformers 前缀 / Transformers prefix
        return TransformersEmbeddingModel
    elif embedding_model_name.startswith("VLLM/"):  # VLLM 前缀 / VLLM prefix
        return VLLMEmbeddingModel
    # 如果没有匹配的模型，抛出断言错误 / Throw assertion error if no model matches
    assert False, f"Unknown embedding model name: {embedding_model_name}"