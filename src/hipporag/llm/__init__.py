# LLM（大语言模型）模块初始化文件
# LLM (Large Language Model) module initialization file

import os

# 导入工具函数和配置类 / Import utility functions and configuration classes
from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig

# 导入各种 LLM 实现 / Import various LLM implementations
from .openai_gpt import CacheOpenAI  # 带缓存的 OpenAI GPT / OpenAI GPT with caching
from .base import BaseLLM  # LLM 基类 / LLM base class
from .bedrock_llm import BedrockLLM  # AWS Bedrock LLM
from .transformers_llm import TransformersLLM  # Hugging Face Transformers LLM

# 获取日志记录器 / Get logger
logger = get_logger(__name__)


def _get_llm_class(config: BaseConfig):
    """
    根据配置获取相应的 LLM 类实例
    Get the appropriate LLM class instance based on configuration
    
    该函数根据配置中的模型名称和 URL，返回对应的 LLM 实现实例。
    支持 OpenAI、AWS Bedrock 和 Hugging Face Transformers 模型。
    This function returns the corresponding LLM implementation instance based on
    model name and URL in config. Supports OpenAI, AWS Bedrock, and Hugging Face Transformers.
    
    参数 / Parameters:
        config: BaseConfig - 包含 LLM 配置的配置对象 / Configuration object with LLM settings
        
    返回值 / Returns:
        BaseLLM - LLM 实例 / LLM instance
    """
    # 如果使用本地部署的模型且没有设置 API 密钥，使用占位符 / Use placeholder if local deployment without API key
    if config.llm_base_url is not None and 'localhost' in config.llm_base_url and os.getenv('OPENAI_API_KEY') is None:
        os.environ['OPENAI_API_KEY'] = 'sk-'

    # 如果模型名称以 'bedrock' 开头，返回 Bedrock LLM / Return Bedrock LLM if name starts with 'bedrock'
    if config.llm_name.startswith('bedrock'):
        return BedrockLLM(config)
    
    # 如果模型名称以 'Transformers/' 开头，返回 Transformers LLM / Return Transformers LLM if name starts with 'Transformers/'
    if config.llm_name.startswith('Transformers/'):
        return TransformersLLM(config)
    
    # 默认返回带缓存的 OpenAI LLM / Default to cached OpenAI LLM
    return CacheOpenAI.from_experiment_config(config)
    