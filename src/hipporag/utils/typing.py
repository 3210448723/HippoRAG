# 类型定义模块 / Type Definitions Module
# 该模块定义了项目中使用的自定义类型
# This module defines custom types used throughout the project

from typing import Dict, Any, List, TypedDict, Tuple

# Triple 类型：表示知识图谱中的三元组（主语，关系，宾语）
# Triple type: Represents a triple in knowledge graph (subject, relation, object)
Triple = Tuple[str, str, str]