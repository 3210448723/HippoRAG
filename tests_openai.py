import os
from typing import List
import json
import argparse
import logging

from src.hipporag import HippoRAG

def main():
    """
    主函数：测试 HippoRAG 的核心功能
    Main function: Test the core functionalities of HippoRAG
    
    该函数演示了 HippoRAG 的以下功能：
    This function demonstrates the following functionalities of HippoRAG:
    1. 文档索引（indexing） - Document indexing
    2. 检索增强生成问答（RAG QA） - Retrieval-Augmented Generation QA
    3. 增量索引（incremental indexing） - Incremental indexing
    4. 文档删除（deletion） - Document deletion
    """

    # 准备数据集和评估数据
    # Prepare datasets and evaluation data
    docs = [
        "Oliver Badman is a politician.",  # Oliver Badman 是一位政治家
        "George Rankin is a politician.",  # George Rankin 是一位政治家
        "Thomas Marwick is a politician.",  # Thomas Marwick 是一位政治家
        "Cinderella attended the royal ball.",  # 灰姑娘参加了皇家舞会
        "The prince used the lost glass slipper to search the kingdom.",  # 王子用丢失的玻璃鞋搜寻整个王国
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",  # 当玻璃鞋完美合脚时，灰姑娘与王子重聚
        "Erik Hort's birthplace is Montebello.",  # Erik Hort 的出生地是 Montebello
        "Marina is bom in Minsk.",  # Marina 出生在 Minsk
        "Montebello is a part of Rockland County."  # Montebello 是 Rockland County 的一部分
    ]

    # 定义 HippoRAG 对象的保存目录（每个 LLM/嵌入模型组合将创建一个新的子目录）
    # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    save_dir = 'outputs/openai_test'
    # 任意 OpenAI 模型名称
    # Any OpenAI model name
    llm_model_name = 'gpt-4o-mini'
    # 嵌入模型名称（目前支持 NV-Embed、GritLM 或 Contriever）
    # Embedding model name (NV-Embed, GritLM or Contriever for now)
    embedding_model_name = 'text-embedding-3-small'

    # 启动一个 HippoRAG 实例
    # Startup a HippoRAG instance
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name)

    # 运行索引：对文档进行处理，构建知识图谱和嵌入向量
    # Run indexing: Process documents, build knowledge graph and embeddings
    hipporag.index(docs=docs)

    # 分离的检索和问答过程
    # Separate Retrieval & QA process
    queries = [
        "What is George Rankin's occupation?",  # George Rankin 的职业是什么？
        "How did Cinderella reach her happy ending?",  # 灰姑娘是如何达到幸福结局的？
        "What county is Erik Hort's birthplace a part of?"  # Erik Hort 的出生地属于哪个县？
    ]

    # 用于评估的标准答案
    # Standard answers for evaluation
    answers = [
        ["Politician"],  # 政治家
        ["By going to the ball."],  # 通过参加舞会
        ["Rockland County"]  # Rockland County
    ]

    # 用于评估的黄金标准文档（每个问题对应的相关文档）
    # Gold standard documents for evaluation (relevant documents for each question)
    gold_docs = [
        ["George Rankin is a politician."],  # George Rankin 是一位政治家
        ["Cinderella attended the royal ball.",  # 灰姑娘参加了皇家舞会
         "The prince used the lost glass slipper to search the kingdom.",  # 王子用丢失的玻璃鞋搜寻整个王国
         "When the slipper fit perfectly, Cinderella was reunited with the prince."],  # 当玻璃鞋完美合脚时，灰姑娘与王子重聚
        ["Erik Hort's birthplace is Montebello.",  # Erik Hort 的出生地是 Montebello
         "Montebello is a part of Rockland County."]  # Montebello 是 Rockland County 的一部分
    ]

    # 执行检索增强生成问答，并打印最后两项结果（检索评估和问答评估）
    # Perform retrieval-augmented generation QA, and print the last two items (retrieval evaluation and QA evaluation)
    print(hipporag.rag_qa(queries=queries,
                                  gold_docs=gold_docs,
                                  gold_answers=answers)[-2:])

    # 重新启动一个 HippoRAG 实例（测试从已保存的状态加载）
    # Restart a HippoRAG instance (testing loading from saved state)
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name)

    # 在加载已有索引后执行同样的查询，验证索引持久化功能
    # Execute the same queries after loading existing index, verifying index persistence
    print(hipporag.rag_qa(queries=queries,
                                  gold_docs=gold_docs,
                                  gold_answers=answers)[-2:])

    # 再次启动一个 HippoRAG 实例（测试增量索引功能）
    # Start another HippoRAG instance (testing incremental indexing)
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name)

    # 定义新文档用于增量索引测试
    # Define new documents for incremental indexing test
    new_docs = [
        "Tom Hort's birthplace is Montebello.",  # Tom Hort 的出生地是 Montebello
        "Sam Hort's birthplace is Montebello.",  # Sam Hort 的出生地是 Montebello
        "Bill Hort's birthplace is Montebello.",  # Bill Hort 的出生地是 Montebello
        "Cam Hort's birthplace is Montebello.",  # Cam Hort 的出生地是 Montebello
        "Montebello is a part of Rockland County.."]  # Montebello 是 Rockland County 的一部分（注意有两个句点）

    # 运行增量索引：将新文档添加到现有索引中
    # Run incremental indexing: Add new documents to existing index
    hipporag.index(docs=new_docs)

    # 在增量索引后执行同样的查询，观察新文档的影响
    # Execute the same queries after incremental indexing, observing the impact of new documents
    print(hipporag.rag_qa(queries=queries,
                          gold_docs=gold_docs,
                          gold_answers=answers)[-2:])

    # 定义要删除的文档列表（测试文档删除功能）
    # Define list of documents to delete (testing document deletion)
    docs_to_delete = [
        "Tom Hort's birthplace is Montebello.",  # Tom Hort 的出生地是 Montebello
        "Sam Hort's birthplace is Montebello.",  # Sam Hort 的出生地是 Montebello
        "Bill Hort's birthplace is Montebello.",  # Bill Hort 的出生地是 Montebello
        "Cam Hort's birthplace is Montebello.",  # Cam Hort 的出生地是 Montebello
        "Montebello is a part of Rockland County.."  # Montebello 是 Rockland County 的一部分（注意有两个句点）
    ]

    # 从索引中删除指定文档
    # Delete specified documents from index
    hipporag.delete(docs_to_delete)

    # 在删除文档后执行同样的查询，验证删除功能
    # Execute the same queries after deleting documents, verifying deletion functionality
    print(hipporag.rag_qa(queries=queries,
                          gold_docs=gold_docs,
                          gold_answers=answers)[-2:])

if __name__ == "__main__":
    # 程序入口点：当脚本直接运行时执行 main 函数
    # Program entry point: Execute main function when script is run directly
    main()
