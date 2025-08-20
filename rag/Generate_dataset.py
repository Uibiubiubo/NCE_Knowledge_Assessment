import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings  # HuggingFaceEmbeddings是LangChain接口，用于加载本地的Hugging Face格式模型
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
import pandas as pd
import aiohttp
import json
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional

# DeepSeek API配置
DEEPSEEK_API_KEY = "sk-c6c279dbfe08466588189cb7e1fcb33b"  # 请替换为您的实际DeepSeek API密钥

# 配置参数
MARKDOWN_DIR = "./markdown_base"  # 您的Markdown文件夹路径
OUTPUT_FILE = "question_answer_dataset.csv"  # 输出文件名
TEST_SET_SIZE = 10  # 增加问答对数量以覆盖更多文档
LOCAL_EMBEDDING_MODEL_PATH = "./models/bge-m3"  # 本地BGE嵌入模型路径

# 自定义DeepSeek LLM类
class DeepSeekLLM(LLM):
    api_key: str                # DeepSeek API密钥，必须提供
    model_name: str = "deepseek-chat"       # 默认使用的模型名称
    temperature: float = 0.1                    # 温度参数，控制输出的随机性，较低值使输出更确定
    max_tokens: int = 4000              # 生成文本的最大标记数
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        # 同步调用API
        import requests
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"API请求失败: {response.status_code}, {response.text}")
        
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]

def generate_qa_dataset():
    print("开始加载文档...")
    # 加载Markdown文件
    loader = DirectoryLoader(MARKDOWN_DIR, glob="**/*.md")
    docs = loader.load()
    print(f"成功加载 {len(docs)} 个文档")
    
    # 计算每个文档应生成的问答对数量
    docs_count = len(docs)
    qa_per_doc = max(5, TEST_SET_SIZE // docs_count)  # 确保每个文档至少生成5个问答对
    total_qa = qa_per_doc * docs_count
    print(f"计划为每个文档生成约 {qa_per_doc} 个问答对，总计约 {total_qa} 个")

    print("初始化DeepSeek LLM和本地BGE嵌入模型...")
    # 初始化DeepSeek LLM
    deepseek_llm = DeepSeekLLM(api_key=DEEPSEEK_API_KEY)
    generator_llm = LangchainLLMWrapper(deepseek_llm)
    
    # 使用本地的BGE嵌入模型
    print(f"加载本地嵌入模型: {LOCAL_EMBEDDING_MODEL_PATH}")
    local_embeddings = HuggingFaceEmbeddings(
        model_name=LOCAL_EMBEDDING_MODEL_PATH,
        model_kwargs={'device': 'cuda'},  # 使用GPU加速，如果不需要可改为'cpu'
        encode_kwargs={'normalize_embeddings': True}  # 标准化嵌入向量
    )
    generator_embeddings = LangchainEmbeddingsWrapper(local_embeddings)

    print("创建测试集生成器...")
    # 创建知识图和应用转换
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.transforms import default_transforms, apply_transforms

    print("构建知识图...")
    kg = KnowledgeGraph()
    for doc in docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
            )
        )
    
    print("应用知识图转换...")
    trans = default_transforms(documents=docs, llm=generator_llm, embedding_model=generator_embeddings)
    apply_transforms(kg, trans)
    
    # 自定义查询分布，增加覆盖率
    from ragas.testset.synthesizers import (
        SingleHopSpecificQuerySynthesizer,
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer
    )
    
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.4),
        (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.3),
        (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.3)
    ]
    
    # 创建测试集生成器
    generator = TestsetGenerator(
        llm=generator_llm, 
        embedding_model=generator_embeddings,
        knowledge_graph=kg
    )
    
    print(f"开始生成问答对...")
    # 生成测试集
    dataset = generator.generate(testset_size=total_qa, query_distribution=query_distribution)
    
    # 将测试集转换为pandas DataFrame
    df = dataset.to_pandas()
    
    # 保存为CSV文件
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"问答数据集已保存到 {OUTPUT_FILE}，共生成 {len(df)} 个问答对")
    
    # 显示生成的问答对示例
    print("\n生成的问答对示例:")
    for i, row in df.head(5).iterrows():
        print(f"问题 {i+1}: {row['question']}")
        print(f"答案: {row['ground_truth']}")
        print("-" * 50)
    
    return df

if __name__ == "__main__":
    generate_qa_dataset()
