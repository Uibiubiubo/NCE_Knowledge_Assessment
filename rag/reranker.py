import os
import torch
from typing import List, Dict, Any, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

class Reranker:
    """
    使用BGE Reranker重新排序搜索结果
    """
    
    def __init__(self, model_path: str = "./models/bge-reranker-v2-m3", use_gpu: bool = True):
        """
        初始化重排序模型
        
        Args:
            model_path: 本地模型路径
            use_gpu: 是否使用GPU
        """
        self.model_path = model_path
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # 延迟加载模型
        self.tokenizer = None
        self.model = None
    
    def _load_model(self):
        """加载重排序模型和分词器"""
        if self.model is None:
            try:
                print(f"加载重排序模型: {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                
                if self.use_gpu:
                    self.model = self.model.to("cuda")
                    print("成功将重排序模型移动到GPU")
                    
                # 设置为评估模式
                self.model.eval()
                print("重排序模型加载成功")
            except Exception as e:
                print(f"加载重排序模型失败: {e}")
                raise
    
    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        重新排序搜索结果
        
        Args:
            query: 查询文本
            results: 原始搜索结果列表，格式为 [{"chunk": {...}, "distance": float, ...}, ...]
            top_k: 返回的结果数量，如果为None则返回所有排序后的结果
            
        Returns:
            List[Dict[str, Any]]: 重新排序后的结果列表
        """
        if not results:
            return []
            
        # 加载模型
        try:
            self._load_model()
        except Exception as e:
            print(f"无法加载重排序模型: {e}")
            # 如果模型加载失败，返回原始排序的结果
            return results[:top_k] if top_k is not None else results
        
        # 准备输入对
        pairs = []
        for result in results:
            chunk = result["chunk"]
            pairs.append((query, chunk["content"]))
        
        # 使用BGE Reranker评分
        try:
            # 准备模型输入
            features = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512  # 可以根据需要调整
            )
            
            if self.use_gpu:
                features = {k: v.to("cuda") for k, v in features.items()}
            
            # 模型推理
            with torch.no_grad():
                scores = self.model(**features).logits.cpu().numpy()
            
            # 为每个结果添加重排序得分
            for i, result in enumerate(results):
                result["rerank_score"] = float(scores[i][0])  # 假设模型输出的第一个值是相关性得分
            
            # 根据重排序得分排序
            reranked_results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
            
            # 限制返回数量
            if top_k is not None:
                reranked_results = reranked_results[:top_k]
            
            print(f"成功对 {len(results)} 个结果进行重排序")
            return reranked_results
        except Exception as e:
            print(f"重排序过程中出错: {e}")
            # 出错时返回原始排序
            return results[:top_k] if top_k is not None else results

# 示例用法
if __name__ == "__main__":
    # 模拟一些搜索结果
    sample_results = [
        {"chunk": {"content": "人工智能在医疗领域的应用十分广泛。"}, "distance": 0.2},
        {"chunk": {"content": "人工智能算法可以帮助医生诊断疾病。"}, "distance": 0.3},
        {"chunk": {"content": "深度学习是人工智能的一个分支。"}, "distance": 0.4}
    ]
    
    # 初始化重排序器
    reranker = Reranker()
    
    # 重排序
    reranked = reranker.rerank("人工智能在医疗中的作用", sample_results)
    
    # 打印结果
    print("\n重排序结果:")
    for i, result in enumerate(reranked):
        print(f"结果 {i+1}, 分数: {result.get('rerank_score', 'N/A')}")
        print(f"内容: {result['chunk']['content']}") 