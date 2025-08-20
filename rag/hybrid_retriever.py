from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .vector_store import VectorStore
from .trust_evaluator import DocumentTrustEvaluator
from .reranker import Reranker
import logging

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    混合检索器，负责以下功能：
    1. 向量检索 - 基于语义相似度查找相关文本块
    2. 混合评分 - 结合语义相似度和文档信任度
    3. 重排序 - 使用重排序模型对结果进行精细排序
    
    这些功能可以单独或组合使用，形成完整的检索增强管道
    """
    
    def __init__(self, 
                 vector_store: VectorStore,
                 trust_evaluator: Optional[DocumentTrustEvaluator] = None,
                 reranker: Optional[Reranker] = None,
                 similarity_weight: float = 0.7,
                 trust_weight: float = 0.3):
        """
        初始化混合检索器
        
        Args:
            vector_store: 向量存储实例
            trust_evaluator: 信任度评估器实例（可选）
            reranker: 重排序器实例（可选）
            similarity_weight: 相似度得分的权重
            trust_weight: 信任度得分的权重
        """
        if vector_store is None:
            raise ValueError("vector_store不能为None")
            
        if similarity_weight < 0 or trust_weight < 0:
            raise ValueError("权重不能为负值")
            
        self.vector_store = vector_store
        self.trust_evaluator = trust_evaluator
        self.reranker = reranker
        
        # 当没有信任度评估器时，调整权重
        if trust_evaluator is None and trust_weight > 0:
            logger.warning("未提供信任度评估器，信任度权重将被忽略")
            similarity_weight = 1.0
            trust_weight = 0.0
        
        # 权重
        self.similarity_weight = similarity_weight
        self.trust_weight = trust_weight
        
        # 归一化权重
        total_weight = similarity_weight + trust_weight
        if total_weight != 1.0:
            self.similarity_weight /= total_weight
            self.trust_weight /= total_weight
    
    def retrieve(self, 
                 query: str, 
                 top_k: int = 5, 
                 initial_fetch: int = 20) -> List[Dict[str, Any]]:
        """
        基本检索方法 - 仅执行向量检索，不应用混合评分或重排序
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            initial_fetch: 初始获取的相似度结果数量（可大于top_k以便后续处理）
            
        Returns:
            List[Dict[str, Any]]: 基础检索结果列表
        """
        if top_k <= 0 or initial_fetch <= 0 or initial_fetch < top_k:
            raise ValueError("top_k和initial_fetch必须为正数，且initial_fetch应大于等于top_k")
        
        # 执行向量检索
        results = self.vector_store.search(query, top_k=initial_fetch)
        
        if not results:
            return []
            
        return results[:top_k]
    
    def hybrid_retrieve(self, 
                       query: str, 
                       top_k: int = 5, 
                       initial_fetch: int = 20) -> List[Dict[str, Any]]:
        """
        混合检索方法 - 执行向量检索并应用混合评分（相似度+信任度）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            initial_fetch: 初始获取的相似度结果数量
            
        Returns:
            List[Dict[str, Any]]: 应用了混合评分的结果列表
        """
        if top_k <= 0 or initial_fetch <= 0 or initial_fetch < top_k:
            raise ValueError("top_k和initial_fetch必须为正数，且initial_fetch应大于等于top_k")
        
        # 1. 相似度搜索 - 获取初始结果集
        initial_results = self.vector_store.search(query, top_k=initial_fetch)
        
        if not initial_results:
            return []
            
        # 2. 应用混合评分（如果有信任度评估器）
        if self.trust_evaluator and self.trust_weight > 0:
            initial_results = self._apply_hybrid_scoring(initial_results)
        
        # 3. 返回前top_k个结果
        return initial_results[:top_k]
    
    def rerank(self, 
              query: str, 
              results: List[Dict[str, Any]], 
              top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        重排序方法 - 对检索结果进行重排序
        
        Args:
            query: 查询文本
            results: 需要重排序的结果列表
            top_k: 返回结果数量，如果为None则返回所有重排序结果
            
        Returns:
            List[Dict[str, Any]]: 重排序后的结果列表
        """
        if not results:
            return []
            
        if not self.reranker:
            logger.warning("未提供重排序器，返回原始结果")
            return results[:top_k] if top_k is not None else results
        
        # 执行重排序
        reranked_results = self.reranker.rerank(query, results)
        
        # 返回结果
        if top_k is not None:
            return reranked_results[:min(top_k, len(reranked_results))]
        return reranked_results
    
    def full_retrieve(self, 
                     query: str, 
                     top_k: int = 5, 
                     initial_fetch: int = 20,
                     apply_hybrid_scoring: bool = True,
                     apply_reranking: bool = True) -> List[Dict[str, Any]]:
        """
        完整检索流程 - 按顺序执行向量检索、混合评分和重排序
        
        Args:
            query: 查询文本
            top_k: 最终返回的结果数量
            initial_fetch: 初始获取的相似度结果数量
            apply_hybrid_scoring: 是否应用混合评分
            apply_reranking: 是否应用重排序
            
        Returns:
            List[Dict[str, Any]]: 最终检索结果
        """
        if top_k <= 0 or initial_fetch <= 0 or initial_fetch < top_k:
            raise ValueError("top_k和initial_fetch必须为正数，且initial_fetch应大于等于top_k")
        
        # 1. 相似度搜索 - 获取初始结果集
        initial_results = self.vector_store.search(query, top_k=initial_fetch)
        
        if not initial_results:
            return []
        
        # 2. 混合评分 (如果启用且有信任度评估器)
        if apply_hybrid_scoring and self.trust_evaluator and self.trust_weight > 0:
            initial_results = self._apply_hybrid_scoring(initial_results)
        
        # 3. 重排序 (如果启用且有重排序器)
        if apply_reranking and self.reranker:
            initial_results = self.reranker.rerank(query, initial_results)
        
        # 4. 返回前top_k个结果
        return initial_results[:top_k]
    
    def _apply_hybrid_scoring(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对检索结果应用混合评分（相似度 + 信任度）

        Args:
            results: 原始检索结果列表

        Returns:
            List[Dict[str, Any]]: 添加了混合分数并排序的结果列表
        """
        import logging as logger # 保持日志记录器

        if not results:
            return []

        # 检查是否有distance字段，用于计算相似度
        # 注意：假设distance越小越好 (L2距离)
        distances = []
        valid_results = []
        for res in results:
            if "distance" in res:
                distances.append(res["distance"])
                valid_results.append(res)
            else:
                logger.warning(f"结果缺少distance字段，将被忽略: {res.get('chunk', {}).get('metadata', {}).get('source', '未知来源')}")

        if not valid_results:
            logger.warning("所有结果都缺少distance字段，无法计算相似度，跳过混合评分。")
            return [] # 返回空列表或原始列表取决于需求

        results = valid_results # 只处理包含距离的结果

        min_distance = min(distances)
        max_distance = max(distances)
        distance_range = max_distance - min_distance

        # 避免除以零
        if distance_range == 0:
            logger.warning("所有有效检索结果的距离值相同，相似度归一化可能不准确")

        # 添加混合分数
        scored_results = []

        for result in results:
            # 尝试从结果块的元数据中获取信任分数
            metadata = result.get("chunk", {}).get("metadata", {})
            # 首先尝试从 'trust_scores' 获取，然后尝试从 'document_trust_scores' 获取（为了兼容）
            trust_scores_dict = metadata.get("trust_scores")
            if not isinstance(trust_scores_dict, dict): # 如果不是字典或为None
                 trust_scores_dict = metadata.get("document_trust_scores") # 尝试备用键

            # 打印读取到的元数据信任分数（用于调试）
            print(f"--- 块 {metadata.get('chunk_id', '未知')} (来自 {metadata.get('source', '?')}) 的元数据信任分数: {trust_scores_dict} ---")

            trust_score = 0.5 # 默认信任分数
            if isinstance(trust_scores_dict, dict) and 'score' in trust_scores_dict:
                try:
                    # 确保分数是有效的数值类型
                    score_value = float(trust_scores_dict['score'])
                    # 可以在这里添加范围检查，例如确保分数在0到1之间
                    trust_score = max(0.0, min(1.0, score_value)) # Clamp between 0 and 1
                except (ValueError, TypeError) as e:
                     logger.warning(f"块 {metadata.get('chunk_id', '未知')} 的 trust_scores['score'] 不是有效数值 ({trust_scores_dict['score']})，错误: {e}，使用默认值 0.5")
                     trust_score = 0.5
            else:
                 logger.warning(f"块 {metadata.get('chunk_id', '未知')} 未找到有效的 'score' 键于 trust_scores 字典中 ({trust_scores_dict})，使用默认值 0.5")

            # 归一化相似度分数为0-1范围（1表示最相似）
            # L2距离越小越好，所以用 1 - normalized_distance
            normalized_similarity = 1.0
            if distance_range > 0:
                # 将距离归一化到0-1 (0是最好，1是最差)
                normalized_distance = (result["distance"] - min_distance) / distance_range
                normalized_similarity = 1.0 - normalized_distance
            # 如果所有距离都一样，则相似度都设为1.0（或者可以根据业务逻辑设为0.5）
            # 之前的逻辑是正确的，保持 normalized_similarity = 1.0

            # 计算混合分数
            hybrid_score = (
                normalized_similarity * self.similarity_weight +
                trust_score * self.trust_weight
            )

            # 打印计算细节（用于调试）
            print(f"--- 计算得到的 hybrid_score: {hybrid_score:.4f} (相似度: {normalized_similarity:.4f}, 信任度: {trust_score:.4f}) ---")

            # 添加分数并保存
            result_copy = result.copy()
            result_copy["normalized_similarity"] = normalized_similarity
            result_copy["trust_score"] = trust_score # 存储实际使用的信任分数
            result_copy["hybrid_score"] = hybrid_score

            # 将原始的信任度分数详情也添加到结果中，方便前端展示
            # 确保在元数据层级添加
            if "metadata" not in result_copy.get("chunk", {}):
                 if "chunk" in result_copy: result_copy["chunk"]["metadata"] = {}
                 else: result_copy["metadata"] = {} # 如果没有 chunk 结构

            target_metadata = result_copy.get("chunk", {}).get("metadata", result_copy.get("metadata", {}))
            target_metadata["trust_scores_details"] = trust_scores_dict if isinstance(trust_scores_dict, dict) else {}


            scored_results.append(result_copy)

        # 根据混合分数排序
        sorted_results = sorted(scored_results, key=lambda x: x["hybrid_score"], reverse=True)

        return sorted_results
    
    def load_and_index_documents(self, 
                                 markdown_dir: str = "./markdown_base", 
                                 chunk_size: int = 1000, 
                                 chunk_overlap: int = 200,
                                 save_index: bool = True,
                                 index_path: str = "./data/faiss_index") -> int:
        """
        加载、分割、评估和索引文档的便捷方法
        
        Args:
            markdown_dir: Markdown文件目录
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            save_index: 是否保存索引
            index_path: 索引保存路径
            
        Returns:
            int: 索引的文本块数量
        """
        from .document_loader import DocumentLoader
        import logging as logger
        
        try:
            # 1. 初始化加载器
            loader = DocumentLoader(
                markdown_dir=markdown_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            # 2. 加载所有文档
            documents = loader.load_documents()
            print(f"加载了 {len(documents)} 个文档")
            
            # 3. 确保每个文档都有稳定ID
            for i, doc in enumerate(documents):
                if "id" not in doc:
                    doc_content = doc.get("content", "")
                    doc["id"] = loader._generate_stable_id(doc_content)
                    print(f"为文档 {i} 分配ID: {doc['id']}")

            enriched_documents = documents # 初始化
            # 4. 如果有信任度评估器，计算文档级信任度分数
            if self.trust_evaluator:
                print("开始计算文档级信任度分数...")
                enriched_documents = self.trust_evaluator.enrich_documents_with_trust_scores(documents)
                
                # 输出样例分数以便调试
                if len(enriched_documents) > 0:
                    sample_doc = enriched_documents[0]
                    sample_scores = sample_doc.get("metadata", {}).get("trust_scores", {})
                    print(f"样例文档信任分数: {sample_scores}")
                
                print("文档级信任度评分完成")
            
            # 5. 遍历富集后的文档，进行分块，并将文档级信任度传递给块
            all_enriched_chunks = []
            print("开始分割文档并将信任度分数传递给文本块...")
            
            for doc in enriched_documents:
                doc_id = doc.get("id")
                print(f"处理文档ID: {doc_id}")
                
                # 提取文档级信任度分数
                document_trust_scores = doc.get("metadata", {}).get("trust_scores", {})
                # 检查信任分数是否存在
                has_trust_scores = bool(document_trust_scores)
                print(f"  是否在trust_scores中: {has_trust_scores}")
                
                if not has_trust_scores and "trust_scores" in doc:
                    document_trust_scores = doc["trust_scores"]
                    has_trust_scores = bool(document_trust_scores)
                    print(f"  从文档根级别获取trust_scores: {has_trust_scores}")
                
                if not has_trust_scores:
                    print(f"  警告：文档ID在trust_scores中未找到")
                    # 创建默认分数，避免空值问题
                    document_trust_scores = {
                        "score": 0.75,
                        "pagerank": 0.5,
                        "topic_pagerank": 0.8,
                        "model_score": 0.7,
                        "note": "default_score"
                    }
                
                # 分割当前文档成块
                try:
                    # 假设DocumentLoader可以处理单个文档的分割
                    chunks = loader.split_documents([doc]) # 使用列表包装单个文档
                except Exception as e:
                    print(f"分割文档 {doc.get('metadata', {}).get('source', '未知来源')} 时出错: {e}")
                    continue # 跳过这个文档

                # 将文档级信任度分数添加到每个块的元数据中
                for chunk in chunks:
                    chunk_copy = chunk.copy()
                    # 确保metadata存在且是可修改的字典
                    if "metadata" not in chunk_copy or not isinstance(chunk_copy["metadata"], dict):
                        chunk_copy["metadata"] = {}
                    else:
                        chunk_copy["metadata"] = chunk_copy["metadata"].copy()
                    
                    # 添加文档级信任分数到元数据的两个位置，以保持兼容性
                    # 1. trust_scores 键 - 主要位置
                    chunk_copy["metadata"]["trust_scores"] = document_trust_scores
                    # 2. document_trust_scores 键 - 确保兼容性
                    chunk_copy["metadata"]["document_trust_scores"] = document_trust_scores
                    
                    # 确保块有ID，与文档ID相关联
                    if "id" not in chunk_copy:
                        chunk_id = chunk_copy["metadata"].get("chunk_id", 0)
                        chunk_copy["id"] = f"{doc_id}_chunk_{chunk_id}"
                    
                    all_enriched_chunks.append(chunk_copy)
            
            print(f"总共生成了 {len(all_enriched_chunks)} 个带有信任度信息的文本块")

            # 6. 添加所有处理过的文本块到向量存储
            if not all_enriched_chunks:
                 print("没有生成任何文本块，无法建立索引。请检查文档内容或分割逻辑。")
                 return 0
                 
            self.vector_store.add_chunks(all_enriched_chunks)
            print(f"已将 {len(all_enriched_chunks)} 个文本块添加到向量存储")
            
            # 7. 如果需要，保存索引
            if save_index:
                self.vector_store.save_index(index_path)
                print(f"索引已保存到 {index_path}")
            
            return len(all_enriched_chunks)
            
        except FileNotFoundError as e:
            print(f"文件或目录不存在: {e}")
            raise
        except PermissionError as e:
            print(f"权限错误: {e}")
            raise
        except Exception as e:
            print(f"加载和索引文档时发生错误: {e}")
            raise

# 示例用法
if __name__ == "__main__":
    from .vector_store import VectorStore
    from .trust_evaluator import DocumentTrustEvaluator
    from .reranker import Reranker
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 初始化组件
    vector_store = VectorStore(
        embedding_model_path="./models/bge-m3",
        index_path="./data/faiss_index"
    )
    
    trust_evaluator = DocumentTrustEvaluator(
        deepseek_model_path="./models/deepseek-evaluator"
    )
    
    reranker = Reranker(
        model_path="./models/BAAI/bge-reranker-v2-m3"
    )
    
    # 初始化混合检索器
    retriever = HybridRetriever(
        vector_store=vector_store,
        trust_evaluator=trust_evaluator,
        reranker=reranker,
        similarity_weight=0.7,
        trust_weight=0.3
    )
    
    # 示例：加载和索引文档
    # retriever.load_and_index_documents()
    
    # 示例：完整检索流程
    query = "人工智能在医疗领域的应用"
    
    # 1. 先进行向量检索和混合评分
    hybrid_results = retriever.hybrid_retrieve(
        query=query,
        top_k=10,
        initial_fetch=20
    )
    
    print(f"\n查询: {query}")
    print(f"混合评分结果数量: {len(hybrid_results)}")
    for i, result in enumerate(hybrid_results[:3]):
        print(f"\n混合评分结果 {i+1}:")
        print(f"混合得分: {result.get('hybrid_score', 'N/A'):.4f} (相似度: {result.get('normalized_similarity', 'N/A'):.4f}, 信任度: {result.get('trust_score', 'N/A'):.4f})")
        print(f"内容: {result['chunk']['content'][:150]}...")
    
    # 2. 再对混合评分结果进行重排序
    reranked_results = retriever.rerank(
        query=query,
        results=hybrid_results,
        top_k=10
    )
    
    print(f"\n重排序后的结果数量: {len(reranked_results)}")
    for i, result in enumerate(reranked_results):
        print(f"\n重排序结果 {i+1}:")
        print(f"重排序得分: {result.get('rerank_score', 'N/A'):.4f}")
        print(f"内容: {result['chunk']['content'][:150]}...")
        
    # 注意：从这里开始，可以将reranked_results发送给大模型进行处理
    print("\n流程完成，下一步可以将重排序后的结果发送给大模型处理") 