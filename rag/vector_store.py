import os
import numpy as np
import faiss
import pickle
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import torch
import time

class VectorStore:
    """
    向量存储模块，使用FAISS来存储和检索向量化的文本块
    """
    
    def __init__(self, 
                 embedding_model_path: str = "./models/bge-m3",
                 index_path: Optional[str] = None,
                 dimension: int = 1024,  # BGE-M3的向量维度
                 use_gpu: bool = True):
        """
        初始化向量存储
        
        Args:
            embedding_model_path: 本地嵌入模型路径
            index_path: FAISS索引保存路径，如果提供，则尝试加载现有索引
            dimension: 向量维度，默认为1024（BGE-M3）
            use_gpu: 是否使用GPU
        """
        self.embedding_model_path = embedding_model_path
        self.index_path = index_path
        self.dimension = dimension
        # 检查 FAISS 是否有 GPU 支持，并结合用户设置
        self.has_gpu_support = hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu")
        self.use_gpu = use_gpu and torch.cuda.is_available() and self.has_gpu_support
        
        # 创建FAISS索引
        self.index = None
        self.chunk_ids = []  # 存储文本块ID，与索引位置对应
        self.id_to_chunk = {}  # 存储文本块数据，通过ID访问
        
        # 延迟加载嵌入模型
        self.embedding_model = None
        
        # 检查索引文件(.faiss 和 .pkl)是否存在
        faiss_file_exists = index_path and os.path.exists(f"{index_path}.faiss")
        pkl_file_exists = index_path and os.path.exists(f"{index_path}.pkl")
        
        # 如果两个文件都存在，则尝试加载索引
        if faiss_file_exists and pkl_file_exists:
            try:
                self.load_index(index_path)
                print(f"成功加载FAISS索引和元数据: {index_path}.faiss, {index_path}.pkl")
            except Exception as e:
                print(f"加载FAISS索引失败: {e}")
                print("将创建新的索引")
                self._create_new_index()
        else:
            # 如果文件不全或未提供路径，则创建新索引
            if index_path:
                 print(f"未找到完整的索引文件 ({index_path}.faiss 和 {index_path}.pkl)，将创建新的索引")
            self._create_new_index()
    
    def _create_new_index(self):
        """创建新的FAISS索引"""
        # 使用L2距离的IndexFlatL2
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # 如果有GPU支持，并且设置使用GPU
        if self.use_gpu:
            try:
                # 将索引转移到GPU
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                print("成功将FAISS索引移动到GPU")
            except Exception as e:
                print(f"移动FAISS索引到GPU失败: {e}")
                print("将使用CPU版本的FAISS")
                self.use_gpu = False # 明确禁用GPU，避免后续保存出错
        
        self.chunk_ids = []
        self.id_to_chunk = {}
    
    def _load_embedding_model(self):
        """延迟加载嵌入模型"""
        if self.embedding_model is None:
            try:
                print(f"加载嵌入模型: {self.embedding_model_path}")
                self.embedding_model = SentenceTransformer(self.embedding_model_path)
                
                # 如果可用且设置了使用GPU，将模型移到GPU
                if self.use_gpu:
                    self.embedding_model = self.embedding_model.to(torch.device("cuda"))
                    print("成功将嵌入模型移动到GPU")
                
                print("嵌入模型加载成功")
            except Exception as e:
                print(f"加载嵌入模型失败: {e}")
                raise ValueError(f"无法加载嵌入模型: {e}")
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表编码为向量
        
        Args:
            texts: 要编码的文本列表
            
        Returns:
            np.ndarray: 文本向量数组，形状为 (len(texts), self.dimension)
        """
        # 确保嵌入模型已加载
        self._load_embedding_model()
        
        # 编码文本
        try:
            # 使用批处理，避免OOM问题
            batch_size = 32  # 可以根据GPU内存调整
            all_embeddings = []
            
            start_time = time.time()
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch_texts, 
                    show_progress_bar=True if i == 0 else False,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # L2归一化，对于余弦相似度很重要
                )
                all_embeddings.append(batch_embeddings)
                
                if i == 0:
                    first_batch_time = time.time() - start_time
                    estimated_total_time = first_batch_time * (len(texts) / batch_size)
                    print(f"完成第一批编码，预计总时间: {estimated_total_time:.2f}秒")
            
            embeddings = np.vstack(all_embeddings)
            print(f"编码完成，总耗时: {time.time() - start_time:.2f}秒")
            return embeddings
        except Exception as e:
            print(f"编码文本时出错: {e}")
            raise
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        将文本块添加到索引中
        
        Args:
            chunks: 要添加的文本块列表 [{"content": "...", "metadata": {...}}, ...]
        """
        if not chunks:
            print("没有要添加的文本块")
            return
        
        # 提取文本内容
        texts = [chunk["content"] for chunk in chunks]
        
        # 编码文本
        try:
            embeddings = self.encode_texts(texts)
        except Exception as e:
            print(f"编码文本块失败: {e}")
            return
        
        # 确保索引存在
        if self.index is None:
            self._create_new_index()
        
        # 将向量添加到索引
        try:
            # 获取当前索引大小作为新块的起始ID
            start_id = len(self.chunk_ids)
            
            # 为每个块分配ID并存储
            for i, chunk in enumerate(chunks):
                chunk_id = start_id + i
                self.chunk_ids.append(chunk_id)
                self.id_to_chunk[chunk_id] = chunk
            
            # 添加向量到FAISS索引
            self.index.add(embeddings)
            print(f"成功将 {len(chunks)} 个文本块添加到索引，当前索引大小: {len(self.chunk_ids)}")
        except Exception as e:
            print(f"将文本块添加到索引时出错: {e}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        基于查询文本搜索最相似的文本块
        
        Args:
            query: 查询文本
            top_k: 返回的结果数量
            
        Returns:
            List[Dict[str, Any]]: 相似文本块列表，每个包含距离和原始块信息
        """
        # 确保索引已创建并包含数据
        if self.index is None or len(self.chunk_ids) == 0:
            print("索引为空或未初始化")
            return []
        
        # 编码查询文本
        try:
            query_embedding = self.encode_texts([query])
        except Exception as e:
            print(f"编码查询文本失败: {e}")
            return []
        
        # 使用FAISS搜索
        try:
            # 限制结果数量不超过索引大小
            k = min(top_k, len(self.chunk_ids))
            distances, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, idx in enumerate(indices[0]):  # indices是二维数组，第一维是查询数量
                if idx < 0 or idx >= len(self.chunk_ids):  # 确保索引有效
                    continue
                    
                chunk_id = self.chunk_ids[idx]
                chunk = self.id_to_chunk.get(chunk_id)
                
                if chunk:
                    # 距离越小表示越相似（L2距离）
                    distance = distances[0][i]
                    
                    # 将结果打包为字典
                    result = {
                        "chunk": chunk,
                        "distance": float(distance),
                        "index": int(idx)
                    }
                    results.append(result)
            
            return results
        except Exception as e:
            print(f"搜索时出错: {e}")
            return []
    
    def save_index(self, path: Optional[str] = None) -> bool:
        """
        保存FAISS索引和元数据
        
        Args:
            path: 保存路径，如果不提供，则使用初始化时的path
            
        Returns:
            bool: 保存是否成功
        """
        # 使用提供的路径或默认路径
        save_path = path or self.index_path
        if not save_path:
            print("未提供保存路径")
            return False
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            index_to_save = self.index
            # 如果索引在GPU上 (并且库支持相关函数)，首先复制回CPU
            if self.use_gpu and hasattr(faiss, "index_gpu_to_cpu"):
                try:
                    index_to_save = faiss.index_gpu_to_cpu(self.index)
                except Exception as gpu_err:
                    print(f"将索引从GPU复制到CPU时出错: {gpu_err}，尝试直接保存CPU索引")
                    # 如果复制失败，可能意味着虽然 use_gpu=True 但实际在CPU上，尝试直接保存
                    # 或者安装的是不完整的GPU库
            elif not hasattr(faiss, "StandardGpuResources"):
                 print("已安装FAISS CPU版本，直接保存CPU索引")

            # 保存索引
            if index_to_save is not None:
                 faiss.write_index(index_to_save, f"{save_path}.faiss")
            else:
                 print("错误：索引对象为空，无法保存")
                 return False
            
            # 保存元数据
            metadata = {
                "chunk_ids": self.chunk_ids,
                "id_to_chunk": self.id_to_chunk,
                "dimension": self.dimension
            }
            with open(f"{save_path}.pkl", "wb") as f:
                pickle.dump(metadata, f)
            
            print(f"成功保存索引到 {save_path}")
            return True
        except Exception as e:
            print(f"保存索引时出错: {e}")
            return False
    
    def load_index(self, path: str) -> bool:
        """
        加载FAISS索引和元数据
        
        Args:
            path: 索引路径（不包含扩展名）
            
        Returns:
            bool: 加载是否成功
        """
        try:
            # 加载索引
            index_path = f"{path}.faiss"
            metadata_path = f"{path}.pkl"
            
            if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
                print(f"无法找到索引文件或元数据文件: {path}")
                return False
            
            # 加载FAISS索引
            self.index = faiss.read_index(index_path)
            
            # 如果设置使用GPU，并且库支持GPU，将索引移到GPU
            if self.use_gpu and self.has_gpu_support: 
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    print("成功将加载的索引移动到GPU")
                except Exception as e:
                    print(f"将索引移动到GPU失败: {e}")
                    self.use_gpu = False # 加载后移动失败，禁用GPU模式
            elif self.use_gpu and not self.has_gpu_support:
                 print("警告：请求使用GPU但安装的FAISS库不支持，索引将在CPU上运行")
                 self.use_gpu = False # 明确禁用GPU

            # 加载元数据
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            
            self.chunk_ids = metadata["chunk_ids"]
            self.id_to_chunk = metadata["id_to_chunk"]
            self.dimension = metadata["dimension"]
            
            print(f"成功加载索引，包含 {len(self.chunk_ids)} 个文本块")
            return True
        except Exception as e:
            print(f"加载索引时出错: {e}")
            return False

# 示例用法
if __name__ == "__main__":
    from document_loader import DocumentLoader
    
    # 加载和分割文档
    loader = DocumentLoader()
    documents = loader.load_documents()
    chunks = loader.split_documents(documents)
    
    print(f"加载了 {len(documents)} 个文档，生成了 {len(chunks)} 个文本块")
    
    # 创建向量存储
    vector_store = VectorStore(embedding_model_path="./models/bge-m3", 
                               index_path="./data/faiss_index")
    
    # 添加文本块到索引
    vector_store.add_chunks(chunks)
    
    # 保存索引
    vector_store.save_index()
    
    # 示例查询
    if chunks:
        results = vector_store.search("人工智能的应用", top_k=3)
        print("\n示例查询结果:")
        for i, result in enumerate(results):
            print(f"结果 {i+1}, 距离: {result['distance']}")
            print(f"内容预览: {result['chunk']['content'][:100]}...") 