import os
import glob
import hashlib
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentLoader:
    """加载 Markdown 文件并进行文本分割的组件"""
    
    def __init__(self, markdown_dir: str = "./markdown_base", chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        初始化文档加载器
        
        Args:
            markdown_dir: Markdown 文件所在目录
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
        """
        self.markdown_dir = markdown_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
        )
    
    def _generate_stable_id(self, content: str) -> str:
        """生成基于内容的稳定ID"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _load_single_file(self, file_path: str) -> Dict[str, Any]:
        """加载单个 Markdown 文件并提取元数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 生成稳定的文档ID
            doc_id = self._generate_stable_id(content)
            
            # 提取简单的元数据（可扩展）
            metadata = {
                "source": file_path,
                "filename": os.path.basename(file_path),
                # 假设第一行是标题
                "title": content.split('\n')[0].replace('# ', '') if content.startswith('# ') else os.path.basename(file_path),
            }
            
            return {"content": content, "metadata": metadata, "id": doc_id}
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
            return {"content": "", "metadata": {"source": file_path, "error": str(e)}, "id": f"error_{os.path.basename(file_path)}"}
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """加载所有 Markdown 文件"""
        markdown_files = glob.glob(os.path.join(self.markdown_dir, "*.md"))
        documents = []
        
        for file_path in markdown_files:
            doc = self._load_single_file(file_path)
            if doc["content"]:  # 只添加成功加载的文档
                documents.append(doc)
        
        return documents
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将文档分割成较小的块"""
        chunks = []
        
        for doc in documents:
            content = doc["content"]
            metadata = doc["metadata"].copy() if "metadata" in doc else {}
            doc_id = doc.get("id", self._generate_stable_id(content))
            
            # 使用 LangChain 的文本分割器
            text_chunks = self.text_splitter.split_text(content)
            
            # 为每个块创建元数据
            for i, chunk in enumerate(text_chunks):
                chunk_metadata = metadata.copy()
                # 添加原始文档ID和块ID
                chunk_metadata["doc_id"] = doc_id
                chunk_metadata["chunk_id"] = i
                # 为块生成唯一ID
                chunk_id = f"{doc_id}_chunk_{i}"
                
                # 如果文档有信任分数，将其保留在块元数据中
                if "trust_scores" in doc:
                    chunk_metadata["trust_scores"] = doc["trust_scores"]
                if "metadata" in doc and "trust_scores" in doc["metadata"]:
                    chunk_metadata["trust_scores"] = doc["metadata"]["trust_scores"]
                    
                chunks.append({
                    "content": chunk,
                    "metadata": chunk_metadata,
                    "id": chunk_id
                })
        
        return chunks

# 示例用法
if __name__ == "__main__":
    loader = DocumentLoader()
    documents = loader.load_documents()
    chunks = loader.split_documents(documents)
    print(f"加载了 {len(documents)} 个文档，生成了 {len(chunks)} 个文本块") 