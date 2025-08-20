import os
import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import glob # 新增导入

# 导入自定义模块
from .document_loader import DocumentLoader
from .trust_evaluator import DocumentTrustEvaluator 
from .vector_store import VectorStore
from .reranker import Reranker
from .hybrid_retriever import HybridRetriever

# 设置目录常量
OUTPUT_DIR = r"./output"  # magic-pdf 临时输出目录
PAGES_DIR = r"./pages"    # PDF 存储目录
MARKDOWN_BASE_DIR = r"./markdown_base" # 最终 Markdown 存储目录
DATA_DIR = r"./data"      # 索引数据存储目录

# 新增：配置加载函数
def load_config():
    """
    加载RAG系统的默认配置
    
    Returns:
        Dict[str, Any]: 包含系统配置的字典
    """
    return {
        "embedding_model": "./models/bge-m3",
        "reranker_model": "./models/bge-reranker-v2-m3",
        "index_path": "./data/faiss_index",
        "markdown_base_dir": MARKDOWN_BASE_DIR,
        "similarity_weight": 0.7,
        "trust_weight": 0.3,
        "use_gpu": True,
        "chunk_size": 1000,
        "chunk_overlap": 200
    }

# FastAPI模型
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranker: bool = True
    use_trust_score: bool = True
    initial_fetch: int = 20

class IndexRequest(BaseModel):
    markdown_dir: str = MARKDOWN_BASE_DIR
    chunk_size: int = 1500
    chunk_overlap: int = 300
    save_index: bool = True
    index_path: str = "./data/faiss_index"

class UploadFileResponse(BaseModel):
    status: str
    file_path: str
    message: str

class SearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float  # 这个通常是最终的混合或重排分数
    similarity: Optional[float] = None # 归一化后的相似度分数
    trust_score: Optional[float] = None # 用于混合计算的单个信任度分数
    trust_scores: Optional[Dict[str, Any]] = None # 详细信任度分数 (允许混合类型)

class QueryResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int

# 新增模型：知识库文件信息
class KBFile(BaseModel):
    filename: str
    # 可以添加其他元数据，如大小、修改时间等

# 新增模型：知识库文件列表响应
class KBListResponse(BaseModel):
    files: List[KBFile]
    count: int

# 新增响应模型，包含LLM生成的答案
class TopSourceInfo(BaseModel):
    source: str
    similarity: Optional[float] = None
    trust_score: Optional[float] = None

class CompleteQueryResponse(BaseModel):
    query: str
    answer: str  # LLM生成的回答
    sources: List[SearchResult]  # 用于生成回答的来源文档
    total_sources: int
    top_sources: Optional[List[TopSourceInfo]] = None  # 新增字段

# 新增：RAG API响应模型
class RagApiResponse(BaseModel):
    context: str
    response: str

# 初始化组件
class RAGSystem:
    def __init__(self, 
                 embedding_model_path: str = "./models/bge-m3",
                 reranker_model_path: str = "./models/bge-reranker-v2-m3",
                 index_path: str = "./data/faiss_index",
                 similarity_weight: float = 0.6,
                 trust_weight: float = 0.4,
                 use_gpu: bool = True):
        """
        初始化RAG系统
        
        Args:
            embedding_model_path: 嵌入模型路径
            reranker_model_path: 重排序模型路径
            index_path: FAISS索引路径
            similarity_weight: 相似度权重
            trust_weight: 信任度权重
            use_gpu: 是否使用GPU
        """
        print("初始化RAG系统...")
        
        # 创建所有必要的目录
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(PAGES_DIR, exist_ok=True)
        os.makedirs(MARKDOWN_BASE_DIR, exist_ok=True)
        
        # 初始化向量存储
        self.vector_store = VectorStore(
            embedding_model_path=embedding_model_path,
            index_path=index_path,
            use_gpu=use_gpu
        )
        
         # --- 修改：直接硬编码 API Key (请注意安全风险) ---
        # 推荐使用环境变量或配置文件来管理密钥
        api_key = "" # <-- 请将这里替换为你的真实密钥
        if api_key == "":
            # logger.warning("API Key 使用了占位符，请替换为真实密钥。DeepSeek评分可能失败或使用备用方法。") # 如果使用了 logger
            print("警告：API Key 使用了占位符，请替换为真实密钥。DeepSeek评分可能失败或使用备用方法。")
            # 也可以考虑在这里设置为 None，如果不想在没有真实密钥时尝试调用API
            # api_key = None
            
        # --- 结束修改 ---
        
        # 初始化信任度评估器 - 使用 DocumentTrustEvaluator 并传入 API Key
        self.trust_evaluator = DocumentTrustEvaluator(
            deepseek_api_key=api_key
            # 如果需要自定义权威列表或期刊评分，在这里传入
            # authority_keywords=..., 
            # journal_ratings=...
        )
        
        # 初始化重排序器
        self.reranker = Reranker(
            model_path=reranker_model_path,
            use_gpu=use_gpu
        )
        
        # 初始化混合检索器
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            trust_evaluator=self.trust_evaluator,# 确保传入的是新实例
            reranker=self.reranker,
            similarity_weight=similarity_weight,
            trust_weight=trust_weight
        )
        
        print("RAG系统初始化完成")
    
    def build_index(self, request: IndexRequest) -> Dict[str, Any]:
        """
        构建或重建完整的 RAG 索引
        
        Args:
            request: 索引请求参数
            
        Returns:
            Dict[str, Any]: 索引操作结果
        """
        print(f"开始重建索引，来源目录: {request.markdown_dir}")
        try:
            # 使用 retriever 中的便捷方法，它会处理加载、分割、评分和索引
            num_chunks = self.retriever.load_and_index_documents(
                markdown_dir=request.markdown_dir,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                save_index=request.save_index,
                index_path=request.index_path
            )
            
            return {
                "status": "success",
                "indexed_chunks": num_chunks,
                "message": f"成功重建索引，共处理 {num_chunks} 个文本块"
            }
        except Exception as e:
            print(f"重建索引时出错: {e}")
            return {
                "status": "error",
                "message": f"索引重建失败: {e}"
            }
    
    async def process_pdf_file(self, file: UploadFile) -> Dict[str, Any]:
        """
        处理上传的PDF文件（转换并保存）
        
        Args:
            file: 上传的PDF文件
            
        Returns:
            Dict[str, Any]: 处理结果 (只包含保存状态和路径)
        """
        filename = file.filename
        file_ext = os.path.splitext(filename)[1].lower()
        filename_without_ext = os.path.splitext(filename)[0]
        
        if file_ext != ".pdf":
            return {"status": "error", "file_path": "", "message": "不是PDF文件"}
            
        # 设置路径
        pdf_save_path = os.path.join(PAGES_DIR, filename)
        generated_md_expected_parent_dir = os.path.join(OUTPUT_DIR, filename_without_ext, "auto")
        generated_md_expected_path = os.path.join(generated_md_expected_parent_dir, f"{filename_without_ext}.md")
        markdown_final_dest_path = os.path.join(MARKDOWN_BASE_DIR, f"{filename_without_ext}.md")
        temp_output_dir_for_pdf = os.path.join(OUTPUT_DIR, filename_without_ext)
        
        try:
            # 1. 保存PDF文件
            with open(pdf_save_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            print(f"PDF文件已保存到: {pdf_save_path}")
            
            # 2. 构建并执行magic-pdf命令
            abs_pdf_path = os.path.abspath(pdf_save_path)
            abs_output_dir = os.path.abspath(OUTPUT_DIR)
            magic_pdf_cmd = f'magic-pdf -p "{abs_pdf_path}" -o "{abs_output_dir}" -m auto'
            print(f"执行命令: {magic_pdf_cmd}")
            
            try:
                process = subprocess.run(
                    magic_pdf_cmd, shell=True, check=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, encoding='utf-8'
                )
                print("magic-pdf命令执行成功")
                if process.stdout: print("标准输出:", process.stdout)
                if process.stderr: print("标准错误:", process.stderr)
                
            except subprocess.CalledProcessError as e:
                stdout = e.stdout.decode('utf-8', errors='ignore') if e.stdout else ""
                stderr = e.stderr.decode('utf-8', errors='ignore') if e.stderr else ""
                print(f"magic-pdf命令执行失败: {e}")
                print("标准输出:", stdout)
                print("标准错误:", stderr)
                if os.path.exists(pdf_save_path):
                    try: os.remove(pdf_save_path)
                    except Exception as rm_err: print(f"警告：清理失败的PDF文件时出错: {rm_err}")
                return {"status": "error", "file_path": "", "message": f"PDF转换失败: {stderr or stdout or str(e)}"}
            except Exception as e:
                print(f"执行magic-pdf时发生未知错误: {e}")
                return {"status": "error", "file_path": "", "message": f"执行PDF转换命令时出错: {e}"}
                
            # 3. 检查生成的Markdown文件是否存在
            print(f"检查Markdown文件是否存在于: {generated_md_expected_path}")
            if not os.path.exists(generated_md_expected_path):
                fallback_md_files = list(Path(temp_output_dir_for_pdf).glob("*.md"))
                if fallback_md_files:
                    generated_md_expected_path = str(fallback_md_files[0])
                    print(f"在预期路径未找到，但在 {temp_output_dir_for_pdf} 找到备选Markdown: {generated_md_expected_path}")
                else:
                    print(f"错误：在 {generated_md_expected_path} 或 {temp_output_dir_for_pdf}/*.md 未找到生成的Markdown文件。")
                    # 列出输出目录内容帮助调试
                    if os.path.exists(temp_output_dir_for_pdf):
                        print(f"目录 {temp_output_dir_for_pdf} 内容: {os.listdir(temp_output_dir_for_pdf)}")
                    if os.path.exists(generated_md_expected_parent_dir):
                        print(f"目录 {generated_md_expected_parent_dir} 内容: {os.listdir(generated_md_expected_parent_dir)}")
                        
                    return {"status": "error", "file_path": "", "message": "PDF转换成功，但未能找到生成的Markdown文件。请检查magic-pdf的输出结构。"}
                    
            # 4. 移动Markdown文件到markdown_base目录
            try:
                print(f"移动 {generated_md_expected_path} 到 {markdown_final_dest_path}")
                shutil.move(generated_md_expected_path, markdown_final_dest_path)
                print(f"Markdown文件已成功移动到: {markdown_final_dest_path}")
            except Exception as move_err:
                print(f"移动Markdown文件失败: {move_err}")
                return {"status": "error", "file_path": "", "message": f"移动生成的Markdown文件时出错: {move_err}"}
                
            # 成功处理PDF
            return {"status": "success", "file_path": markdown_final_dest_path, "message": f"PDF处理成功，生成的Markdown已保存到: {markdown_final_dest_path}"}
            
        except Exception as e:
            print(f"处理PDF过程中发生意外错误: {e}")
            return {"status": "error", "file_path": "", "message": f"处理PDF文件时发生内部错误: {e}"}
        finally:
            # 确保关闭文件流，即使失败也要关闭
            try:
                await file.close()
            except Exception: pass
    
    async def process_markdown_file(self, file: UploadFile) -> Dict[str, Any]:
        """
        处理上传的Markdown文件（直接保存）
        
        Args:
            file: 上传的Markdown文件
            
        Returns:
            Dict[str, Any]: 处理结果 (只包含保存状态和路径)
        """
        filename = file.filename
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext != ".md":
            return {"status": "error", "file_path": "", "message": "不是Markdown文件"}
            
        markdown_dest_path = os.path.join(MARKDOWN_BASE_DIR, filename)
        
        try:
            with open(markdown_dest_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            print(f"Markdown文件已直接保存到: {markdown_dest_path}")
            
            return {"status": "success", "file_path": markdown_dest_path, "message": f"Markdown文件已成功保存到: {markdown_dest_path}"}
        except Exception as e:
            print(f"保存Markdown文件失败: {e}")
            return {"status": "error", "file_path": "", "message": f"保存Markdown文件时发生错误: {e}"}
        finally:
            # 确保关闭文件流
            try:
                await file.close()
            except Exception: pass
    
    def search(self, request: QueryRequest) -> QueryResponse:
        """
        搜索查询
        
        Args:
            request: 查询请求
            
        Returns:
            QueryResponse: 查询响应
        """
        # 步骤1：向量检索+混合评分（相似度+信任度）
        if request.use_trust_score and self.retriever.trust_evaluator:
            results = self.retriever.hybrid_retrieve(
                query=request.query,
                top_k=request.top_k * 2,  # 获取更多结果用于重排序
                initial_fetch=30
            )
        else:
            # 仅使用向量检索
            results = self.retriever.retrieve(
                query=request.query,
                top_k=request.top_k * 2,
                initial_fetch=30
            )
        
        # 步骤2：重排序（如果启用）
        if request.use_reranker and self.retriever.reranker:
            results = self.retriever.rerank(
                query=request.query,
                results=results,
                top_k=request.top_k
            )
        else:
            # 不使用重排序，直接截取top_k
            results = results[:request.top_k]
        
        # 步骤3：格式化检索结果
        formatted_results = []
        for result in results:
            # 改进：更健壮地获取元数据，兼容不同层级
            chunk_data = result.get("chunk", {})
            metadata = chunk_data.get("metadata", result.get("metadata", {})) # 获取元数据，优先从chunk内取

            # 提取得分 - 优先使用重排序分数 ('score')，其次混合分数 ('hybrid_score')，最后是距离反算
            if "score" in result: # reranker 通常会添加顶层 score 键
                final_score = result["score"]
            elif "hybrid_score" in result:
                final_score = result["hybrid_score"]
            elif "distance" in result:
                # 确保 distance 是数值
                try:
                    final_score = 1.0 / (1.0 + float(result["distance"]))
                except (ValueError, TypeError):
                    final_score = 0.0 # 如果 distance 无效
            else:
                final_score = 0.0 # 没有可用的分数来源

            # 获取其他评分细节 (使用 .get() 避免 KeyError)
            similarity_score = result.get("normalized_similarity") # 归一化相似度
            single_trust_score = result.get("trust_score") # 用于计算的单个信任分数
            # 之前在 _apply_hybrid_scoring 中将详细字典存入了 metadata['trust_scores_details']
            detailed_trust_scores_dict = metadata.get("trust_scores_details")

            # 创建结果对象，使用正确的字段映射
            search_result = SearchResult(
                content=chunk_data.get("content", result.get("content", "内容未找到")), # 获取内容
                metadata=metadata,                     # 使用提取的元数据
                score=final_score,                     # 使用最终用于排序的分数
                similarity=similarity_score,           # 使用归一化相似度
                trust_score=single_trust_score,        # 使用计算混合分时用的单个信任分
                trust_scores=detailed_trust_scores_dict # 使用完整的信任分数详情字典
            )

            formatted_results.append(search_result)
        
        # 添加在构建context后、生成prompt前
        print("引用的前5个检索结果:")
        for i, res in enumerate(formatted_results[:5]):
            print(f"{i+1}. 文件: {res.metadata.get('source', '未知')} (相似度: {res.similarity:.4f}, 信任度: {res.trust_score:.4f})")
        
        # 创建响应
        response = QueryResponse(
            query=request.query,
            results=formatted_results,
            total_results=len(formatted_results)
        )
        
        return response

    def list_kb_files(self) -> KBListResponse:
        """
        列出 markdown_base 目录中的所有 Markdown 文件
        """
        try:
            md_files = glob.glob(os.path.join(MARKDOWN_BASE_DIR, "*.md"))
            kb_files = [
                KBFile(filename=os.path.basename(f))
                for f in md_files
            ]
            return KBListResponse(files=kb_files, count=len(kb_files))
        except Exception as e:
            print(f"列出知识库文件时出错: {e}")
            # 返回空列表，或者可以抛出异常由API处理
            return KBListResponse(files=[], count=0)

# 创建FastAPI应用
app = FastAPI(
    title="核化工学术论文RAG检索",
    description="接收PDF/Markdown文件，进行转换存储，并提供基于信任度的RAG检索功能",
    version="1.1.0"
)
# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建RAG系统实例
rag_system = None

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化RAG系统"""
    global rag_system
    rag_system = RAGSystem()

@app.post("/upload", response_model=UploadFileResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    上传并处理PDF或Markdown文件。
    此接口只负责文件转换和保存，**不进行自动索引**。
    请在上传一批文件后，手动调用 /index 接口进行索引。
    """
    global rag_system
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG系统正在初始化或未就绪")
    
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext == ".pdf":
        result = await rag_system.process_pdf_file(file)
    elif file_ext == ".md":
        result = await rag_system.process_markdown_file(file)
    else:
        raise HTTPException(status_code=400, detail="无效的文件类型，请上传.pdf或.md文件")
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
        
    return result # 只返回保存结果

@app.post("/index", response_model=Dict[str, Any])
async def trigger_index_build(request: IndexRequest):
    """
    触发对指定目录下所有Markdown文件的全量索引重建。
    这是一个耗时操作，会加载所有文件，计算信任度，并重建向量数据库。
    """
    global rag_system
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG系统正在初始化或未就绪")
    
    # 注意：这是一个同步调用，在大量文件时可能导致请求超时
    # 在生产环境中，建议将其改为后台任务（如使用 Celery）
    result = rag_system.build_index(request)
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
        
    return result

@app.post("/search", response_model=CompleteQueryResponse)
async def search_documents(request: QueryRequest):
    """
    使用当前的索引进行文档检索并生成回答。
    """
    global rag_system
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG系统正在初始化或未就绪")
    
    # 步骤1：向量检索+混合评分（与原代码相同）
    if request.use_trust_score and rag_system.retriever.trust_evaluator:
        results = rag_system.retriever.hybrid_retrieve(
            query=request.query,
            top_k=request.top_k * 2,
            initial_fetch=30
        )
    else:
        results = rag_system.retriever.retrieve(
            query=request.query,
            top_k=request.top_k * 2,
            initial_fetch=30
        )
    
    # 步骤2：重排序（与原代码相同）
    if request.use_reranker and rag_system.retriever.reranker:
        results = rag_system.retriever.rerank(
            query=request.query,
            results=results,
            top_k=request.top_k
        )
    else:
        results = results[:request.top_k]
    
    # 步骤3：格式化检索结果
    formatted_results = []
    for result in results:
        # 改进：更健壮地获取元数据，兼容不同层级
        chunk_data = result.get("chunk", {})
        metadata = chunk_data.get("metadata", result.get("metadata", {})) # 获取元数据，优先从chunk内取

        # 提取得分 - 优先使用重排序分数 ('score')，其次混合分数 ('hybrid_score')，最后是距离反算
        if "score" in result: # reranker 通常会添加顶层 score 键
            final_score = result["score"]
        elif "hybrid_score" in result:
            final_score = result["hybrid_score"]
        elif "distance" in result:
            # 确保 distance 是数值
            try:
                final_score = 1.0 / (1.0 + float(result["distance"]))
            except (ValueError, TypeError):
                final_score = 0.0 # 如果 distance 无效
        else:
            final_score = 0.0 # 没有可用的分数来源

        # 获取其他评分细节 (使用 .get() 避免 KeyError)
        similarity_score = result.get("normalized_similarity") # 归一化相似度
        single_trust_score = result.get("trust_score") # 用于计算的单个信任分数
        # 之前在 _apply_hybrid_scoring 中将详细字典存入了 metadata['trust_scores_details']
        detailed_trust_scores_dict = metadata.get("trust_scores_details")

        # 创建结果对象，使用正确的字段映射
        search_result = SearchResult(
            content=chunk_data.get("content", result.get("content", "内容未找到")), # 获取内容
            metadata=metadata,                     # 使用提取的元数据
            score=final_score,                     # 使用最终用于排序的分数
            similarity=similarity_score,           # 使用归一化相似度
            trust_score=single_trust_score,        # 使用计算混合分时用的单个信任分
            trust_scores=detailed_trust_scores_dict # 使用完整的信任分数详情字典
        )

        formatted_results.append(search_result)
    
    # 步骤4：构建提示模板并调用DeepSeek API生成回答
    if formatted_results:
        context_parts = []
        print(f"检索到 {len(formatted_results)} 个结果，将格式化并加入信任度分数到 Context 中。")
        
        # 在这里添加计算平均信任度分数的代码
        total_trust_score = 0.0
        count = min(5, len(formatted_results))
        
        for i, res in enumerate(formatted_results[:count]):
            trust_score = res.trust_score if hasattr(res, "trust_score") and res.trust_score is not None else 0.0
            total_trust_score += trust_score
        
        avg_trust_score = total_trust_score / count
        print(f"\n前{count}个文档块的平均信任度分数: {avg_trust_score:.4f}\n")
        
        # 然后继续原有的代码
        for i, res in enumerate(formatted_results):
            chunk_content = res.content if hasattr(res, "content") else "内容缺失"
            trust_score_data = res.metadata.get("trust_scores", {}) if hasattr(res, "metadata") else {}
            # 获取归一化的最终信任分数，如果不存在则给个默认值或标记
            trust_score = res.trust_score if hasattr(res, "trust_score") and res.trust_score is not None else 0.0
            
            # 格式化每个块，包含索引和信任度分数
            formatted_chunk = f"--- 文档块 {i+1} (信任度: {trust_score:.4f}) ---\n{chunk_content}"
            context_parts.append(formatted_chunk)
        
        # 将所有格式化后的块用换行符合并成最终的 context 字符串
        context = "\n\n".join(context_parts)
        # print(f"构建的 Context:\n{context[:500]}...") # 调试时可以取消注释查看
    else:
        print("未检索到任何相关文档块。")
        context = "" # 如果没有结果，context 为空

    # 步骤5：构建提示模板并调用DeepSeek API生成回答
    if context:
        # 当有相关文档时，使用包含信任度标注的 Context 和增强的指令
        prompt = f"""请基于以下文档回答用户的问题。文档按照其信任度分数进行了标注（分数越高越可信）。
如果文档中有足够信息，请直接使用文档内容回答。
**重要：如果不同文档块之间存在冲突或矛盾的信息，请优先采纳信任度分数更高的文档块中的信息和观点，并在回答中适当说明或体现这种侧重。**
如果文档中没有足够的信息，您可以使用自身的知识来回答，但必须明确指出您使用了自身知识而非文档内容。
请不要凭空捏造信息，始终清晰区分哪些信息来自文档，哪些来自您自身的知识库。

**请使用 Markdown 格式组织您的回答**，例如使用标题、列表、粗体等，以提高可读性。

文档内容:
{context}

用户问题: {request.query}

回答:"""
    else:
        # 当没有找到相关文档时，使用之前的无文档提示
        prompt = f"""用户提出了以下问题，但我们没有找到相关的文档内容。
请尝试使用您自身的知识来回答问题，并明确说明这是基于您的知识而非文档内容。
如果您不确定或无法回答，请诚实地表明。

**请使用 Markdown 格式组织您的回答**，例如使用标题、列表、粗体等，以提高可读性。

用户问题: {request.query}

回答:"""
    
    try:
        # # 在调用Ollama API前添加
        # print(f"发送给Ollama DeepSeek-r1:7b的提示词：\n{prompt[:300]}...（仅显示前300字符）")
        # 使用本地Ollama服务
        response = await call_ollama_api(prompt, "deepseek-r1:7b")
        answer = response.get("answer", "无法生成回答，请稍后再试。")
    except Exception as e:
        print(f"调用Ollama API失败: {e}")
        answer = f"生成回答时出错: {str(e)}"
    
    # 步骤6：返回包含生成回答的响应
    top_sources = []
    for res in formatted_results[:5]:
        top_sources.append(TopSourceInfo(
            source=res.metadata.get('source', '未知'),
            similarity=getattr(res, 'similarity', 0.0) if hasattr(res, 'similarity') else 0.0,
            trust_score=getattr(res, 'trust_score', 0.0) if hasattr(res, 'trust_score') else 0.0
        ))

    return CompleteQueryResponse(
        query=request.query,
        answer=answer,
        sources=formatted_results,
        total_sources=len(formatted_results),
        top_sources=top_sources  # 添加新信息
    )

@app.get("/kb/list", response_model=KBListResponse)
async def list_knowledge_base_files():
    """
    获取当前知识库 (`markdown_base` 目录) 中的文件列表。
    """
    global rag_system
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG系统正在初始化或未就绪")
    
    return rag_system.list_kb_files()

@app.get("/health")
async def health_check():
    """健康检查，返回服务状态和目录存在情况"""
    global rag_system
    initialized = rag_system is not None
    vector_store_ready = initialized and rag_system.vector_store is not None and rag_system.vector_store.index is not None
    index_size = rag_system.vector_store.index.ntotal if vector_store_ready else 0
        
    return {
        "status": "online" if initialized else "initializing", 
        "system_initialized": initialized,
        "vector_store_ready": vector_store_ready,
        "indexed_chunks": index_size,
        "directories": {
            "markdown_base": os.path.exists(MARKDOWN_BASE_DIR),
            "pages": os.path.exists(PAGES_DIR),
            "output": os.path.exists(OUTPUT_DIR),
            "data": os.path.exists(DATA_DIR)
        }
    }

@app.post("/api/rag")
async def rag_api(request: QueryRequest):
    """
    针对批量评估的简化RAG API接口。
    提供与search相同的检索和生成功能，但返回更简化的结构。
    """
    global rag_system
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG系统正在初始化或未就绪")
    
    # 步骤1：向量检索+混合评分（与search接口相同）
    if request.use_trust_score and rag_system.retriever.trust_evaluator:
        results = rag_system.retriever.hybrid_retrieve(
            query=request.query,
            top_k=request.top_k * 2,
            initial_fetch=30
        )
    else:
        results = rag_system.retriever.retrieve(
            query=request.query,
            top_k=request.top_k * 2,
            initial_fetch=30
        )
    
    # 步骤2：重排序
    if request.use_reranker and rag_system.retriever.reranker:
        results = rag_system.retriever.rerank(
            query=request.query,
            results=results,
            top_k=request.top_k
        )
    else:
        results = results[:request.top_k]
    
    # 步骤3：格式化检索结果
    formatted_results = []
    for result in results:
        chunk_data = result.get("chunk", {})
        metadata = chunk_data.get("metadata", result.get("metadata", {}))
        
        # 提取分数
        if "score" in result:
            final_score = result["score"]
        elif "hybrid_score" in result:
            final_score = result["hybrid_score"]
        elif "distance" in result:
            try:
                final_score = 1.0 / (1.0 + float(result["distance"]))
            except (ValueError, TypeError):
                final_score = 0.0
        else:
            final_score = 0.0
            
        similarity_score = result.get("normalized_similarity")
        single_trust_score = result.get("trust_score")
        
        # 创建格式化的结果
        content = chunk_data.get("content", result.get("content", "内容未找到"))
        
        # 带有信任度信息的格式化结果
        formatted_result = {
            "content": content,
            "metadata": metadata,
            "score": final_score,
            "similarity": similarity_score,
            "trust_score": single_trust_score
        }
        formatted_results.append(formatted_result)
    
    # 步骤4：构建context
    if formatted_results:
        context_parts = []
        for i, res in enumerate(formatted_results):
            chunk_content = res["content"]
            trust_score = res["trust_score"] if res["trust_score"] is not None else 0.0
            formatted_chunk = f"--- 文档块 {i+1} (信任度: {trust_score:.4f}) ---\n{chunk_content}"
            context_parts.append(formatted_chunk)
        
        context = "\n\n".join(context_parts)
    else:
        context = ""
    
    # 步骤5：构建提示模板并生成回答
    if context:
        prompt = f"""请基于以下文档回答用户的问题。文档按照其信任度分数进行了标注（分数越高越可信）。
如果文档中有足够信息，请直接使用文档内容回答。
**重要：如果不同文档块之间存在冲突或矛盾的信息，请优先采纳信任度分数更高的文档块中的信息和观点，并在回答中适当说明或体现这种侧重。**
如果文档中没有足够的信息，您可以使用自身的知识来回答，但必须明确指出您使用了自身知识而非文档内容。
请不要凭空捏造信息，始终清晰区分哪些信息来自文档，哪些来自您自身的知识库。

**请使用 Markdown 格式组织您的回答**，例如使用标题、列表、粗体等，以提高可读性。

文档内容:
{context}

用户问题: {request.query}

回答:"""
    else:
        prompt = f"""用户提出了以下问题，但我们没有找到相关的文档内容。
请尝试使用您自身的知识来回答问题，并明确说明这是基于您的知识而非文档内容。
如果您不确定或无法回答，请诚实地表明。

**请使用 Markdown 格式组织您的回答**，例如使用标题、列表、粗体等，以提高可读性。

用户问题: {request.query}

回答:"""
    
    try:
        print(f"发送给Ollama DeepSeek-r1:7b的提示词：\n{prompt[:300]}...（仅显示前300字符）")
        response = await call_ollama_api(prompt, "deepseek-r1:7b")
        answer = response.get("answer", "无法生成回答，请稍后再试。")
    except Exception as e:
        print(f"调用Ollama API失败: {e}")
        answer = f"生成回答时出错: {str(e)}"
    
    # 返回简化的响应结构
    return RagApiResponse(
        context=context,
        response=answer
    )

# 辅助函数：调用DeepSeek API生成答案
async def call_ollama_api(prompt: str, model_id: str = "deepseek-r1:7b"):
    """调用本地Ollama API生成回答"""
    import aiohttp
    
    url = "http://localhost:11434/api/generate"  # Ollama默认API地址
    payload = {
        "model": model_id,  # 使用本地的deepseek-r1:7b模型
        "prompt": prompt,
        "stream": False,
        "temperature": 0.1,  # 低温度使回答更加确定性
        "max_tokens": 1000
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API请求失败: {response.status}, {error_text}")
            
            response_data = await response.json()
            answer = response_data.get("response", "")
            return {"answer": answer, "full_response": response_data}

def main():
    """启动FastAPI应用"""
    parser = argparse.ArgumentParser(description="学术论文RAG检索系统")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--reload", action="store_true", help="是否启用热重载（开发模式）")
    
    args = parser.parse_args()
    
    uvicorn.run("rag.app:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main() 