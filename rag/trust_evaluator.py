import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import os
import re
import requests
import json
import hashlib
from transformers import AutoTokenizer, AutoModel
import torch

class DocumentTrustEvaluator:
    """
    文档级信任度评估模块，专为核化工领域设计，包含三个评分指标：
    1. 基本PageRank评分（基于论文间引用关系）
    2. 主题PageRank评分（考虑权威机构和期刊等级）
    3. DeepSeek模型API评估内容质量
    """
    
    
    def __init__(self, 
                 deepseek_api_key: Optional[str] = None,
                 authority_keywords: Optional[List[str]] = None,
                 journal_ratings: Optional[Dict[str, float]] = None,
                 weights: Optional[Dict[str, float]] = None,
                 alpha: float = 0.35,  # 新增：配置 PageRank 的 alpha 参数
                 pagerank_alpha: Optional[float] = None,  # 新增参数
                 precomputed_scores_path: Optional[str] = None,  # 新增：预计算分数的文件路径
                 deepseek_api_url: str = "https://api.deepseek.com/v1/chat/completions",
                 use_mock_scores: bool = False):  # 新增：是否使用模拟分数
        """
        初始化文档级信任度评估器
        
        Args:
            deepseek_api_key: DeepSeek API密钥
            authority_keywords: 权威机构关键词列表
            journal_ratings: 期刊评级字典 {期刊名: 评分}
            weights: 三个评分的权重 {pagerank: 权重, topic_pagerank: 权重, model_score: 权重}
            alpha: PageRank 和主题 PageRank 的随机跳转概率（默认 0.85）
            pagerank_alpha: 与 alpha 相同，为兼容实验脚本（如果提供则覆盖 alpha 值）
            precomputed_scores_path: 预计算信任分数的 JSON 文件路径
            deepseek_api_url: DeepSeek API的URL
            use_mock_scores: 是否使用模拟分数而不是调用API
        """
        # 兼容性处理：如果提供了 pagerank_alpha，优先使用它
        if pagerank_alpha is not None:
            alpha = pagerank_alpha
        
        self.deepseek_api_key = deepseek_api_key
        self.deepseek_api_url = deepseek_api_url
        self.alpha = alpha  # 新增：存储 alpha 值
        self.use_mock_scores = use_mock_scores  # 新增：存储是否使用模拟分数
        
        # 设置核化工领域的权威机构关键词
        self.authority_keywords = authority_keywords or [
            # 核化工领域权威机构
            "国家自然科学基金", "中国科学院", "教育部", "中国原子能科学研究院", "国家核安全局",
            "中国工程物理研究院", "中国核工业集团", "中国核动力研究设计院", "国防科工委",
            "清华大学核研院", "中国辐射防护研究院", "上海应用物理研究所", "近代物理研究所",
            "兰州重离子加速器", "中国原子能科学研究院", "北京航空航天大学", "哈尔滨工程大学",
            # 国际核化工权威机构
            "International Atomic Energy Agency", "IAEA", "Nuclear Regulatory Commission", "NRC",
            "Department of Energy", "DOE", "Commissariat à l'énergie atomique", "CEA",
            "Rosatom", "Joint Research Centre", "JRC", "Oak Ridge National Laboratory", "ORNL", 
            "Los Alamos National Laboratory", "LANL", "Idaho National Laboratory", "INL", 
            "Pacific Northwest National Laboratory", "PNNL", "Argonne National Laboratory", "ANL"
        ]
        
        # 设置核化工领域的期刊评级
        self.journal_ratings = journal_ratings or {
            # 一般高影响力期刊
            "Nature": 10.0, "Science": 9.8, "Cell": 9.5,
            "Physical Review Letters": 9.5, "PNAS": 9.0,
            # 核科学与技术相关期刊
            "Journal of Nuclear Materials": 8.8, 
            "Nuclear Engineering and Design": 8.6,
            "Journal of Radioanalytical and Nuclear Chemistry": 8.5,
            "Annals of Nuclear Energy": 8.3,
            "Nuclear Science and Engineering": 8.2,
            "Progress in Nuclear Energy": 8.0,
            "Nuclear Technology": 7.9,
            "Radiation Physics and Chemistry": 7.8,
            "Journal of Radiation Research": 7.7,
            "Radiation Measurements": 7.6,
            "Nuclear Instruments and Methods": 8.9,
            "Fusion Engineering and Design": 8.0,
            "Radiation Effects and Defects in Solids": 7.4,
            "Health Physics": 7.5,
            # 中文期刊
            "核科学与工程": 7.8, "原子能科学技术": 7.7, 
            "核技术": 7.6, "辐射研究与辐射工艺学报": 7.5,
            "核动力工程": 7.5, "中国科学": 7.5, "科学通报": 7.0,
            "同位素": 7.6  # 添加同位素期刊
        }
        
        # 设置默认的权重
        self.weights = weights or {
            "pagerank": 1.0,
            "topic_pagerank": 0.0,
            "model_score": 0.0
        }
        
        # 新增：加载预计算的信任分数（如果提供了路径）
        self.precomputed_scores = {}
        if precomputed_scores_path and os.path.exists(precomputed_scores_path):
            self.load_precomputed_scores(precomputed_scores_path)
                
    # 新增：获取文档哈希值的辅助方法
    def _get_document_hash(self, text: str) -> str:
        """计算文档内容的哈希值作为唯一标识符"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _extract_references_from_markdown(self, text: str) -> List[str]:
        """
        从Markdown格式的论文文本中提取参考文献，用于构建引用网络
        
        Args:
            text: Markdown格式的论文文本
            
        Returns:
            List[str]: 提取到的参考文献标识符列表
        """
        # 尝试找到参考文献部分（一般在文档末尾）
        ref_section_patterns = [
            r"## 参考文献[\s\S]*",
            r"## References[\s\S]*",
            r"# 参考文献[\s\S]*",
            r"# References[\s\S]*",
            r"\n参考文献[\s\S]*",
            r"\nReferences[\s\S]*"
        ]
        
        ref_section = ""
        for pattern in ref_section_patterns:
            matches = re.search(pattern, text)
            if matches:
                ref_section = matches.group(0)
                break
        
        # 如果找到了参考文献部分，从中提取引用
        if ref_section:
            # 提取类似 [1] Author, Title, Journal... 格式的引用
            references = re.findall(r'\[\d+\].*?(?=\[\d+\]|\Z)', ref_section)
            # 提取引用编号
            ref_ids = [re.search(r'\[(\d+)\]', ref).group(1) for ref in references if re.search(r'\[(\d+)\]', ref)]
            return ref_ids
        
        # 如果没有找到清晰的参考文献部分，尝试直接从正文中提取引用标记
        # 例如 [1], [2,3], [4-6] 等格式
        ref_patterns = [
            r'\[(\d+)\]',              # [1]
            r'\[(\d+(?:,\s*\d+)+)\]',  # [1,2,3]
            r'\[(\d+(?:\-\d+)+)\]'     # [1-3]
        ]
        
        all_refs = []
        for pattern in ref_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # 如果是逗号分隔的列表
                if ',' in match:
                    refs = [r.strip() for r in match.split(',')]
                    all_refs.extend(refs)
                # 如果是范围表示
                elif '-' in match:
                    start, end = match.split('-')
                    refs = [str(i) for i in range(int(start), int(end)+1)]
                    all_refs.extend(refs)
                else:
                    all_refs.append(match)
        
        return list(set(all_refs))  # 去重
    
    def _extract_title_from_markdown(self, text: str) -> str:
        """从Markdown提取论文标题"""
        # 尝试找到标题（一般是文档的第一个标题）
        title_patterns = [
            r'^# (.*?)$',
            r'^## (.*?)$',
            r'^### (.*?)$'
        ]
        
        for pattern in title_patterns:
            matches = re.search(pattern, text, re.MULTILINE)
            if matches:
                return matches.group(1).strip()
        
        # 如果没找到标题格式，尝试获取第一行作为标题
        first_line = text.strip().split('\n')[0]
        if first_line and len(first_line) < 200:  # 确保不是异常长的行
            return first_line
            
        return "Unknown Title"
    
    def _extract_authority_mentions(self, text: str) -> List[str]:
        """提取文本中提到的权威机构"""
        mentioned = []
        for keyword in self.authority_keywords:
            if keyword in text:
                mentioned.append(keyword)
        return mentioned
    
    def _extract_journal_mentions(self, text: str) -> List[str]:
        """提取文本中提到的期刊"""
        mentioned = []
        for journal in self.journal_ratings.keys():
            if journal in text:
                mentioned.append(journal)
        return mentioned
    
    def _call_deepseek_api(self, paper_text: str) -> Tuple[float, Dict[str, float]]:
        """
        调用DeepSeek API评估论文内容
        
        Args:
            paper_text: 论文内容文本
            
        Returns:
            Tuple[float, Dict[str, float]]: (总分, 各维度得分详情)
        """
        # 使用模拟分数
        if self.use_mock_scores:
            print("使用模拟分数代替API调用...")
            return self._calculate_fallback_document_score(paper_text)
            
        # 检查文本长度，如果过长则直接使用备用评分方法
        
        
        if not self.deepseek_api_key:
            print("警告：未提供DeepSeek API密钥，将使用备用评分方法")
            return self._calculate_fallback_document_score(paper_text)
        
        # 构建prompt
        prompt = f"""
请作为核化工领域的专家，对以下文献进行严格评估。请使用SESA指标体系进行打分（每项1-10分），并在末尾给出一个综合评分（0-1之间的小数，保留四位小数），该评分为四项得分的平均值除以10。请严格按照以下格式输出：

Scientific Rigor（科学严谨性）：[分数]
Engineering Relevance（工程相关性）：[分数]
Safety & Compliance（安全与合规性）：[分数]
Applicability（应用前景）：[分数]
综合评分：[综合得分]

评估维度说明：

1. Scientific Rigor（科学严谨性）
● 是否采用公认的核化工原理、模型或方法？
● 是否提供了清晰、可重复的实验流程或数学推导？
● 是否引用了权威、最新的文献支持其研究立场？
评分标准：
● 1-3：理论松散，缺乏模型支持或数据来源。
● 4-6：基本合理，但实验或模拟方法不够严谨。
● 7-10：高质量的科学推理、实验或理论建模，引用得当。

2. Engineering Relevance（工程相关性）
● 是否针对实际核化工过程中的关键问题（如萃取、纯化、废料处理）？
● 是否能与工业流程或现有设备对接？
评分标准：
● 1-3：理论空谈，缺乏工程落地的可能。
● 4-6：有一定工程意义，但适用范围受限。
● 7-10：高度贴合核工业现状或具有直接工程价值。

3. Safety & Compliance（安全与合规性）
● 是否充分评估了放射性物质、废液、废渣的安全处置？
● 是否符合IAEA 或国家核安全法规？
评分标准：
● 1-3：完全忽视安全问题或违反基本规范。
● 4-6：部分考量，但未系统化处理合规性。
● 7-10：全面覆盖法规、环境和辐射安全因素。

4. Applicability（应用前景）
● 是否具备可推广性？是否提出了技术优化、成本效益等分析？
● 是否能解决当前技术瓶颈？
评分标准：
● 1-3：仅限于实验室研究，缺乏后续应用讨论。
● 4-6：理论上可行，缺乏实证或示范验证。
● 7-10：明确提出推广路径，可能引发产业改进或技术突破。

重要提示：请严格遵循评分标准，避免主观偏好。评分必须基于文献内容的客观事实，不得凭空推测。如文献内容过于缺乏某维度信息导致无法评估，该维度应从低分起评。保持严格的专业态度，确保评分的一致性和公正性。

请仔细阅读以下文献，并根据SESA指标体系提供详细的评分：

{paper_text}
"""
        
        try:
            # API请求参数
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,  # 按要求设置
                "max_tokens": 1000
            }
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.deepseek_api_key}"
            }
            
            # 发送请求
            response = requests.post(
                self.deepseek_api_url,
                headers=headers,
                json=payload
            )
            
            # 解析响应
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                
                # 解析评分
                scientific_rigor = float(re.search(r'Scientific Rigor：(\d+(?:\.\d+)?)', content).group(1)) if re.search(r'Scientific Rigor：(\d+(?:\.\d+)?)', content) else 5.0
                engineering_relevance = float(re.search(r'Engineering Relevance：(\d+(?:\.\d+)?)', content).group(1)) if re.search(r'Engineering Relevance：(\d+(?:\.\d+)?)', content) else 5.0
                safety_compliance = float(re.search(r'Safety & Compliance：(\d+(?:\.\d+)?)', content).group(1)) if re.search(r'Safety & Compliance：(\d+(?:\.\d+)?)', content) else 5.0
                applicability = float(re.search(r'Applicability：(\d+(?:\.\d+)?)', content).group(1)) if re.search(r'Applicability：(\d+(?:\.\d+)?)', content) else 5.0
                
                # 尝试直接从输出中获取综合评分
                combined_match = re.search(r'综合评分：(\d+\.\d+)', content)
                if combined_match:
                    combined_score = float(combined_match.group(1))
                else:
                    # 如果没有直接提供，则计算平均值
                    combined_score = (scientific_rigor + engineering_relevance + safety_compliance + applicability) / (4 * 10)
                
                # 确保分数在0-1范围内
                combined_score = min(max(combined_score, 0.0), 1.0)
                
                scores_detail = {
                    "scientific_rigor": scientific_rigor / 10.0,
                    "engineering_relevance": engineering_relevance / 10.0,
                    "safety_compliance": safety_compliance / 10.0,
                    "applicability": applicability / 10.0
                }
                
                return combined_score, scores_detail
            
            else:
                print(f"调用DeepSeek API失败: {result.get('error', '未知错误')}")
                return self._calculate_fallback_document_score(paper_text)
                
        except Exception as e:
            print(f"调用DeepSeek API时出错: {e}")
            return self._calculate_fallback_document_score(paper_text)
    
    def _calculate_fallback_document_score(self, document_text: str) -> Tuple[float, Dict[str, float]]:
        """
        当无法调用DeepSeek API时，使用备用方法评估文档质量
        
        Returns:
            Tuple[float, Dict[str, float]]: (总分, 各维度得分详情)
        """
        # 基础分数
        score = 0.5
        
        # 文本长度 - 假设高质量论文通常较长
        length_score = min(len(document_text) / 10000, 1.0) * 0.2
        
        # 关键词密度 - 检查权威关键词和期刊名称
        authority_mentions = self._extract_authority_mentions(document_text)
        journal_mentions = self._extract_journal_mentions(document_text)
        
        # 计算权威机构和期刊的得分
        authority_score = min(len(authority_mentions) / 3, 1.0) * 0.3
        
        journal_score = 0.0
        for journal in journal_mentions:
            if journal in self.journal_ratings:
                journal_score += self.journal_ratings[journal] / 10.0
        journal_score = min(journal_score, 1.0) * 0.3
        
        # 参考文献数量
        refs = self._extract_references_from_markdown(document_text)
        ref_score = min(len(refs) / 20, 1.0) * 0.2  # 假设高质量论文通常有较多引用
        
        # 综合得分
        total_score = score + length_score + authority_score + journal_score + ref_score
        final_score = min(total_score, 1.0)
        
        # 确保分数不为零
        if final_score < 0.1:
            final_score = 0.5  # 设置一个默认的中等分数
        
        # 模拟各维度得分
        scores_detail = {
            "scientific_rigor": final_score * 0.9,  # 略微调整权重模拟各维度
            "engineering_relevance": final_score * 0.8,
            "safety_compliance": final_score * 0.85,
            "applicability": final_score * 0.95
        }
        
        return final_score, scores_detail
    
    def calculate_document_pagerank(self, documents: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算基于论文间引用关系的PageRank得分
        
        Args:
            documents: 论文列表，每个元素应包含ID和内容
            
        Returns:
            Dict[str, float]: {document_id: pagerank_score}
        """
        # 创建有向图
        G = nx.DiGraph()
        
        # 文档ID映射
        doc_ids = {}
        for i, doc in enumerate(documents):
            doc_id = doc.get("id", str(i))
            doc_ids[i] = doc_id
            G.add_node(i)
        
        # 构建引用网络
        for i, citing_doc in enumerate(documents):
            citing_text = citing_doc.get("content", "")
            citing_refs = self._extract_references_from_markdown(citing_text)
            
            # 查找被引用的文档
            for j, cited_doc in enumerate(documents):
                if i == j:  # 跳过自引用
                    continue
                    
                # 检查是否有引用关系
                # 方法1: 通过引用编号匹配
                cited_refs = set(self._extract_references_from_markdown(cited_doc.get("content", "")))
                if any(ref in cited_refs for ref in citing_refs):
                        G.add_edge(i, j)
                        continue
                
                # 方法2: 通过标题匹配
                cited_title = self._extract_title_from_markdown(cited_doc.get("content", ""))
                if cited_title and cited_title in citing_text:
                    G.add_edge(i, j)
        
        # 如果图为空（没有引用关系），返回均匀分布
        if len(G.edges()) == 0:
            return {doc_ids[i]: 1.0/len(documents)}
            
        # 计算PageRank，使用实例的 alpha 值
        pr_scores = nx.pagerank(G, alpha=self.alpha)

        # 添加调试信息
        print(f"计算PageRank (alpha={self.alpha})：共有{len(G.edges())}条边，{len(G.nodes())}个节点")
        
        # 转换为文档ID的映射
        return {doc_ids[i]: score for i, score in pr_scores.items()}
    
    def calculate_document_topic_pagerank(self, documents: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算基于主题（权威机构和期刊）的PageRank得分
        
        Args:
            documents: 论文列表，每个元素应包含ID和内容
            
        Returns:
            Dict[str, float]: {document_id: topic_pagerank_score}
        """
        # 创建有向图
        G = nx.DiGraph()
        
        # 文档ID映射
        doc_ids = {}
        
        # 添加节点并设置节点权重
        for i, doc in enumerate(documents):
            doc_id = doc.get("id", str(i))
            doc_ids[i] = doc_id
            
            # 获取文本内容
            text = doc.get("content", "")
            
            # 初始权重为1.0
            node_weight = 1.0
            
            # 提取权威机构和期刊信息，增加权重
            authorities = self._extract_authority_mentions(text)
            journals = self._extract_journal_mentions(text)
            
            # 每个权威机构增加权重
            for auth in authorities:
                node_weight += 0.5
                
            # 每个期刊按其评级增加权重
            for journal in journals:
                if journal in self.journal_ratings:
                    node_weight += self.journal_ratings[journal] / 10.0  # 归一化期刊权重
            
            # 添加带权重的节点
            G.add_node(i, weight=node_weight)
        
        # 添加边 - 基于引用关系和共同主题
        for i, doc_i in enumerate(documents):
            text_i = doc_i.get("content", "")
            refs_i = self._extract_references_from_markdown(text_i)
            i_authorities = self._extract_authority_mentions(text_i)
            i_journals = self._extract_journal_mentions(text_i)
            
            for j, doc_j in enumerate(documents):
                if i == j:  # 跳过自引用
                    continue
                
                text_j = doc_j.get("content", "")
                
                # 共同主题计算
                j_authorities = self._extract_authority_mentions(text_j)
                j_journals = self._extract_journal_mentions(text_j)
                
                # 计算共同主题数量作为边的权重
                common_authorities = set(i_authorities) & set(j_authorities)
                common_journals = set(i_journals) & set(j_journals)
                edge_weight = len(common_authorities) + len(common_journals)
                
                # 引用关系也增加边权重
                if any(ref in text_j for ref in refs_i):
                    edge_weight += 2  # 引用关系有更高权重
                
                if edge_weight > 0:
                    G.add_edge(i, j, weight=edge_weight)
        
        # 如果图为空，根据节点权重分配得分
        if len(G.edges()) == 0:
            node_weights = {i: G.nodes[i].get('weight', 1.0) for i in G.nodes()}
            total_weight = sum(node_weights.values())
            if total_weight == 0:
                return {doc_ids[i]: 1.0/len(documents)}
            else:
                return {doc_ids[i]: node_weights[i]/total_weight for i in G.nodes()}
        
        # 计算带权重的PageRank，使用实例的 alpha 值
        personalization = {i: G.nodes[i].get('weight', 1.0) for i in G.nodes()}
        pr_scores = nx.pagerank(G, alpha=self.alpha, personalization=personalization)  # 修改：使用 self.alpha
        
        # 转换为文档ID的映射
        return {doc_ids[i]: score for i, score in pr_scores.items()}
    
    def calculate_document_model_scores(self, documents: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        使用DeepSeek API评估各文档内容质量，优先使用预计算的分数
        
        Args:
            documents: 论文列表，每个元素应包含ID和内容
            
        Returns:
            Dict[str, Dict[str, float]]: {document_id: {"score": 总分, "details": {各维度详情}}}
        """
        scores = {}
        
        for doc in documents:
            # 获取文档ID，确保ID存在且唯一
            doc_id = doc.get("id")
            if not doc_id:
                # 如果没有ID字段，使用哈希作为ID
                doc_content = doc.get("content", "")
                doc_id = self._get_document_hash(doc_content)
                # 添加ID字段到文档
                doc["id"] = doc_id
            
            text = doc.get("content", "")
            
            # 计算文档哈希以检查预计算分数
            doc_hash = self._get_document_hash(text)
            
            # 首先检查是否有预计算的分数
            if doc_hash in self.precomputed_scores:
                # 使用预计算的分数
                precomputed_data = self.precomputed_scores[doc_hash]
                scores[doc_id] = precomputed_data
                print(f"使用预计算分数: {doc_id} -> {precomputed_data}")
                continue
                
            # 如果没有预计算分数，再尝试调用API
            try:
                # 调用DeepSeek API或备用方法评估
                score, details = self._call_deepseek_api(text)
                scores[doc_id] = {
                    "score": score,
                    "details": details
                }
                print(f"计算得分: {doc_id} -> score: {score}")
                
                # 将计算结果添加到预计算数据中（内存）
                self.precomputed_scores[doc_hash] = {
                    "score": score,
                    "details": details
                }
            except Exception as e:
                print(f"评估文档 {doc_id} 时出错: {e}")
                # 使用备用评分
                fallback_score, fallback_details = self._calculate_fallback_document_score(text)
                scores[doc_id] = {
                    "score": fallback_score,
                    "details": fallback_details
                }
                print(f"使用备用评分: {doc_id} -> score: {fallback_score}")
        
        return scores
    
    def load_precomputed_scores(self, scores_path: str) -> bool:
        """
        从预计算的JSON文件中加载信任度分数
        
        Args:
            scores_path: 信任度分数JSON文件的路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            with open(scores_path, 'r', encoding='utf-8') as f:
                self.precomputed_scores = json.load(f)
            print(f"已加载 {len(self.precomputed_scores)} 个预计算的信任度分数")
            return True
        except Exception as e:
            print(f"加载预计算信任度分数时出错: {str(e)}")
            self.precomputed_scores = {}
            return False
            
    def save_precomputed_scores(self, output_path: str) -> bool:
        """
        将当前的信任度分数保存到JSON文件
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            bool: 是否成功保存
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.precomputed_scores, f, ensure_ascii=False, indent=2)
                
            print(f"已将 {len(self.precomputed_scores)} 个信任度分数保存到: {output_path}")
            return True
        except Exception as e:
            print(f"保存信任度分数时出错: {str(e)}")
            return False
    
    def _normalize_scores(self, scores_dict: Dict[str, float]) -> Dict[str, float]:
        """对分数进行 Min-Max 归一化到 [0, 1] 区间"""
        normalized_scores = {}
        score_values = list(scores_dict.values())
        
        if not score_values:
            return {}
            
        min_score = np.min(score_values)
        max_score = np.max(score_values)
        
        # 处理所有分数相同的情况
        if max_score == min_score:
            # 如果所有分数都相同，可以都设为0.5 (中性) 或根据具体值判断
            # 这里我们设为 0.5
            normalized_value = 0.5
            for doc_id in scores_dict.keys():
                normalized_scores[doc_id] = normalized_value
        else:
            range_score = max_score - min_score
            for doc_id, score in scores_dict.items():
                normalized_scores[doc_id] = (score - min_score) / range_score
                
        return normalized_scores

    def calculate_document_trust_scores(
        self,
        documents: List[Dict[str, Any]],
        precomputed_model_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        计算一批文档的综合信任度分数。

        Args:
            documents: 文档列表，每个文档是一个字典，应包含 'id' 和 'content'。
            precomputed_model_scores: 可选的预计算模型评分字典 {doc_id: score}。

        Returns:
            一个字典，键是文档ID，值是包含详细分数的字典
            e.g., {'doc_id': {'score': 0.8, 'pagerank': 0.7, 'topic_pagerank': 0.9, 'model_score': 0.8, 'weights': {...}, 'alpha': 0.85}}
        """
        # 确保每个文档都有唯一ID
        for i, doc in enumerate(documents):
            if "id" not in doc:
                doc_content = doc.get("content", "")
                doc["id"] = self._get_document_hash(doc_content)
                print(f"为文档 {i} 分配ID: {doc['id']}")
        
        # 打印所有文档ID以便调试
        doc_ids = [doc.get("id") for doc in documents]
        print(f"计算信任分数的文档ID列表: {doc_ids}")
        
        # 如果已有预计算的分数，直接返回
        if self.precomputed_scores and len(self.precomputed_scores) > 0 and not self.use_mock_scores:
            print(f"使用预计算的信任度分数 ({len(self.precomputed_scores)} 个文档)")
            # 确保使用文档ID作为键返回预计算分数，而不是哈希
            result = {}
            for doc in documents:
                doc_id = doc.get("id")
                doc_hash = self._get_document_hash(doc.get("content", ""))
                if doc_hash in self.precomputed_scores:
                    result[doc_id] = self.precomputed_scores[doc_hash]
            return result
            
        print(f"计算 {len(documents)} 个文档的信任度分数")
        
        # 步骤1: 计算原始 PageRank 得分
        print("步骤1/3: 计算PageRank分数...")
        pagerank_scores_raw = self.calculate_document_pagerank(documents)
        print(f"PageRank分数数量={len(pagerank_scores_raw)}")
        # print(f"PageRank分数键={list(pagerank_scores_raw.keys())}") # 可以注释掉，减少日志量
        
        # 步骤2: 计算原始主题PageRank得分
        print("步骤2/3: 计算主题PageRank分数...")
        topic_pagerank_scores_raw = self.calculate_document_topic_pagerank(documents)
        print(f"主题PageRank分数数量={len(topic_pagerank_scores_raw)}")
        # print(f"主题PageRank分数键={list(topic_pagerank_scores_raw.keys())}") # 可以注释掉
        
        # 步骤3: 计算原始模型评分
        print("步骤3/3: 计算模型评分...")
        model_scores_raw = {}
        if precomputed_model_scores is not None and len(precomputed_model_scores) > 0:
             print(f"使用预计算的模型评分 ({len(precomputed_model_scores)} 个文档)")
             model_scores_raw = precomputed_model_scores # 假设已包含所有 doc_id
        else:
            print("未提供预计算模型评分，执行实时计算...")
            # 注意：这里的 calculate_document_model_scores 需要确保在API失败时返回合理的默认值 (如 0.5)
            # 并且其返回值结构可能需要调整或解析以得到 {doc_id: score} 的字典
            model_scores_details = self.calculate_document_model_scores(documents)
            # 从 details 中提取 score，如果失败则用 0.5
            model_scores_raw = {
                doc_id: details.get("score", 0.5)
                for doc_id, details in model_scores_details.items()
            }
            # 确保所有文档都有分数，即使计算失败
            all_doc_ids = [doc.get("id") for doc in documents]
            for doc_id in all_doc_ids:
                if doc_id not in model_scores_raw:
                    print(f"警告：文档 {doc_id} 模型评分计算失败或缺失，使用默认值 0.5")
                    model_scores_raw[doc_id] = 0.5


        print(f"模型评分数量={len(model_scores_raw)}")
        # print(f"模型评分键={list(model_scores_raw.keys())}") # 可以注释掉

        # --- 新增：归一化处理 ---
        print("对各维度分数进行 Min-Max 归一化...")
        pagerank_scores_norm = self._normalize_scores(pagerank_scores_raw)
        topic_pagerank_scores_norm = self._normalize_scores(topic_pagerank_scores_raw)
        model_scores_norm = self._normalize_scores(model_scores_raw)
        # --- 结束归一化 ---
        
        # 计算综合信任度分数
        results = {}
        for doc in documents:
            doc_id = doc.get("id")
            if not doc_id: continue # 跳过没有ID的文档 (理论上前面已处理)
            
            # 获取原始分数，用于记录
            pr_score_raw = pagerank_scores_raw.get(doc_id, 0.5)
            tpr_score_raw = topic_pagerank_scores_raw.get(doc_id, 0.5)
            ms_score_raw = model_scores_raw.get(doc_id, 0.5) # 确保这里也有默认值

            # 获取归一化后的分数，用于计算加权总分 (默认值设为0.5，因为是归一化后的中性值)
            pr_score_norm = pagerank_scores_norm.get(doc_id, 0.5)
            tpr_score_norm = topic_pagerank_scores_norm.get(doc_id, 0.5)
            ms_score_norm = model_scores_norm.get(doc_id, 0.5)

            # --- 修改：使用归一化后的分数计算加权平均分 ---
            weighted_score = (
                self.weights["pagerank"] * pr_score_norm +
                self.weights["topic_pagerank"] * tpr_score_norm +
                self.weights["model_score"] * ms_score_norm
            )
            # --- 结束修改 ---

            # 存储详细分数 (包括原始和归一化后的)
            results[doc_id] = {
                "score": weighted_score, # 这是最终的加权分数
                "pagerank": pr_score_raw,
                "topic_pagerank": tpr_score_raw,
                "model_score": ms_score_raw,
                "pagerank_norm": pr_score_norm,         # 记录归一化分数
                "topic_pagerank_norm": tpr_score_norm,  # 记录归一化分数
                "model_score_norm": ms_score_norm,      # 记录归一化分数
                "weights": self.weights.copy(),
                "alpha": self.alpha
            }
            
            # 打印时可以区分原始和归一化分数，以及最终结果
            # print(f"处理文档ID信任分数: {doc_id}")
            # print(f"  Raw Scores: PR={pr_score_raw:.4f}, TPR={tpr_score_raw:.4f}, MS={ms_score_raw:.4f}")
            # print(f"  Norm Scores: PR={pr_score_norm:.4f}, TPR={tpr_score_norm:.4f}, MS={ms_score_norm:.4f}")
            # print(f"  最终信任分数 (加权归一化): {weighted_score:.4f}")

        print(f"信任分数计算完成，共 {len(results)} 个分数")
        
        # 同时保存基于内容哈希和ID的分数（双份保存）
        for doc in documents:
            doc_id = doc.get("id")
            doc_content = doc.get("content", "")
            doc_hash = self._get_document_hash(doc_content)
            
            if doc_id in results:
                self.precomputed_scores[doc_hash] = results[doc_id]
                self.precomputed_scores[doc_id] = results[doc_id]
        
        return results
    
    def enrich_documents_with_trust_scores(
        self,
        documents: List[Dict[str, Any]],
        precomputed_model_scores: Optional[Dict[str, float]] = None # 新增参数
    ) -> List[Dict[str, Any]]:
        """
        计算文档级信任度分数并将结果添加到文档的元数据中

        Args:
            documents: 文档列表
            precomputed_model_scores: 可选的预计算模型评分字典 {doc_id: score}

        Returns:
            List[Dict[str, Any]]: 添加了信任度分数的文档列表
        """
        # 确保每个文档都有唯一ID
        for i, doc in enumerate(documents):
            if "id" not in doc:
                doc_content = doc.get("content", "")
                doc["id"] = self._get_document_hash(doc_content)
                print(f"在enrich前为文档 {i} 分配ID: {doc['id']}")
        
        # 计算信任度分数，传递预计算的模型分数
        trust_scores = self.calculate_document_trust_scores(
            documents,
            precomputed_model_scores=precomputed_model_scores # 传递参数
        )
        
        # 打印信任分数摘要以便调试
        print(f"计算得到的信任分数总数: {len(trust_scores)}")
        if len(trust_scores) > 0:
            sample_scores = list(trust_scores.items())[:3]
            print(f"样例分数: {sample_scores}")
        
        # 将分数添加到元数据中
        enriched_documents = []
        for doc in documents:
            doc_copy = doc.copy()
            doc_id = doc.get("id")
            
            # 确保metadata存在
            if "metadata" not in doc_copy:
                doc_copy["metadata"] = {}
            else:
                doc_copy["metadata"] = doc_copy["metadata"].copy()
            
            # 添加信任度分数
            if doc_id in trust_scores:
                doc_copy["metadata"]["trust_scores"] = trust_scores[doc_id]
                # 同时在文档根级别添加信任分数（双保险）
                doc_copy["trust_scores"] = trust_scores[doc_id]
                print(f"为文档 {doc_id} 添加信任分数: {trust_scores[doc_id]}")
            else:
                # 即使找不到对应ID，也添加一个默认分数
                default_score = {"score": 0.75, "pagerank": 0.5, "topic_pagerank": 0.8, "model_score": 0.9, "note": "default_score"}
                doc_copy["metadata"]["trust_scores"] = default_score
                doc_copy["trust_scores"] = default_score
                print(f"警告: 文档 {doc_id} 没有找到对应信任分数，使用默认值")
            
            enriched_documents.append(doc_copy)
        
        return enriched_documents

# 示例用法
if __name__ == "__main__":
    from document_loader import DocumentLoader
    
    # 1. 加载文档
    loader = DocumentLoader()
    all_documents = loader.load_documents() 
    
    # 2. 进行文档级信任度评估
    evaluator = DocumentTrustEvaluator(
        deepseek_api_key="sk-c6c279dbfe08466588189cb7e1fcb33b",  # 实际使用时替换为真实的API密钥
        use_mock_scores=True  # 使用模拟分数代替API调用
    )
    
    # 3. 对文档进行信任度评分并添加到元数据
    enriched_documents = evaluator.enrich_documents_with_trust_scores(all_documents)
    
    # 4. 分割文档并将文档级信任度分数传递给每个块
    all_enriched_chunks = []
    
    for doc in enriched_documents:
        # 提取文档级信任度分数
        document_trust_score = doc.get("metadata", {}).get("trust_scores", {})
        
        # 分割文档成块
        chunks = loader.split_documents([doc])
        
        # 将文档级信任度分数添加到每个块的元数据中
        for chunk in chunks:
            chunk_copy = chunk.copy()
            if "metadata" not in chunk_copy:
                chunk_copy["metadata"] = {}
            # 添加文档级信任分数
            chunk_copy["metadata"]["document_trust_scores"] = document_trust_score
            # 同时也添加到trust_scores字段，保持兼容性
            chunk_copy["metadata"]["trust_scores"] = document_trust_score
            all_enriched_chunks.append(chunk_copy)
    
    # 5. 输出结果
    print(f"总共处理了 {len(enriched_documents)} 个文档，生成了 {len(all_enriched_chunks)} 个文本块。")
    for i, doc in enumerate(enriched_documents[:2]):
        print(f"文档 {i}:")
        print(f"  信任度分数: {doc.get('metadata', {}).get('trust_scores', {})}")
        print("---")
    
    print("\n文本块示例:")
    for i, chunk in enumerate(all_enriched_chunks[:3]):
        print(f"块 {i}:")
        print(f"  元数据: {chunk.get('metadata', {})}")
        print("---") 