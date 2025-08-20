import requests
import json
from typing import List, Dict, Any, Optional, Union

class DeepSeekAPI:
    """
    DeepSeek API 客户端类，用于与DeepSeek API进行交互。
    提供方便的方法来使用DeepSeek的大语言模型进行文本生成和评估。
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        """
        初始化DeepSeek API客户端
        
        Args:
            api_key: DeepSeek API密钥
            base_url: API基础URL，默认为官方API地址
        """
        self.api_key = api_key
        self.base_url = base_url
        
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "deepseek-chat", 
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        调用DeepSeek聊天补全API
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "你好"}]
            model: 使用的模型，默认为"deepseek-chat"
            temperature: 采样温度，控制输出的随机性
            max_tokens: 生成的最大token数量
            top_p: 核采样参数
            frequency_penalty: 频率惩罚系数
            presence_penalty: 存在惩罚系数
            stop: 停止生成的标志，可以是字符串或字符串列表
            
        Returns:
            Dict[str, Any]: API响应
        """
        endpoint = f"{self.base_url}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }
        
        if stop:
            payload["stop"] = stop
        
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload
            )
            
            response.raise_for_status()  # 如果HTTP错误，抛出异常
            
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API请求失败: {e}")
            # 返回一个错误响应结构，方便调用者处理
            return {
                "error": {
                    "message": f"API请求失败: {str(e)}",
                    "type": "api_error",
                    "code": getattr(e.response, 'status_code', 500) if hasattr(e, 'response') else 500
                },
                "choices": [{"message": {"content": "API请求失败，请检查网络或密钥。"}}]
            }
    
    def evaluate_text(
        self, 
        text: str, 
        criteria: List[str] = ["质量", "相关性", "准确性", "完整性"],
        scale: int = 10
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        评估文本质量，使用DeepSeek模型进行打分
        
        Args:
            text: 待评估的文本
            criteria: 评估标准列表
            scale: 评分尺度
            
        Returns:
            Dict: 包含总分和各维度分数的字典
        """
        criteria_str = "、".join(criteria)
        prompt = f"""
请作为一个专业评估者，对以下文本进行严格评估。
请在{criteria_str}等方面进行评估，每项0-{scale}分，并给出综合得分。
请直接以JSON格式输出评分结果，不要有其他解释，格式如下：
{{
  "总分": 分数,
  "详细评分": {{
    "标准1": 分数,
    "标准2": 分数,
    ...
  }}
}}

待评估文本:
{text}
"""
        
        try:
            response = self.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # 使用低温度以获得更确定性的评估
                max_tokens=500
            )
            
            # 提取JSON响应
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # 尝试解析JSON
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # 如果JSON解析失败，返回一个默认值
                return {
                    "总分": scale / 2,
                    "详细评分": {c: scale / 2 for c in criteria}
                }
                
        except Exception as e:
            print(f"评估文本时出错: {e}")
            # 返回默认评分
            return {
                "总分": scale / 2,
                "详细评分": {c: scale / 2 for c in criteria}
            }