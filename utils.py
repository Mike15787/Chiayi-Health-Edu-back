import requests
import logging
import os
import httpx
from google import genai
from google.genai import types

# --- 全局配置 ---
# 切換這裡來改變使用的 LLM: "ollama" 或 "gemini"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama") 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_DEFAULT = "gemini-2.5-flash" 

logger = logging.getLogger(__name__)

VLLM_API_URL = os.getenv("VLLM_API_URL", "http://192.168.1.100:8001/v1/chat/completions")



async def generate_llm_response(prompt: str, model_name: str = "gemma3:4b") -> str:
    """
    統一的 LLM 入口函數。
    """
    # 根據環境變數決定路由
    if LLM_PROVIDER.lower() == "gemini":
        return await _gemini_generate(prompt, GEMINI_MODEL_DEFAULT)
    else:
        return await _ollama_generate(prompt, model_name)

async def _ollama_generate(prompt: str, model_name: str) -> str:
    """
    本地 Ollama 生成邏輯
    """
    try:
        timeout = 120 if "12b" in model_name or "summary" in prompt.lower() else 60
        ollama_payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "top_p": 0.9}
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/generate", 
                json=ollama_payload, 
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '抱歉，我無法回應您的問題。')
            else:
                logger.error(f"Ollama API 錯誤: {response.status_code}")
                return "抱歉，AI 服務暫時無法使用。"
            
    except Exception as e:
        logger.error(f"Ollama LLM 生成錯誤: {e}")
        return "抱歉，處理您的請求時發生錯誤。"

async def _gemini_generate(prompt: str, model_name: str) -> str:
    """
    Google Gemini 線上 API 生成邏輯 (使用新版 google-genai SDK)
    """
    try:

        logger.info(f"正在呼叫 Gemini API ({model_name}) [New SDK]...")
        
        # 初始化 Client (使用環境變數中的 Key)
        client = genai.Client()
        
        # 設定配置
        config = types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.9,
        )

        # 非同步呼叫
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config
        )
        
        return response.text

    except Exception as e:
        logger.error(f"Gemini API 生成錯誤: {e}")
        return "抱歉，線上 AI 服務暫時無法使用。"
    
async def vllm_generate_llm_response(prompt: str, model_name: str = "google/gemma-3-4b-it") -> str:
    """
    Generates a response from the vLLM OpenAI-compatible server.
    """
    # 使用 httpx.AsyncClient 來發送異步請求
    async with httpx.AsyncClient() as client:
        try:
            # vLLM 的超時可以設得稍微寬鬆一些
            timeout = 120.0

            # OpenAI 相容的 payload 格式
            vllm_payload = {
                "model": model_name, # 這裡的 model name 需與 vLLM 伺服器啟動時一致
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "top_p": 0.9,
                "max_tokens": 1024 # 建議設定一個最大生成長度，防止無限生成
            }
            
            response = await client.post(
                VLLM_API_URL, 
                json=vllm_payload, 
                timeout=timeout
            )
            
            # 檢查 HTTP 狀態碼
            response.raise_for_status() # 如果狀態碼不是 2xx，會拋出異常
            
            result = response.json()
            # 解析 OpenAI 格式的回應
            content = result['choices'][0]['message']['content']
            return content.strip()
            
        except httpx.ConnectError:
            logger.error(f"無法連接到 vLLM API ({VLLM_API_URL})，請確認 vLLM 服務是否運行且網路可達")
            return "抱歉，AI 服務未啟動。"
        except httpx.TimeoutException:
            logger.error(f"vLLM API 超時 ({timeout}s)")
            return "抱歉，回應時間過長，請稍後再試。"
        except httpx.HTTPStatusError as e:
            logger.error(f"vLLM API 錯誤: {e.response.status_code} - {e.response.text}")
            return "抱歉，AI 服務暫時無法使用。"
        except (KeyError, IndexError) as e:
            logger.error(f"解析 vLLM 回應錯誤: {e} - 回應內容: {response.text}")
            return "抱歉，解析 AI 回應時發生錯誤。"
        except Exception as e:
            logger.error(f"LLM 生成錯誤: {e}")
            return "抱歉，處理您的請求時發生錯誤。"