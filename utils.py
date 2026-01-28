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

# vLLM 提供 OpenAI 兼容的 API，路徑通常是 /v1/chat/completions
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8243/v1/chat/completions")
# 因為 vLLM server 啟動時指定了模型，API 呼叫時必須傳入完全一樣的字串
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "google/gemma-3-4b-it")


async def generate_llm_response(prompt: str, model_name: str = "gemma3:4b") -> str:
    """
    統一的 LLM 入口函數。
    """
    provider = LLM_PROVIDER.lower()
    
    # [修改處 2] 加入 vllm 的判斷
    if provider == "gemini":
        return await _gemini_generate(prompt, GEMINI_MODEL_DEFAULT)
    elif provider == "vllm":
        # 注意：這裡我們忽略傳入的 model_name (例如 gemma3:12b)，
        # 強制使用 vLLM Server 上載入的模型 (google/gemma-3-4b-it)
        return await _vllm_generate(prompt, VLLM_MODEL_NAME)
    else:
        # 預設 Ollama
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
    
# [修改處 3] 完善 vLLM 生成邏輯 (OpenAI Compatible)
async def _vllm_generate(prompt: str, model_name: str) -> str:
    """
    Generates a response from the vLLM OpenAI-compatible server.
    """
    async with httpx.AsyncClient() as client:
        try:
            logger.info(f"正在呼叫 vLLM API ({VLLM_API_URL}) 使用模型: {model_name}...")
            
            # vLLM 有時初次載入較慢，設定較長的 timeout
            timeout = 120.0

            # OpenAI 相容的 payload 格式
            vllm_payload = {
                "model": model_name, 
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "top_p": 0.9,
                "max_tokens": 1024,
                # vLLM 特有參數，如果遇到 stop token 問題可以加
                # "stop": ["<end_of_turn>", "<eos>"] 
            }
            
            response = await client.post(
                VLLM_API_URL, 
                json=vllm_payload, 
                timeout=timeout
            )
            
            response.raise_for_status()
            
            result = response.json()
            # 解析 OpenAI 格式的回應
            content = result['choices'][0]['message']['content']
            return content.strip()
            
        except httpx.ConnectError:
            logger.error(f"無法連接到 vLLM API ({VLLM_API_URL})，請確認 vLLM 服務是否運行且網路可達 (Port 8243)")
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
            logger.error(f"vLLM 生成錯誤: {e}")
            return "抱歉，處理您的請求時發生錯誤。"