import requests
import logging

logger = logging.getLogger(__name__)

async def generate_llm_response(prompt: str, model_name: str = "gemma3:4b") -> str:
    """
    Generates a response from the Ollama LLM.
    Allows specifying the model, defaulting to a smaller one for general chat.
    """
    try:
        # Use a longer timeout for more complex models/tasks
        timeout = 120 if "12b" in model_name or "summary" in prompt.lower() else 60

        ollama_payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9
            }
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate", 
            json=ollama_payload, 
            timeout=timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '抱歉，我無法回應您的問題。')
        else:
            logger.error(f"Ollama API 錯誤: {response.status_code} - {response.text}")
            return "抱歉，AI 服務暫時無法使用。"
            
    except requests.exceptions.ConnectionError:
        logger.error("無法連接到 Ollama API，請確認 Ollama 服務是否運行")
        return "抱歉，AI 服務未啟動。"
    except requests.exceptions.Timeout:
        logger.error(f"Ollama API 超時 ({timeout}s)")
        return "抱歉，回應時間過長，請稍後再試。"
    except Exception as e:
        logger.error(f"LLM 生成錯誤: {e}")
        return "抱歉，處理您的請求時發生錯誤。"