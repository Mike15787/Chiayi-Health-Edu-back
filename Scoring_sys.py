from fastapi import APIRouter, Request, HTTPException, Depends, Query
import requests
import logging
import asyncio
from pydantic import BaseModel
from databases import get_db
from sqlalchemy.orm import Session

#scoring system
#主要功能是針對 在一輪對話結束之後 對這個對話session 以及評分要點打分
#並且需要打分的結果 以及 對每個錯誤地方的講評
#打分的結果會存到 ???資料表 錯誤地方講評會放在 answer資料表

#所以整個程式就是會在 前端chatview 按下結束按鈕後 給予指令到後端 後端就開始計算評分結果
#評分結束之後 再將結果回傳給前端

#目前的評分總分是32分


# 配置日誌
logger = logging.getLogger(__name__)

# 創建路由器
score_router = APIRouter(prefix="/score", tags=["ScoringSystem"])

class ScoreRequest(BaseModel):
    text: str
    session_id: str

class ScoreResponse(BaseModel):
    access_token: str
    token_type: str
    score: int

'''檢閱藥歷
能詢問患者是否有使用此類藥物經驗(+2分)
知道如何查閱患者目前用藥(+4分)

醫療面談
向病人問好(+1分)
請病人坐下(+1分)
確認是否本人使用(+1分)
適切發問及引導以獲得正確且足夠的訊息(+3分)

諮商衛教
說明藥物開立目的(+0.5分)
有註記服藥日期及相關時間點(+1分)
說明藥物使用時機及方式(+1分)
說明水分補充方式及清腸理想狀態(+1分)
說明開始作用時間及平均時數(+0.5分)
確認是否進行無痛麻醉(+1分)
能說明禁水時間(+1分)
有詢問是否有使用降血糖或防栓塞藥物(+0.5分)
能做簡易飲食衛教(+0.5分)

組織效率
按優先順序處置(衛教內容依順序執行)(+6分)

臨床判斷
能依病人狀況判斷服藥時間(+1.5分)
能依病人狀況判斷禁水時間(+1.5分)
能依檢查時間判斷早上服藥時間點(+1.5分)
能判斷病人是否有”特殊藥物”(降血糖/防栓塞/便祕藥物)(依案例而定)(+1分)
能依病人狀況正確說明”特殊藥物”是否需停藥或續用(依案例而定)(+1.5分)'''

# --- 工具函數 ---
def build_scoring_prompt(user_text: str, history: str, agent_settings: AgentSettings = None) -> str:
    """構建評分的 prompt，包含 agent 設定"""
    
    # 基礎系統角色設定
    base_system_prompt = (
        "你是一個之後要進行大腸鏡檢查的病患 而使用者會扮演醫生的角色進行衛教 告訴你在檢查的前幾天需要做什麼或注意甚麼\n"
        "請保持以下特點：\n"
        "用繁體中文回應\n"
        "覺得醫生講得模糊的時候可以提問 其他時候只需單純應答\n"
        "回應簡潔有力，不超過20字\n"
        "不要重複醫生的問題，直接給出回應\n"
        "你是沒有醫療背景的病患 所以不懂太多醫學知識\n"
    )
    
    # 如果有 agent 設定，加入個人化資訊
    if agent_settings:
        personal_info = f"""
你的個人資訊：
- 性別：{agent_settings['gender']}
- 年齡：{agent_settings['age']}
- 疾病狀況：{agent_settings['disease']}
- 目前用藥：{agent_settings['med_info']}
- 特殊狀況：{agent_settings['special_status']}
- 檢查日期：{agent_settings['check_day']}
- 檢查時間：{agent_settings['check_time']}
- 檢查類型：{agent_settings['check_type']}
- 低渣食物購買：{agent_settings['low_cost_med']}

請根據你的個人狀況來回應使用者的問題，展現出符合你背景的關切和疑問。
"""
        system_prompt = base_system_prompt + personal_info
    else:
        system_prompt = base_system_prompt
    
    # 組合完整 prompt
    if history.strip():
        prompt = (
            f"{system_prompt}\n\n"
            f"{history}\n\n"
            f"使用者：{user_text}\n"
            f"你："
        )
    else:
        prompt = (
            f"{system_prompt}\n\n"
            f"使用者：{user_text}\n"
            f"你："
        )
    
    return prompt

async def generate_llm_response(prompt: str) -> str:
    """生成 LLM 回應"""
    try:
        ollama_payload = {
            "model": "gemma3:4b-it-q4_K_M",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 500
            }
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate", 
            json=ollama_payload, 
            timeout=30
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
        logger.error("Ollama API 超時")
        return "抱歉，回應時間過長，請稍後再試。"
    except Exception as e:
        logger.error(f"LLM 生成錯誤: {e}")
        return "抱歉，處理您的請求時發生錯誤。"

# --- API 端點 ---
@auth_router.post("/sys_scoring", response_model=ScoreResponse)
async def score(request: ScoreRequest, db: Session = Depends(get_db)):
    #先要有能夠連線LLM的function
    
    prompt = base_system_prompt = (
        "請根\n"
        "請保持以下特點：\n"
        "用繁體中文回應\n"
        "覺得醫生講得模糊的時候可以提問 其他時候只需單純應答\n"
        "回應簡潔有力，不超過20字\n"
        "不要重複醫生的問題，直接給出回應\n"
        "你是沒有醫療背景的病患 所以不懂太多醫學知識\n"
    )
    
    llm_response = await generate_llm_response(prompt)