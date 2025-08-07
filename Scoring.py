from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, ValidationError, ConfigDict
from sqlalchemy.orm import Session
import logging
import re, json
import requests
from databases import (
    get_db, 
    ChatLog, 
    AgentSettings, 
    SessionUserMap, 
    Scores, 
    Summary
)

# 配置日誌
logger = logging.getLogger(__name__)

# 創建路由器
score_router = APIRouter(prefix="/scoring", tags=["Scoring"])

class FinishRequest(BaseModel):
    session_id: str
    username: str
    
class ScoresModel(BaseModel):
    total_score: str
    organization_efficient_score: str # 組織效率分數
    clinical_judgement_score: str #臨床判斷分數
    diet_edu_score: str # 飲食衛教分數
    medication_edu_score: str # 藥物衛教分數
    npo_edu_score: str #禁水衛教分數 NPO = nothing by mouth，常見術語
    special_med_handling_score: str #特殊藥物處理分數
    
    # This tells Pydantic to read data from object attributes (like scores.total_score)
    # instead of just from dictionary keys.
    model_config = ConfigDict(from_attributes=True)

class SummaryModel(BaseModel):
    total_summary: str
    organization_efficient_summary: str
    clinical_judgement_summary: str
    diet_edu_summary: str
    medication_edu_summary: str
    npo_edu_summary: str
    special_med_handling_summary: str

    model_config = ConfigDict(from_attributes=True)

class SummaryResponse(BaseModel):
    session_id: str
    agent_code: str
    username: str
    level: str  # 來自 AgentSettings
    case_info: str # 組合 AgentSettings 的資訊
    scores: ScoresModel
    summaries: SummaryModel


#這邊要做的是 首先要載入資料(從資料庫裏面取出來) 接著是列出評分項目
#先透過docs api 去輸入

#前端會傳session_id過來 並且要和username進行雙重認證
#傳過來的資料

def parse_llm_json(raw_text: str) -> dict:
    """
    從 LLM 的原始回應中提取並解析 JSON 物件。
    這種方法可以處理前後可能存在的非 JSON 文本或 Markdown 標籤，
    例如 "```json\n{...}\n```" 或 "這是您要的JSON: {...}"。
    """
    try:
        # 找到第一個 '{' 和最後一個 '}'
        start_index = raw_text.find('{')
        end_index = raw_text.rfind('}')

        # 如果找不到大括號，或順序不對，就拋出錯誤
        if start_index == -1 or end_index == -1 or end_index < start_index:
            raise ValueError("在回應中找不到有效的 JSON 物件。")

        # 提取 JSON 字串片段
        json_str = raw_text[start_index : end_index + 1]

        # 嘗試解析提取出的字串
        return json.loads(json_str)
        
    except ValueError as e:
        # 捕捉我們自己拋出的 ValueError，並附加原始文本以便除錯
        logger.error(f"無法從回應中提取 JSON。錯誤: {e}. 原始回應: {raw_text}")
        raise  # 重新拋出異常，讓上層的 try-except 捕捉
    except json.JSONDecodeError as e:
        # 捕捉解析失敗的錯誤，並附加提取出的字串以便除錯
        logger.error(f"解析提取的 JSON 字串時失敗。錯誤: {e}. 提取的字串: '{json_str}'")
        raise  # 重新拋出異常，讓上層的 try-except 捕捉

async def generate_llm_response(prompt: str) -> str:
    """生成 LLM 回應"""
    try:
        ollama_payload = {
            "model": "gemma3:12b",
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
            timeout=120
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

@score_router.post("/finish")
async def finish_and_score_session(request: FinishRequest, db: Session = Depends(get_db)):
    """
    結束對話，觸發評分和總結流程。
    """
    session_id = request.session_id
    username = request.username
    
     # 1. 驗證 Session 存在且屬於該用戶
    session_map = db.query(SessionUserMap).filter(
        SessionUserMap.session_id == session_id, 
        SessionUserMap.username == username
    ).first()
    if not session_map:
        raise HTTPException(status_code=404, detail="指定的 Session 或 Username 不存在")
    
    # 檢查 `Scores` 表中是否已存在該 session_id 的記錄
    existing_score = db.query(Scores).filter(Scores.session_id == session_id).first()
    if existing_score:
        logger.warning(f"Session {session_id} 的評分已存在，將跳過 LLM 呼叫。")
        return {"status": "already_scored", "session_id": session_id}
    
    # 2. 從資料庫撈取所需資料

    agent_settings = db.query(AgentSettings).filter(AgentSettings.agent_code == session_map.agent_code).first()
    if not agent_settings:
        raise HTTPException(status_code=404, detail="找不到此對話對應的案例設定")

    chat_history_rows = db.query(ChatLog).filter(ChatLog.session_id == session_id).order_by(ChatLog.time.asc()).all()
    if not chat_history_rows:
        raise HTTPException(status_code=404, detail="找不到此對話的聊天紀錄")

    # 3. 格式化對話紀錄
    formatted_history = ""
    for entry in chat_history_rows:
        role_name = "User(醫生)" if entry.role == 'user' else "Agent(病人)"
        formatted_history += f"{role_name}: {entry.text}\n"

    # 3. 準備給 LLM 的 Prompts (與之前相同)
    scoring_prompt = f"""
    你是一個專業的臨床指導老師，請根據以下對話紀錄和評分標準，對使用者(醫生)的表現進行評分。
    請嚴格按照下面的JSON格式輸出，不要有任何額外的文字或解釋。

    **評分標準:**
    - 組織效率 (6分): 開場、流程順暢度、結束語。
    - 臨床判斷 (7分): 能否根據病人狀況做出判斷。
    - 飲食衛教 (1.5分): 是否清楚說明飲食注意事項。
    - 藥物衛教 (6分): 是否清楚說明藥物用法。
    - 禁水衛教 (1分): 是否清楚說明NPO時間。
    - 特殊藥物處理 (1分): 是否處理特殊藥物（如抗凝血劑）。

    **病患背景資料:**
    疾病狀況: {agent_settings.disease}
    目前用藥: {agent_settings.med_info}
    特殊狀況: {agent_settings.special_status}

    **對話紀錄:**
    {formatted_history}

    **輸出格式 (JSON):**
    {{
        "total_score": "總分(純數字)",
        "organization_efficient_score": "分數",
        "clinical_judgement_score": "分數",
        "diet_edu_score": "分數",
        "medication_edu_score": "分數",
        "npo_edu_score": "分數",
        "special_med_handling_score": "分數"
    }}
    """
    summary_prompt = f"""
    你是一個專業的臨床指導老師，請根據以下對話紀錄，為使用者(醫生)的各項表現提供具體的文字評語和總結。
    請嚴格按照下面的JSON格式輸出，不要有任何額外的文字或解釋。

    **病患背景資料:**
    {agent_settings.med_info}

    **對話紀錄:**
    {formatted_history}

    **輸出格式 (JSON):**
    {{
        "total_summary": "一句話總評",
        "organization_efficient_summary": "組織效率的評語",
        "clinical_judgement_summary": "臨床判斷的評語",
        "diet_edu_summary": "飲食衛教的評語",
        "medication_edu_summary": "藥物衛教的評語",
        "npo_edu_summary": "禁水衛教的評語",
        "special_med_handling_summary": "特殊藥物處理的評語"
    }}
    """
    
    try:
        # 4. 呼叫真實的 LLM API
        logger.info(f"開始為 Session {session_id} 進行評分...")
        score_result_str = await generate_llm_response(scoring_prompt)
        
        logger.info(f"開始為 Session {session_id} 產生總結...")
        summary_result_str = await generate_llm_response(summary_prompt)
        
        score_data   = parse_llm_json(score_result_str)
        summary_data = parse_llm_json(summary_result_str)

        
        # 5. 將結果存入資料庫
        new_scores = Scores(session_id=session_id, **score_data)
        db.merge(new_scores)

        new_summary = Summary(session_id=session_id, **summary_data)
        db.merge(new_summary)

        session_map.score = score_data.get("total_score", "0")
        
        db.commit()
        logger.info(f"Session {session_id} 評分與儲存成功。")

        return {"status": "completed", "session_id": session_id}

    except json.JSONDecodeError as e:
        logger.error(f"LLM回傳的JSON格式錯誤: {e}. 回應內容: {score_result_str} 或 {summary_result_str}")
        db.rollback()
        raise HTTPException(status_code=500, detail="AI 回應的格式不正確，無法解析。")
    except HTTPException as e:
        # 如果是 generate_llm_response 拋出的 HTTP 異常，直接再次拋出
        db.rollback()
        raise e
    except Exception as e:
        logger.error(f"評分過程中發生錯誤: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"評分過程中發生未知錯誤: {e}")

@score_router.get("/summary/{session_id}", response_model=SummaryResponse)
async def get_summary_data(session_id: str, db: Session = Depends(get_db)):
    """
    獲取指定 session 的評分和總結數據。
    """
    session_map = db.query(SessionUserMap).filter(SessionUserMap.session_id == session_id).first()
    scores = db.query(Scores).filter(Scores.session_id == session_id).first()
    summaries = db.query(Summary).filter(Summary.session_id == session_id).first()
    
    if not all([session_map, scores, summaries]):
        raise HTTPException(status_code=404, detail="Summary data not found for this session. It might not be completed yet.")

    agent_settings = db.query(AgentSettings).filter(AgentSettings.agent_code == session_map.agent_code).first()
    if not agent_settings:
        raise HTTPException(status_code=404, detail="Associated agent settings not found.")

    # 組合 case_info
    case_info = f"性別: {agent_settings.gender}, 年齡: {agent_settings.age}, 主訴: {agent_settings.disease}"

    return SummaryResponse(
        session_id=session_id,
        agent_code=session_map.agent_code,
        username=session_map.username,
        level=agent_settings.med_complexity,
        case_info=case_info,
        scores=scores,      # Pass the SQLAlchemy object directly to the Pydantic model
        summaries=summaries  # Pass the SQLAlchemy object directly to the Pydantic model
    )