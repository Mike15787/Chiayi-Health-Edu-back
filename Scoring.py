from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, ValidationError, ConfigDict
from sqlalchemy.orm import Session
import logging
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
import re, json
import requests
from utils import generate_llm_response
from datetime import datetime
from databases import (
    get_db, 
    ChatLog, 
    AnswerLog,
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

# --- 新增：載入評分標準和定義分類映射 ---
def load_scoring_criteria(file_path='scoring_criteria.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        return {item['id']: item for item in json.load(f)}

scoring_criteria_map = load_scoring_criteria()

# 將 JSON 中的 category 映射到 ScoresModel 的欄位
CATEGORY_TO_FIELD_MAP = {
    "醫療面談": "organization_efficient_score",
    "檢閱藥歷": "medication_edu_score",
    "臨床判斷": "clinical_judgement_score",
    #諮商衛教 and 臨床判斷 have sub-rules handled in a separate function
}

def map_criterion_to_field(criterion: dict) -> str:
    """根據評分項的分類和內容，決定它屬於哪個分數欄位"""
    category = criterion.get("category")
    item_text = criterion.get("item", "")
    
    field = CATEGORY_TO_FIELD_MAP.get(category)
    if field:
        return field
    
    if category == "諮商衛教":
        if "禁水" in item_text:
            return "npo_edu_score"
        if "飲食" in item_text:
            return "diet_edu_score"
        return "medication_edu_score" # Default for this category
    
    if category == "臨床判斷":
        if "特殊藥物" in item_text:
            return "special_med_handling_score"
        return "clinical_judgement_score" # Default for this category

    return None # 如果沒有匹配，返回 None

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



async def check_procedural_order(session_id: str, full_history: str, db: Session):
    """
    NEW: Checks if the conversation followed the correct educational sequence.
    """
    logger.info(f"[{session_id}] Performing final procedural order check...")
    prompt = f"""
    你是一個資深的衛教評分老師。請根據以下的完整對話紀錄，判斷學員(醫生)的衛教流程是否遵循了正確的順序。
    如果流程正確，只輸出 "1"。如果流程錯誤，只輸出 "0"。不要有任何其他文字或解釋。

    [正確的衛教順序]:
    1.  **醫療面談**: 開場問候、確認病人身份、確認檢查時間與類型。
    2.  **飲食管理**: 說明檢查前三天的低渣飲食原則。
    3.  **檢查前一天指導**: 說明無渣流質飲食、第一包清腸劑的服用時間與方法、水分補充。
    4.  **檢查當天指導**: 說明第二包清腸劑的服用時間與方法、禁水時間。
    5.  **額外補充**: 處理特殊用藥(如抗凝血劑)、說明排便理想狀態等。

    [對話紀錄]:
    {full_history}

    [判斷結果 (1 或 0)]:
    """
    response = await generate_llm_response(prompt, "gemma3:12b")
    score = 1 if "1" in response.strip() else 0
    
    # Save the result to AnswerLog
    try:
        stmt = sqlite_insert(AnswerLog).values(
            session_id=session_id,
            scoring_item_id='procedural_order_check',
            score=score,
            created_at=datetime.now()
        )
        on_conflict_stmt = stmt.on_conflict_do_update(
            index_elements=['session_id', 'scoring_item_id'],
            set_=dict(score=score)
        )
        db.execute(on_conflict_stmt)
        db.commit()
        logger.info(f"[{session_id}] Procedural order check result ({score}) saved.")
    except Exception as e:
        logger.error(f"[{session_id}] Failed to save procedural order check result: {e}")
        db.rollback()

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
    
    logger.info(f"開始為 Session {session_id} 匯總 RAG 評分...")
    
    chat_history_rows = db.query(ChatLog).filter(ChatLog.session_id == session_id).order_by(ChatLog.time.asc()).all()
    
    formatted_history = ""
    for entry in chat_history_rows:
        role_name = "User(醫生)" if entry.role == 'user' else "Agent(病人)"
        formatted_history += f"{role_name}: {entry.text}\n"
        
    await check_procedural_order(session_id, formatted_history, db)

    logger.info(f"[{session_id}] Aggregating final scores from AnswerLog...")
    
    passed_items_from_db  = db.query(AnswerLog).filter(
        AnswerLog.session_id == session_id,
        AnswerLog.score == 1
    ).all()
    
    passed_item_ids = {item.scoring_item_id for item in passed_items_from_db}
    
    final_scores = {
        "total_score": 0,
        "organization_efficient_score": 0,
        "clinical_judgement_score": 0,
        "diet_edu_score": 0,
        "medication_edu_score": 0,
        "npo_edu_score": 0,
        "special_med_handling_score": 0,
    }
    
    COMBINED_ITEM_IDS = {
        'proper_guidance_s1', 'proper_guidance_s2', 'proper_guidance_s3', 'proper_guidance_s4',
        'med_usage_timing_method.s1', 'med_usage_timing_method.s2',
        'hydration_and_goal.s1', 'hydration_and_goal.s2'
    }
    
    # First, calculate scores for standard, individual items
    for item_id, criterion in scoring_criteria_map.items():
        if item_id in passed_item_ids and item_id not in COMBINED_ITEM_IDS:
            field = map_criterion_to_field(criterion)
            if field in final_scores:
                final_scores[field] += criterion.get('weight', 0.0)
                
    # "適切發問及引導"
    s1 = 1 if 'proper_guidance_s1' in passed_item_ids else 0
    s2 = 1 if 'proper_guidance_s2' in passed_item_ids else 0
    s3 = 1 if 'proper_guidance_s3' in passed_item_ids else 0
    s4 = 1 if 'proper_guidance_s4' in passed_item_ids else 0
    appropriate_questioning_score = ((s1 + s2 + s3 + s4) / 4.0) * 3.0
    final_scores['organization_efficient_score'] += appropriate_questioning_score

    # "說明藥物使用時機及方式"
    med_s1 = 1 if 'med_usage_timing_method.s1' in passed_item_ids else 0
    med_s2 = 1 if 'med_usage_timing_method.s2' in passed_item_ids else 0
    med_usage_score = ((med_s1 + med_s2) / 2.0) * 1.0
    final_scores['medication_edu_score'] += med_usage_score

    # "說明水分補充方式及清腸理想狀態"
    hydro_s1 = 1 if 'hydration_and_goal.s1' in passed_item_ids else 0
    hydro_s2 = 1 if 'hydration_and_goal.s2' in passed_item_ids else 0
    hydration_score = ((hydro_s1 + hydro_s2) / 2.0) * 1.0
    # This can be categorized under diet or medication; let's put it in medication for now.
    final_scores['medication_edu_score'] += hydration_score
    
    # Calculate total score
    total_score = sum(final_scores.values())
    
    # Prepare data for database insertion (convert to string)
    score_data = {key: str(round(value, 2)) for key, value in final_scores.items()}
    score_data['total_score'] = str(round(total_score, 2))
    
    logger.info(f"[{session_id}] Final calculated scores: {score_data}")
    
    agent_settings = db.query(AgentSettings).filter(AgentSettings.agent_code == session_map.agent_code).first()
    summary_prompt = f"""
    你是一個專業的臨床指導老師，請根據以下對話紀錄，為使用者(醫生)的各項表現提供具體的文字評語和總結。
    請嚴格按照下面的JSON格式輸出，不要有任何額外的文字或解釋。
    **病患背景資料:** {agent_settings.med_info}
    **對話紀錄:** {formatted_history}
    **輸出格式 (JSON):**
    {{
        "total_summary": "一句話總評", "organization_efficient_summary": "組織效率的評語",
        "clinical_judgement_summary": "臨床判斷的評語", "diet_edu_summary": "飲食衛教的評語",
        "medication_edu_summary": "藥物衛教的評語", "npo_edu_summary": "禁水衛教的評語",
        "special_med_handling_summary": "特殊藥物處理的評語"
    }}

    """
    
    try:
        logger.info(f"[{session_id}] Generating text summary...")
        summary_result_str = await generate_llm_response(summary_prompt, "gemma3:12b")
        summary_data = parse_llm_json(summary_result_str)

        # 7. Save scores and summary to the database
        new_scores = Scores(session_id=session_id, **score_data)
        db.merge(new_scores)

        new_summary = Summary(session_id=session_id, **summary_data)
        db.merge(new_summary)

        session_map.score = score_data.get("total_score", "0")
        
        db.commit()
        logger.info(f"Session {session_id} scoring and summary saved successfully.")

        return {"status": "completed", "session_id": session_id}

    except json.JSONDecodeError as e:
        logger.error(f"LLM summary response JSON format error: {e}. Response: {summary_result_str}")
        db.rollback()
        raise HTTPException(status_code=500, detail="AI summary response format is incorrect.")
    except Exception as e:
        logger.error(f"Error during final scoring process: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An unknown error occurred during scoring: {e}")

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