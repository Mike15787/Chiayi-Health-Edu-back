# Scoring.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session
import logging
import json
from typing import Dict, List, Any
from datetime import datetime
from utils import generate_llm_response
from databases import (
    get_db, ChatLog, AgentSettings, SessionUserMap, ScoringAttributionLog, Scores, Summary, AnswerLog, SessionInteractionLog,
    get_module_id_by_session
)

# 新增：從 main.py 中賦值的 ScoringServiceManager 實例
scoring_service: Any = None 

logger = logging.getLogger(__name__)
score_router = APIRouter(prefix="/scoring", tags=["Scoring"])

class FinishRequest(BaseModel):
    session_id: str
    username: str
    
# --- 新增 Pydantic Models ---
class ScoreItemDetail(BaseModel):
    item_id: str
    item_name: str          # item
    description: str        # 任務說明
    weight: float           # weight (滿分)
    user_score: float       # 實際得分
    scoring_type: str       # type (評分方法)
    relevant_dialogues: List[str] = [] # 相關對話內容

class CategoryDetail(BaseModel):
    category_name: str
    items: List[ScoreItemDetail]

class DetailedScoreResponse(BaseModel):
    session_id: str
    details: Dict[str, CategoryDetail] # key 是 category 名稱 (如 "醫療面談")

# --- Pydantic Models (對應新的資料庫欄位) ---
class ScoresModel(BaseModel):
    total_score: str
    review_med_history_score: str
    medical_interview_score: str
    counseling_edu_score: str
    clinical_judgment_score: str
    humanitarian_score: str
    organization_efficiency_score: str
    overall_clinical_skills_score: str
    model_config = ConfigDict(from_attributes=True)

class SummaryModel(BaseModel):
    total_summary: str
    review_med_history_summary: str
    medical_interview_summary: str
    counseling_edu_summary: str
    clinical_judgment_summary: str
    humanitarian_summary: str
    organization_efficiency_summary: str
    overall_clinical_skills_summary: str
    model_config = ConfigDict(from_attributes=True)

class SummaryResponse(BaseModel):
    session_id: str
    agent_code: str
    username: str
    level: str
    case_info: str
    scores: ScoresModel
    summaries: SummaryModel

class AchievedItemModel(BaseModel):
    item_id: str
    item_description: str

class FeedbackMessageModel(BaseModel):
    id: int
    role: str
    text: str
    time: datetime
    achieved_items: List[AchievedItemModel] = []

class FullFeedbackResponse(BaseModel):
    session_id: str
    full_history: List[FeedbackMessageModel]

# --- 輔助函數：安全的 JSON 解析 ---
def parse_llm_json(raw_text: str) -> dict:
    """嘗試解析 JSON，支援 Markdown code block 剝離"""
    try:
        # 移除可能的 Markdown 標記
        clean_text = raw_text.replace("```json", "").replace("```", "").strip()
        
        start_index = clean_text.find('{')
        end_index = clean_text.rfind('}')
        if start_index != -1 and end_index != -1:
            json_str = clean_text[start_index : end_index + 1]
            return json.loads(json_str)
        return json.loads(clean_text)
    except Exception as e:
        logger.error(f"JSON 解析失敗: {e}")
        raise

def get_default_summary() -> dict:
    """回傳預設的總結內容，防止資料庫寫入失敗"""
    return {
        "total_summary": "系統無法生成總結（AI回應格式錯誤，但分數已計算）。",
        "review_med_history_summary": "無評語",
        "medical_interview_summary": "無評語",
        "counseling_edu_summary": "無評語",
        "clinical_judgment_summary": "無評語",
        "humanitarian_summary": "無評語",
        "organization_efficiency_summary": "無評語",
        "overall_clinical_skills_summary": "無評語"
    }

# --- API Endpoints ---

@score_router.post("/finish")
async def finish_and_score_session(request: FinishRequest, db: Session = Depends(get_db)):
    session_id = request.session_id
    username = request.username
    
    logger.info(f"收到結束請求: Session={session_id}, User={username}")

    try:
        # 1. 驗證 Session
        session_map = db.query(SessionUserMap).filter(
            SessionUserMap.session_id == session_id
        ).first()
        
        if not session_map:
            raise HTTPException(status_code=404, detail="Session not found")
        
        module_id = session_map.module_id

        if scoring_service is None:
            raise HTTPException(status_code=500, detail="Service not initialized")

        # 2. 計算分數 (Step 1)
        # 注意：這裡呼叫 scoring_logic.calculate_final_scores，它已經包含了 UI 分數的讀取與合併
        logger.info("開始計算分數...")
        try:
            calculated_scores = await scoring_service.calculate_final_scores(session_id, module_id, db)
            logger.info(f"分數計算完成: {calculated_scores}")
        except Exception as e:
            logger.error(f"分數計算邏輯崩潰: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 如果分數計算失敗，我們無法繼續，必須報錯
            raise HTTPException(status_code=500, detail=f"Score calculation failed: {str(e)}")

        # 3. 生成總結 (Step 2) - 加入強力容錯
        logger.info("開始生成總結...")
        summary_data = get_default_summary() # 先給預設值
        
        try:
            agent_settings = db.query(AgentSettings).filter(AgentSettings.agent_code == session_map.agent_code).first()
            chat_history_rows = db.query(ChatLog).filter(ChatLog.session_id == session_id).order_by(ChatLog.time.asc()).all()
            chat_history_dicts = [{'role': e.role, 'text': e.text} for e in chat_history_rows]

            summary_generator = scoring_service.get_summary_generator(module_id)
            module_config = scoring_service.get_module_config(module_id)
            
            summary_prompt = await summary_generator(agent_settings, chat_history_dicts)
            
            # 呼叫 LLM
            summary_result_str = await generate_llm_response(summary_prompt, module_config.SCORING_MODEL_NAME)
            
            # 解析 JSON
            parsed_summary = parse_llm_json(summary_result_str)
            # 合併，確保不會少了欄位
            summary_data.update(parsed_summary)
            
        except Exception as e:
            logger.error(f"總結生成失敗 (將使用預設值): {e}")
            # 重要：這裡我們捕獲異常但不拋出，讓程式繼續往下走去存分數

        # 4. 寫入資料庫 (Step 3)
        logger.info("正在寫入資料庫...")
        try:
            # 處理分數寫入 (Scores)
            existing_score = db.query(Scores).filter(Scores.session_id == session_id).first()
            
            if existing_score:
                logger.info("更新現有分數紀錄")
                # 更新欄位
                for k, v in calculated_scores.items():
                    if hasattr(existing_score, k):
                        setattr(existing_score, k, v)
            else:
                logger.info("建立新分數紀錄")
                # 過濾掉不在 model 裡的 key，防止報錯
                valid_keys = Scores.__table__.columns.keys()
                filtered_scores = {k: v for k, v in calculated_scores.items() if k in valid_keys}
                new_scores = Scores(session_id=session_id, module_id=module_id, **filtered_scores)
                db.add(new_scores)

            # 處理總結寫入 (Summary)
            valid_summary_keys = Summary.__table__.columns.keys()
            filtered_summary = {k: v for k, v in summary_data.items() if k in valid_summary_keys}
            summary_record = Summary(session_id=session_id, module_id=module_id, **filtered_summary)
            
            # merge 會自動判斷是 insert 還是 update
            db.merge(summary_record)

            # 更新 Session 狀態
            session_map.score = calculated_scores.get("total_score", "0")
            session_map.is_completed = True
            
            db.commit()
            logger.info("資料庫寫入成功！")
            
        except Exception as e:
            db.rollback()
            logger.error(f"資料庫寫入失敗: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Database commit failed: {str(e)}")

        return {"status": "completed", "session_id": session_id}

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"未預期的錯誤: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@score_router.get("/details/{session_id}", response_model=DetailedScoreResponse)
async def get_score_details(session_id: str, db: Session = Depends(get_db)):
    """
    獲取詳細的評分細節。邏輯已委派給各個模組。
    """
    # 1. 取得 Session 對應的 module_id
    session_map = db.query(SessionUserMap).filter(SessionUserMap.session_id == session_id).first()
    if not session_map:
        raise HTTPException(status_code=404, detail="Session not found")
    
    module_id = session_map.module_id

    # 2. 確保 Service 已初始化
    if scoring_service is None:
         raise HTTPException(status_code=500, detail="Scoring Service not initialized")

    try:
        # 3. 呼叫 Manager -> Module 取得詳細資料
        # 注意：這裡回傳的格式是一個 Dict，符合 DetailedScoreResponse 的結構
        details_dict = await scoring_service.get_detailed_scores(session_id, module_id, db)
        
        return DetailedScoreResponse(
            session_id=session_id,
            details=details_dict
        )

    except Exception as e:
        logger.error(f"Error fetching detailed scores for session {session_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to generate score details: {str(e)}")
    
@score_router.get("/summary/{session_id}", response_model=SummaryResponse)
async def get_summary_data(session_id: str, db: Session = Depends(get_db)):
    session_map = db.query(SessionUserMap).filter(SessionUserMap.session_id == session_id).first()
    if not session_map:
        raise HTTPException(status_code=404, detail="Session not found")

    scores = db.query(Scores).filter(Scores.session_id == session_id).first()
    summaries = db.query(Summary).filter(Summary.session_id == session_id).first()
    
    if not scores or not summaries:
        # 如果找不到資料，回傳 404
        raise HTTPException(status_code=404, detail="Score data not found. Processing might have failed.")

    agent_settings = db.query(AgentSettings).filter(AgentSettings.agent_code == session_map.agent_code).first()
    case_info = f"性別: {agent_settings.gender}, 年齡: {agent_settings.age}, 主訴: {agent_settings.disease}" if agent_settings else "未知"

    return SummaryResponse(
        session_id=session_id,
        agent_code=session_map.agent_code,
        username=session_map.username,
        level=agent_settings.med_complexity if agent_settings else "未知",
        case_info=case_info,
        scores=scores,
        summaries=summaries
    )

@score_router.get("/feedback/{session_id}", response_model=FullFeedbackResponse)
async def get_full_feedback(session_id: str, db: Session = Depends(get_db)):
    
    # 1. 先檢查 Session 是否存在 (而不是檢查有沒有對話)
    session_map = db.query(SessionUserMap).filter(SessionUserMap.session_id == session_id).first()
    if not session_map:
        raise HTTPException(status_code=404, detail="Session not found")
    
    chat_logs = db.query(ChatLog).filter(ChatLog.session_id == session_id).order_by(ChatLog.time.asc()).all()

    module_id = get_module_id_by_session(session_id)
    if not module_id: module_id = "colonoscopy_bowklean"

    # 取得評分標準說明
    try:
        if scoring_service:
            scoring_criteria_map = scoring_service.get_scoring_criteria_map(module_id)
        else:
            scoring_criteria_map = {}
    except:
        scoring_criteria_map = {}

    attributions = db.query(ScoringAttributionLog).filter(ScoringAttributionLog.session_id == session_id).all()
    
    attribution_map = {}
    for attr in attributions:
        if attr.chat_log_id not in attribution_map:
            attribution_map[attr.chat_log_id] = []
        
        item_details = scoring_criteria_map.get(attr.scoring_item_id, {})
        desc = item_details.get('item', attr.scoring_item_id)
        
        attribution_map[attr.chat_log_id].append(
            AchievedItemModel(item_id=attr.scoring_item_id, item_description=desc)
        )

    full_history = []
    for log in chat_logs:
        achieved_items = attribution_map.get(log.id, [])
        full_history.append(
            FeedbackMessageModel(
                id=log.id,
                role=log.role,
                text=log.text,
                time=log.time,
                achieved_items=achieved_items
            )
        )
    
    return FullFeedbackResponse(
        session_id=session_id,
        full_history=full_history
    )