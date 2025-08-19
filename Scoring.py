#目前需要大幅修正scoring.py的功能 目前計算總分的方式是錯誤的 需要改正
#真正的評分方式 是我總共有5種分類 分別是檢閱藥歷(6分) 醫療面談(6分) 諮商衛教(6分) 組織效率(6分) 臨床判斷(6分)
#然後summary也做錯了 應該要出現的是這五種分類的總結才對
#之後要測試 是分開做總結比較快 還是單獨總結比較快 也要比較兩者的正確率

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
    
# --- 修改: Pydantic ScoresModel ---
class ScoresModel(BaseModel):
    total_score: str
    review_med_history_score: str
    medical_interview_score: str
    counseling_edu_score: str
    organization_efficiency_score: str
    clinical_judgment_score: str
    
    model_config = ConfigDict(from_attributes=True)

# --- 修改: Pydantic SummaryModel ---
class SummaryModel(BaseModel):
    total_summary: str
    review_med_history_summary: str
    medical_interview_summary: str
    counseling_edu_summary: str
    organization_efficiency_summary: str
    clinical_judgment_summary: str

    model_config = ConfigDict(from_attributes=True)

# --- 修改: Pydantic SummaryResponse ---
class SummaryResponse(BaseModel):
    session_id: str
    agent_code: str
    username: str
    level: str
    case_info: str
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

async def calculate_organization_efficiency_score_llm(full_history: str) -> float:
    """
    呼叫 gemma3:12b 來對對話流程進行 0-6 分的評分。
    """
    prompt = f"""
    你是一位資深的臨床衛教評分老師。你的任務是根據以下的完整對話紀錄，評估學員(醫生)的衛教流程是否清晰、有條理且遵循了標準順序。

    [正確的衛教順序標準]:
    1.  **醫療面談**: 開場問候、確認病人身份、確認檢查時間與類型。
    2.  **飲食管理**: 說明檢查前三天的低渣飲食原則。
    3.  **檢查前一天指導**: 說明無渣流質飲食、第一包清腸劑的服用時間與方法、水分補充。
    4.  **檢查當天指導**: 說明第二包清腸劑的服用時間與方法、禁水時間。
    5.  **額外補充**: 處理特殊用藥(如抗凝血劑)、說明排便理想狀態等。

    [評分指南]:
    - 6分: 完美遵循順序，衛教流暢，所有要點均在正確的階段提出。
    - 4-5分: 大致遵循順序，可能有少數項目順序顛倒，但不影響整體理解。
    - 2-3分: 順序較為混亂，重要衛教項目散落在對話各處，結構性不佳。
    - 0-1分: 完全沒有順序，對話混亂，讓病人難以理解準備流程。

    [對話紀錄]:
    {full_history}

    [你的評分]:
    請根據上述標準，給出一個 0 到 6 之間的整數分數。**請只輸出一個數字，不要有任何其他文字、符號或解釋。**
    """
    
    logger.info(f"正在請求 gemma3:12b 進行組織效率評分 (0-6)...")
    response_text = await generate_llm_response(prompt, "gemma3:12b")
    
    try:
        # 使用正則表達式從回覆中找到第一個數字
        match = re.search(r'\d+', response_text)
        if match:
            score = int(match.group(0))
            # 將分數限制在 0-6 的範圍內，防止 LLM 超出範圍
            clamped_score = max(0, min(6, score))
            logger.info(f"LLM 原始分數: {score}，校正後分數: {clamped_score}")
            return float(clamped_score)
        else:
            logger.warning(f"無法從 LLM 回應中解析出組織效率分數: '{response_text}'。將預設為 0 分。")
            return 0.0
    except Exception as e:
        logger.error(f"解析組織效率分數時發生錯誤: {e}。回應: '{response_text}'。將預設為 0 分。")
        return 0.0
# --- API 端點 ---

@score_router.post("/finish")
async def finish_and_score_session(request: FinishRequest, db: Session = Depends(get_db)):
    """
    結束對話，觸發評分和總結流程 (包含標準及複合項目計分)。
    """
    session_id = request.session_id
    username = request.username
    
    session_map = db.query(SessionUserMap).filter(
        SessionUserMap.session_id == session_id, 
        SessionUserMap.username == username
    ).first()
    if not session_map:
        raise HTTPException(status_code=404, detail="指定的 Session 或 Username 不存在")
    
    if db.query(Scores).filter(Scores.session_id == session_id).first():
        logger.warning(f"Session {session_id} 的評分已存在，將跳過重新計算。")
        return {"status": "already_scored", "session_id": session_id}
    
    logger.info(f"開始為 Session {session_id} 計算最終分數 (包含複合規則)...")

    # 1. 獲取所有得分項目的 ID，放入一個 Set 以便快速查詢
    passed_items_query = db.query(AnswerLog.scoring_item_id).filter(
        AnswerLog.session_id == session_id,
        AnswerLog.score == 1
    ).all()
    passed_item_ids = {item.scoring_item_id for item in passed_items_query}

    # 2. 定義所有複合規則的子項目ID
    COMPOSITE_SUB_ITEM_IDS = {
        'proper_guidance_s1', 'proper_guidance_s2', 'proper_guidance_s3', 'proper_guidance_s4',
        'med_usage_timing_method.s1', 'med_usage_timing_method.s2',
        'hydration_and_goal.s1', 'hydration_and_goal.s2'
    }

    # 3. 初始化五大維度的分數
    final_scores = {
        "review_med_history_score": 0.0,
        "medical_interview_score": 0.0,
        "counseling_edu_score": 0.0,
        "organization_efficiency_score": 0.0,
        "clinical_judgment_score": 0.0,
    }

    CATEGORY_TO_KEY_MAP = {
        "檢閱藥歷": "review_med_history_score",
        "醫療面談": "medical_interview_score",
        "諮商衛教": "counseling_edu_score",
        "臨床判斷": "clinical_judgment_score"
    }

    # 4. 計算【標準項目】的分數
    # 遍歷所有可能的評分項，如果它通過了【且】不是一個複合子項目，就計分
    for item_id, criterion in scoring_criteria_map.items():
        if item_id in passed_item_ids and item_id not in COMPOSITE_SUB_ITEM_IDS:
            weight = criterion.get('weight', 0.0)
            category = criterion.get('category')

            if item_id == 'procedural_order_check':
                final_scores['organization_efficiency_score'] += weight
            elif category in CATEGORY_TO_KEY_MAP:
                score_key = CATEGORY_TO_KEY_MAP[category]
                final_scores[score_key] += weight
            else:
                logger.warning(f"標準項目計分時發現未知 category '{category}' for item ID: {item_id}")
    
    # 5. 根據規則計算【複合項目】的分數
    
    # 規則 1: 適切發問及引導 (醫療面談)
    s1_guidance = 1 if 'proper_guidance_s1' in passed_item_ids else 0
    s2_guidance = 1 if 'proper_guidance_s2' in passed_item_ids else 0
    s3_guidance = 1 if 'proper_guidance_s3' in passed_item_ids else 0
    s4_guidance = 1 if 'proper_guidance_s4' in passed_item_ids else 0
    questioning_score = ((s1_guidance + s2_guidance + s3_guidance + s4_guidance) / 4.0) * 3.0
    final_scores['medical_interview_score'] += questioning_score
    logger.info(f"[{session_id}] 適切發問及引導分數: {questioning_score}")

    # 規則 2: 說明藥物使用時機及方式 (諮商衛教)
    s1_med = 1 if 'med_usage_timing_method.s1' in passed_item_ids else 0
    s2_med = 1 if 'med_usage_timing_method.s2' in passed_item_ids else 0
    med_usage_score = ((s1_med + s2_med) / 2.0) * 1.0
    final_scores['counseling_edu_score'] += med_usage_score
    logger.info(f"[{session_id}] 藥物使用時機方式分數: {med_usage_score}")

    # 規則 3: 說明水分補充方式及清腸理想狀態 (諮商衛教)
    s1_hydro = 1 if 'hydration_and_goal.s1' in passed_item_ids else 0
    s2_hydro = 1 if 'hydration_and_goal.s2' in passed_item_ids else 0
    hydration_score = ((s1_hydro + s2_hydro) / 2.0) * 1.0
    final_scores['counseling_edu_score'] += hydration_score
    logger.info(f"[{session_id}] 水分補充與理想狀態分數: {hydration_score}")

    # 6. 計算總分
    total_score = sum(final_scores.values())

    score_data_to_save = {key: str(round(value, 2)) for key, value in final_scores.items()}
    score_data_to_save['total_score'] = str(round(total_score, 2))
    
    logger.info(f"[{session_id}] 最終計算分數: {score_data_to_save}")
    
    # 7. 生成總結 (此部分邏輯不變)
    agent_settings = db.query(AgentSettings).filter(AgentSettings.agent_code == session_map.agent_code).first()
    chat_history_rows = db.query(ChatLog).filter(ChatLog.session_id == session_id).order_by(ChatLog.time.asc()).all()
    formatted_history = "\n".join([f"{'學員' if e.role == 'user' else '病患'}: {e.text}" for e in chat_history_rows])

    summary_prompt = f"""
    你是一位專業的臨床衛教指導老師。請根據以下病患背景和完整的對話紀錄，為學員的表現提供具體評語。
    請嚴格按照下面的JSON格式輸出，對每一個項目提供簡潔、有建設性的評語。

    [病患背景資料]:
    {agent_settings.med_info if agent_settings else '無'}

    [對話紀錄]:
    {formatted_history}

    [輸出格式 (JSON)]:
    {{
        "total_summary": "一句話總評，點出學員表現最突出或最需改進的地方。",
        "review_med_history_summary": "針對「檢閱藥歷」能力的評語。",
        "medical_interview_summary": "針對「醫療面談」技巧的評語(包含問候、引導、確認資訊等)。",
        "counseling_edu_summary": "針對「諮商衛教」內容正確性的評語(包含飲食、用藥方式、水分補充等)。",
        "organization_efficiency_summary": "針對「組織效率」(衛教流程順序)的評語。",
        "clinical_judgment_summary": "針對「臨床判斷」能力的評語(包含判斷用藥時間、禁水時間、處理特殊藥物等)。"
    }}
    """
    
    try:
        logger.info(f"[{session_id}] 正在生成文字總結...")
        summary_result_str = await generate_llm_response(summary_prompt, "gemma3:4b")
        summary_data = parse_llm_json(summary_result_str)

        db.merge(Scores(session_id=session_id, **score_data_to_save))
        db.merge(Summary(session_id=session_id, **summary_data))

        session_map.score = score_data_to_save.get("total_score")
        session_map.is_completed = True
        
        db.commit()
        logger.info(f"Session {session_id} 的評分和總結已成功儲存。")

        return {"status": "completed", "session_id": session_id}

    except Exception as e:
        logger.error(f"處理 Session {session_id} 評分總結時發生錯誤: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"生成評分報告時發生內部錯誤: {e}")

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