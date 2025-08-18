from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Query, Path, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import requests
import tempfile
import os
import json
import edge_tts
import whisper
import torch
import uuid
from datetime import datetime, timedelta
import logging
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
import traceback
from Login import auth_router
from FindHistory import history_router
from Scoring import score_router
from scoring_service import ScoringService
from utils import generate_llm_response

# 導入資料庫相關模組
from databases import (
    get_conversation_history, 
    save_chat_message, 
    save_answer_log,
    get_db,
    save_db_conversation_summary,
    get_latest_conversation_summary,
    get_user_message_count,
    SessionLocal,
    ChatLog,
    AnswerLog,
    UserLogin,
    AgentSettings,
    SessionUserMap,
    get_latest_user_messages,
    get_latest_chat_history_for_scoring,
    PrecomputedSessionAnswer
)

from agentset import insert_agent_data

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 初始化模型 ---
whisper_model = None
available_voices = []

# --- 目錄設定 ---
AUDIO_DIR = "audio"
TEMP_DIR = "temp"
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# --- 線程池用於處理 AI 任務 ---
executor = ThreadPoolExecutor(max_workers=4)

# --- 新增：初始化評分服務 ---
scoring_service: ScoringService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理"""
    global scoring_service
    
    # 啟動時執行
    logger.info("AI Voice Chat API 啟動中...")
    
    logger.info("正在檢查並插入 Agent 設定資料...")
    insert_agent_data()
    logger.info("Agent 設定資料檢查完畢。")
    
    # 初始化模型
    await init_models()
    
    # 創建無聲音頻檔案
    create_silence_audio()
    
    # --- 新增：初始化 ScoringService ---
    logger.info("Initializing RAG Scoring Service...")
    scoring_service = ScoringService()
    logger.info("RAG Scoring Service initialized.")
    
    # 清理舊的臨時檔案
    for temp_file in os.listdir(TEMP_DIR):
        try:
            temp_path = os.path.join(TEMP_DIR, temp_file)
            if os.path.isfile(temp_path):
                os.remove(temp_path)
        except Exception as e:
            logger.warning(f"清理臨時檔案失敗: {e}")
    
    logger.info("AI Voice Chat API 啟動完成")
    # yield 前的程式碼會在API 啟動時執行
    yield
    #yield 後的程式碼會在API 關閉時執行
    # 關閉時執行
    logger.info("AI Voice Chat API 關閉中...")
    executor.shutdown(wait=True)
    logger.info("AI Voice Chat API 已關閉")

app = FastAPI(title="AI Voice Chat API", version="1.0.0", lifespan=lifespan)

# 修正 CORS 配置 - 更具體的設置
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8080", 
        "http://localhost:5173",  # Vite 默認端口
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(history_router)
app.include_router(score_router)


# --- 初始化模型 ---
async def init_models():
    """初始化模型（異步版本）"""
    global whisper_model, available_voices
    
    try:
        # 初始化 Whisper STT 模型
        logger.info("初始化 Whisper STT 模型...")
        whisper_model = whisper.load_model("base")
        logger.info("Whisper STT 模型初始化完成")
        
    except Exception as e:
        logger.error(f"Whisper STT 模型初始化失敗: {e}")
        logger.error(traceback.format_exc())
    
    try:
        # 獲取可用的 edge-tts 語音列表
        logger.info("獲取 edge-tts 語音列表...")
        voices = await edge_tts.list_voices()
        
        # 篩選繁體中文語音
        chinese_voices = [v for v in voices if 'zh-TW' in v['Locale'] or 'zh-CN' in v['Locale']]
        available_voices = chinese_voices[:5]  # 取前5個
        
        logger.info(f"可用語音數量: {len(available_voices)}")
        for voice in available_voices:
            logger.info(f"  - {voice['Name']}: {voice['ShortName']} ({voice['Gender']})")
            
    except Exception as e:
        logger.error(f"edge-tts 語音列表獲取失敗: {e}")
        logger.error(traceback.format_exc())


# --- Pydantic 模型 ---
class ChatRequest(BaseModel):
    text: str
    session_id: str = None
    voice: str = None  # 可選的語音選擇

class STTResponse(BaseModel):
    text: str
    confidence: float = 0.0

class ChatResponse(BaseModel):
    text: str
    audioUrl: str
    session_id: str

class VoiceInfo(BaseModel):
    name: str
    short_name: str
    gender: str
    locale: str

class AgentInfo(BaseModel):
    agent_code: str
    gender: str
    age: str
    med_info: str
    disease: str
    med_complexity: str
    med_code: str
    special_status: str
    check_time: str
    check_day: Optional[int] = None
    check_type: str
    low_cost_med: str
    payment_fee: str
    laxative_experience: str
    
# 修改 SessionInfo 模型，增加 agent_code
class SessionInfo(BaseModel):
    session_id: str
    username: str
    agent_code: str



# --- 對話管理類別 ---
class ConversationManager:

    """對話管理器，負責處理歷史記錄和總結"""
    
    def __init__(self):
        # 移除了舊的長度限制屬性
        pass
        
    async def get_formatted_history(self, session_id: str, limit: int = 10) -> str:
        """
        獲取格式化的對話歷史。
        如果存在總結，則返回總結；否則，返回最近的對話記錄。
        """
        try:
            # --- MODIFIED (Requirement 4): 優先獲取總結 ---
            summary = get_latest_conversation_summary(session_id)
            
            if summary:
                # 如果有總結，直接使用總結作為歷史
                formatted_history = f"# 之前的對話總結：\n{summary}"
                logger.info(f"Session {session_id}: 使用了對話總結作為歷史。")
                return formatted_history
            else:
                # 如果沒有總結，獲取最近的對話
                recent_history = get_conversation_history(session_id, limit=limit)
                if recent_history:
                    formatted_history = f"# 最近的對話：\n{recent_history}"
                    return formatted_history
                return "" # 如果兩者都沒有，返回空字串
            
        except Exception as e:
            logger.error(f"獲取對話歷史錯誤: {e}")
            return ""
    
   
    def save_conversation_summary(self, session_id: str, summary: str):
        """
        儲存對話總結到資料庫。
        """
        try:
            # --- MODIFIED (Requirement 3): 呼叫新的資料庫函數 ---
            save_db_conversation_summary(session_id, summary)
        except Exception as e:
            logger.error(f"儲存對話總結錯誤: {e}")
    
    async def should_summarize(self, session_id: str) -> bool:
        """
        判斷是否需要總結歷史。
        每 2 次使用者輸入後觸發一次。
        """
        try:
            # --- MODIFIED (Requirement 1): 根據使用者訊息數量判斷 ---
            user_messages_count = get_user_message_count(session_id)
            # 當使用者訊息數為 2, 4, 6... 時觸發
            return user_messages_count > 0 and user_messages_count % 2 == 0
        except Exception as e:
            logger.error(f"檢查總結需求錯誤: {e}")
            return False
    
    async def summarize_conversation(self, session_id: str):
        """總結對話歷史 (這將在背景執行)"""
        try:
            logger.info(f"開始為 Session {session_id} 進行對話總結...")
            # 獲取較長的歷史記錄用於總結 (即使已有總結，也基於更長的歷史生成新總結)
            recent_user_inputs = get_latest_user_messages(session_id, count=3)
            
            # 如果訊息少於2條，總結的意義不大，直接跳過
            if not recent_user_inputs or len(recent_user_inputs) < 2:
                logger.info(f"Session {session_id} 使用者輸入太少，跳過總結。")
                return
            
            # 將訊息列表組合成一個字串，方便放入 prompt
            inputs_text = "\n".join(f"- {msg}" for msg in recent_user_inputs)

            summary_prompt = (
                "你是一個對話助理，請將以下使用者最近的幾句話，濃縮成一個簡潔的要點，說明使用者目前最關心的問題或確認的資訊。\n"
                "總結必須非常簡短，不超過50個字，只包含核心資訊。\n\n"
                f"使用者最近的對話：\n{inputs_text}\n\n"
                "總結："
            )
            
            summary = await generate_llm_response(summary_prompt)
            
            if summary and "抱歉" not in summary:
                # 儲存總結
                # 1. 將生成的總結印到終端機
                logger.info(f"--- [DEBUG] Generated Summary for Session {session_id} ---")
                logger.info(summary)
                logger.info("---------------------------------------------------------")
                self.save_conversation_summary(session_id, summary)
                logger.info(f"Session {session_id} 的對話總結已成功生成並儲存。")
            else:
                logger.warning(f"Session {session_id} 總結生成失敗或內容無效。")
            
        except Exception as e:
            logger.error(f"背景總結任務錯誤 for Session {session_id}: {e}")
            logger.error(traceback.format_exc())


async def generate_tts_audio(text: str, voice: str = None) -> str:
    """使用 edge-tts 生成 TTS 音頻檔案"""
    global available_voices
    
    if not text or text.strip() == "":
        text = "沒有內容"
    
    # 選擇語音
    if voice and voice in [v['ShortName'] for v in available_voices]:
        selected_voice = voice
    else:
        # 預設使用繁體中文女聲
        default_voices = [
            "zh-TW-HsiaoChenNeural",  # 繁體中文女聲
            "zh-TW-YunJheNeural",     # 繁體中文男聲
            "zh-CN-XiaoxiaoNeural",   # 簡體中文女聲
            "zh-CN-YunxiNeural"       # 簡體中文男聲
        ]
        
        selected_voice = default_voices[0]
        
        # 如果有可用語音列表，優先使用
        if available_voices:
            # 優先選擇女聲
            female_voices = [v for v in available_voices if v['Gender'] == 'Female']
            if female_voices:
                selected_voice = female_voices[0]['ShortName']
            else:
                selected_voice = available_voices[0]['ShortName']
    
    try:
        audio_filename = f"tts_{uuid.uuid4().hex}.wav"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        
        # 使用 edge-tts 生成音頻
        communicate = edge_tts.Communicate(text, selected_voice)
        await communicate.save(audio_path)
        
        # 檢查檔案是否生成成功
        if not os.path.exists(audio_path):
            logger.error(f"edge-tts 音頻檔案生成失敗: {audio_path}")
            return "/audio/silence.wav"
        
        logger.info(f"edge-tts 音頻生成成功: {audio_filename} (語音: {selected_voice})")
        return f"/audio/{audio_filename}"
        
    except Exception as e:
        logger.error(f"edge-tts 生成錯誤: {e}")
        logger.error(traceback.format_exc())
        return "/audio/silence.wav"

def create_silence_audio():
    """創建無聲音頻檔案作為備用 生成錯誤時可保證不整個系統停擺"""
    try:
        import numpy as np
        from scipy.io import wavfile
        
        silence_path = os.path.join(AUDIO_DIR, "silence.wav")
        
        if not os.path.exists(silence_path):
            # 創建 1 秒的無聲音頻
            sample_rate = 22050
            duration = 1.0
            samples = int(sample_rate * duration)
            audio_data = np.zeros(samples, dtype=np.int16)
            
            wavfile.write(silence_path, sample_rate, audio_data)
            logger.info("無聲音頻檔案創建成功")
    
    except Exception as e:
        logger.warning(f"無法創建無聲音頻檔案: {e}")

def precompute_and_save_answers(session_id: str, agent_code: str, db: Session):
    """根據 agent 設定預先計算並儲存答案"""
    try:
        agent_settings = db.query(AgentSettings).filter(AgentSettings.agent_code == agent_code).first()
        if not agent_settings:
            logger.error(f"[{session_id}] Cannot precompute answers, agent_settings not found for {agent_code}")
            return

        # 1. 日期計算
        today = datetime.now()
        check_day_offset = agent_settings.check_day
        exam_date = today + timedelta(days=check_day_offset)
        
        def format_date(d):
            return f"{d.month}月{d.day}日"

        exam_day_str = format_date(exam_date)
        prev_1d_str = format_date(exam_date - timedelta(days=1))
        prev_2d_str = format_date(exam_date - timedelta(days=2))
        prev_3d_str = format_date(exam_date - timedelta(days=3))

        # 2. 判斷真實檢查類型
        # 根據模擬答案，即使 agent 不知道，我們也基於費用判斷
        actual_check_type = "一般"
        if agent_settings.payment_fee and '4500' in agent_settings.payment_fee:
            actual_check_type = "無痛"
        
        # 3. 計算第二包藥劑服用時間
        check_time_str = agent_settings.check_time # e.g., "上午11:20"
        hour = int(check_time_str.split(':')[0][-2:]) # 取得小時
        
        second_dose_time = "凌晨7點" # 預設下午
        if "上午" in check_time_str:
            if hour < 10:
                second_dose_time = "凌晨3點"
            elif 10 <= hour < 12:
                second_dose_time = "凌晨4點"
        
        # 4. 計算禁水時間
        npo_hours_before = 3 if actual_check_type == "無痛" else 2
        # 將 "上午11:20" 轉換為 datetime 物件
        check_time_obj = datetime.strptime(f"{exam_date.strftime('%Y-%m-%d')} {hour}:{check_time_str.split(':')[1]}", '%Y-%m-%d %H:%M')
        if "下午" in check_time_str and hour != 12:
             check_time_obj += timedelta(hours=12)

        npo_start_obj = check_time_obj - timedelta(hours=npo_hours_before)
        npo_start_time = npo_start_obj.strftime("上午%I:%M").replace("AM","").replace(" 0","") # 格式化為 "上午8:20"

        # 5. 存入資料庫
        new_precomputed = PrecomputedSessionAnswer(
            session_id=session_id,
            exam_day=exam_day_str,
            prev_1d=prev_1d_str,
            prev_2d=prev_2d_str,
            prev_3d=prev_3d_str,
            second_dose_time=second_dose_time,
            npo_start_time=npo_start_time,
            actual_check_type=actual_check_type
        )
        db.merge(new_precomputed) # merge 可以處理已存在的情況
        db.commit()
        logger.info(f"[{session_id}] Precomputed answers saved successfully.")

    except Exception as e:
        logger.error(f"[{session_id}] Error in precompute_and_save_answers: {e}")
        db.rollback()


def transcribe_audio(audio_path: str) -> dict:
    """使用 Whisper 轉錄音頻"""
    if whisper_model is None:
        return {"text": "", "confidence": 0.0, "language": "zh"}
    
    try:
        result = whisper_model.transcribe(
            audio_path, 
            language="zh",
            fp16=False
        )
        
        text = result["text"].strip()
        segments = result.get("segments", [])
        avg_confidence = sum(seg.get("no_speech_prob", 0) for seg in segments) / len(segments) if segments else 0
        confidence = 1.0 - avg_confidence
        
        return {
            "text": text,
            "confidence": confidence,
            "language": result.get("language", "zh")
        }
    except Exception as e:
        logger.error(f"音頻轉錄錯誤: {e}")
        logger.error(traceback.format_exc())
        return {"text": "", "confidence": 0.0, "language": "zh"}

def build_optimized_prompt(user_text: str, history: str, agent_settings_dict: dict = None) -> str:
    """構建優化的 prompt，包含 agent 設定"""
    
    '''我的錯 應該要把計算後的檢查時間加入到prompt裡面 這才是正確的'''
    
   
    # --- 1. 計算實際檢查日期 ---
    check_day_offset = agent_settings_dict.get('check_day', 0)
    today = datetime.now()
    exam_date = today + timedelta(days=check_day_offset)
    # 使用 %-m 和 %-d (在Linux/macOS) 或 %#m 和 %#d (在Windows) 來去除前導零
    # 為了跨平台，我們手動處理
    exam_day_str = f"{exam_date.month}月{exam_date.day}日"
    today_str = f"{today.month}月{today.day}日"
    
    # --- 2. 構建基礎角色和特殊指令 ---
    # 使用列表來組合指令，更具可讀性和擴展性
    instructions = [
        f"你是一個病患，預計在 {check_day_offset} 天後（也就是 {exam_day_str}）要做大腸鏡檢查。今天（{today_str}）你剛拿到醫生開的清腸藥（保可淨），正要向護理師（使用者）請教如何使用及相關衛教事宜。",
        "你需要仔細聆聽護理師的衛教內容，並給予簡短、自然的回應，讓他知道你有在聽。",
        "你的回答應盡量簡潔，通常在10個字以內。"
    ]
    
    # 處理特殊狀況
    special_status = agent_settings_dict.get('special_status', '無')
    if "不知道檢查型態" in special_status:
        instructions.append(
            "【特殊任務】由於你不確定自己的檢查類型，你必須在對話中主動詢問護理師：『我的檢查是一般檢查還是無痛的？需要麻醉嗎？』"
        )
    # 未來可以繼續增加 elif 條件來處理其他 special_status
    # elif "情緒問題" in special_status:
    #     instructions.append("【特殊任務】你的情緒比較焦慮，可能會重複確認某些細節。")
    
    base_role_prompt = "\n".join(instructions)
    
    # --- 3. 組合個人資訊 ---
    personal_info = f"""
---
你的個人資訊如下，請根據這些資訊與護理師互動：
- 性別：{agent_settings_dict.get('gender', '未提供')}
- 年齡：{agent_settings_dict.get('age', '未提供')}
- 疾病史：{agent_settings_dict.get('disease', '未提供')}
- 目前用藥：{agent_settings_dict.get('med_info', '未提供')}
- 預計檢查日期：{exam_day_str} ({check_day_offset} 天後)
- 預計檢查時間：{agent_settings_dict.get('check_time', '未提供')}
- 檢查類型：{agent_settings_dict.get('check_type', '未提供')}
- 繳交費用：{agent_settings_dict.get('payment_fee', '未提供')}
- 過去使用瀉藥經驗：{agent_settings_dict.get('laxative_experience', '未提供')}
- 是否需要代購低渣飲食：{agent_settings_dict.get('low_cost_med', '未提供')}
---
"""
    system_prompt = base_role_prompt + personal_info
    
    # --- MODIFIED (Requirement 4): 簡化 prompt 組合 ---
    # history 變數現在可能是「最近的對話」或「之前的對話總結」，邏輯保持不變
    # --- 4. 組合最終的 Prompt ---
    if history and history.strip():
        prompt = (
            f"{system_prompt}\n\n"
            "以下是你們之前的對話紀錄：\n"
            f"{history}\n\n"
            f"護理師：{user_text}\n"
            "你："
        )
    else:
        prompt = (
            f"{system_prompt}\n\n"
            "現在，對話開始。\n"
            f"護理師：{user_text}\n"
            "你："
        )
    
    return prompt

def get_agent_code_by_session(session_id: str) -> Optional[str]:
    """根据 session_id 获取 agent_code"""
    try:
        db = SessionLocal()
        session_user_map = db.query(SessionUserMap).filter(
            SessionUserMap.session_id == session_id
        ).first()
        
        if not session_user_map:
            logger.warning(f"找不到 session_id: {session_id} 的资料")
            return None
            
        return session_user_map.agent_code
        
    except Exception as e:
        logger.error(f"获取 agent_code 错误: {e}")
        return None
    finally:
        db.close()

def get_agent_settings_as_dict(agent_code: str) -> Optional[dict]:
    """根據 agent_code 獲取 AgentSettings 並返回一個字典"""
    db = SessionLocal()
    try:
        agent_settings_obj = db.query(AgentSettings).filter(AgentSettings.agent_code == agent_code).first()
        if agent_settings_obj:
            # --- 核心轉換邏輯 ---
            # 將 SQLAlchemy 物件轉換為字典
            return {c.key: getattr(agent_settings_obj, c.key) for c in agent_settings_obj.__table__.columns}
        return None
    finally:
        db.close()

# --- API 端點 ---

@app.get("/")
async def root():
    """健康檢查端點"""
    return {
        "message": "AI Voice Chat API is running", 
        "status": "healthy",
        "models": {
            "edge_tts": True,
            "whisper": whisper_model is not None
        },
        "available_voices": len(available_voices)
    }

@app.get("/voices", response_model=list[VoiceInfo])
async def get_voices():
    """獲取可用語音列表"""
    return [
        VoiceInfo(
            name=voice['Name'],
            short_name=voice['ShortName'],
            gender=voice['Gender'],
            locale=voice['Locale']
        )
        for voice in available_voices
    ]

@app.options("/{path:path}")
async def options_handler(path: str):
    """處理 OPTIONS 請求"""
    return JSONResponse(content={"message": "OK"})

@app.post("/stt", response_model=STTResponse)
async def speech_to_text(audio: UploadFile = File(...)):
    """完整音頻轉文字 API"""
    logger.info(f"收到 STT 請求: {audio.filename}, 類型: {audio.content_type}")
    
    # 檢查檔案類型
    allowed_types = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/webm', 'audio/ogg']
    if audio.content_type not in allowed_types:
        logger.warning(f"不支援的音頻類型: {audio.content_type}")
        # 不嚴格限制，允許嘗試處理
    
    temp_filename = f"stt_{uuid.uuid4().hex}_{audio.filename}"
    temp_path = os.path.join(TEMP_DIR, temp_filename)
    
    try:
        # 保存上傳的音頻檔案
        with open(temp_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        logger.info(f"音頻檔案大小: {len(content)} bytes")
        
        # 檢查檔案大小
        if len(content) < 100:
            logger.warning("音頻檔案太小，可能是空檔案")
            return STTResponse(text="", confidence=0.0)
        
        # 轉錄音頻
        loop = asyncio.get_event_loop()
        transcription = await loop.run_in_executor(executor, transcribe_audio, temp_path)
        
        logger.info(f"轉錄結果: {transcription}")
        
        return STTResponse(
            text=transcription["text"],
            confidence=transcription["confidence"]
        )
        
    except Exception as e:
        logger.error(f"STT 處理錯誤: {e}")
        logger.error(traceback.format_exc())
        return STTResponse(text="", confidence=0.0)
    finally:
        # 清理臨時檔案
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"清理臨時檔案失敗: {e}")

@app.post("/stt-chunk", response_model=STTResponse)
async def speech_to_text_chunk(audio: UploadFile = File(...)):
    """實時音頻塊轉文字 API"""
    logger.info(f"收到 STT 塊請求: {audio.filename}, 類型: {audio.content_type}")
    
    temp_filename = f"chunk_{uuid.uuid4().hex}.webm"
    temp_path = os.path.join(TEMP_DIR, temp_filename)
    
    try:
        with open(temp_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        logger.info(f"音頻塊大小: {len(content)} bytes")
        
        # 如果音頻檔案太小，可能是靜音
        if len(content) < 1000:
            logger.info("音頻塊太小，跳過轉錄")
            return STTResponse(text="", confidence=0.0)
        
        # 轉錄音頻塊
        loop = asyncio.get_event_loop()
        transcription = await loop.run_in_executor(executor, transcribe_audio, temp_path)
        
        logger.info(f"塊轉錄結果: {transcription}")
        
        return STTResponse(
            text=transcription["text"],
            confidence=transcription["confidence"]
        )
        
    except Exception as e:
        logger.error(f"STT 塊處理錯誤: {e}")
        logger.error(traceback.format_exc())
        return STTResponse(text="", confidence=0.0)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"清理臨時檔案失敗: {e}")

# 創建對話管理器實例
conversation_manager = ConversationManager()

# 新增：根据 agent_code 获取 AgentSettings 的函数
def get_agent_settings(agent_code: str) -> Optional[dict]:
    """根据 agent_code 获取 AgentSettings 资料"""
    try:
        db = SessionLocal()
        agent_setting = db.query(AgentSettings).filter(
            AgentSettings.agent_code == agent_code
        ).first()
        
        if not agent_setting:
            logger.warning(f"找不到 agent_code: {agent_code} 的资料")
            return None
        
        return {
            "agent_code": agent_setting.agent_code,
            "gender": agent_setting.gender,
            "age": agent_setting.age,
            "med_info": agent_setting.med_info,
            "disease": agent_setting.disease,
            "med_complexity": agent_setting.med_complexity,
            "med_code": agent_setting.med_code,
            "special_status": agent_setting.special_status,
            "check_day": agent_setting.check_day,
            "check_time": agent_setting.check_time,
            "check_type": agent_setting.check_type,
            "low_cost_med": agent_setting.low_cost_med
        }
        
    except Exception as e:
        logger.error(f"获取 AgentSettings 错误: {e}")
        return None
    finally:
        db.close()

def get_agent_code_by_session(session_id: str) -> Optional[str]:
    """根据 session_id 获取 agent_code"""
    try:
        db = SessionLocal()
        session_user_map = db.query(SessionUserMap).filter(
            SessionUserMap.session_id == session_id
        ).first()
        
        if not session_user_map:
            logger.warning(f"找不到 session_id: {session_id} 的资料")
            return None
            
        return session_user_map.agent_code
        
    except Exception as e:
        logger.error(f"获取 agent_code 错误: {e}")
        return None
    finally:
        db.close()

@app.post("/chat", response_model=ChatResponse)
async def chat_api(req: ChatRequest):
    """主要聊天 API"""
    logger.info(f"收到聊天請求: {req.text[:50]}...")
    
    user_text = req.text.strip()
    #session_id 基本上會在進到聊天畫面時就會傳送給前端
    #這樣就可以直接接收session_id 就設在SessionUserMap吧
    session_id = req.session_id or str(uuid.uuid4())
    
    if not user_text:
        raise HTTPException(status_code=400, detail="訊息內容不能為空")
    
    try:
        # 儲存使用者訊息
        save_chat_message('user', user_text, session_id)
        
        # --- NEW (Requirement 2): 觸發非同步背景總結任務 ---
        # 這個檢查和任務創建不會阻塞後續的回應生成流程
        if await conversation_manager.should_summarize(session_id):
            logger.info(f"觸發 Session {session_id} 的背景任務...")
            # 使用 asyncio.create_task 將總結任務放到背景執行
            asyncio.create_task(conversation_manager.summarize_conversation(session_id))
            
            # 評分任務
            recent_chat_snippet = get_latest_chat_history_for_scoring(session_id, limit=6)
            if recent_chat_snippet:
                # 必須在背景任務中創建新的 DB session
                async def scoring_task():
                    db_task = SessionLocal()
                    try:
                        # 將完整的對話片段傳遞給評分服務
                        await scoring_service.process_user_inputs_for_scoring(session_id, recent_chat_snippet, db_task)
                    finally:
                        db_task.close()
                asyncio.create_task(scoring_task())
        
        # --- 主要流程繼續 ---
        
        agent_code = get_agent_code_by_session(session_id)
        if not agent_code:
            raise HTTPException(status_code=400, detail="无法获取 agent_code")
        
        agent_settings = get_agent_settings_as_dict(agent_code)
        if not agent_settings:
            raise HTTPException(status_code=404, detail=f"找不到 agent_code: {agent_code} 的设定")
        
        logger.info(f"取得agent settings: {agent_settings['gender']}...")
        
        # 獲取對話歷史或總結
        history = await conversation_manager.get_formatted_history(session_id, limit=15)
        
        # 組合 prompt
        prompt = build_optimized_prompt(user_text, history, agent_settings)
        
        logger.info(f"使用的 prompt 長度: {len(prompt)} 字符")
        
        # 生成 LLM 回應
        llm_response = await generate_llm_response(prompt)
               
        if llm_response.startswith("你："):
            llm_response = llm_response[5:].strip()
        
        logger.info(f"Patient 回應: {llm_response}")
        
        # 儲存 AI 回應
        save_chat_message('patient', llm_response, session_id, agent_code)
        
        # 生成 TTS 音頻
        audio_url = await generate_tts_audio(llm_response, req.voice)
        
        return ChatResponse(
            text=llm_response,
            audioUrl=audio_url,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"聊天處理錯誤: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"處理聊天請求時發生錯誤: {str(e)}")

@app.post("/chat/end/{session_id}")
async def end_chat_session(session_id: str = Path(..., description="要結束的對話 Session ID")):
    """
    標記一個對話 session 為已結束 (is_completed = True)。
    這個動作是評分的前提。
    """
    db = SessionLocal()
    try:
        session_map = db.query(SessionUserMap).filter(SessionUserMap.session_id == session_id).first()
        if not session_map:
            raise HTTPException(status_code=404, detail="指定的 Session 不存在")

        if session_map.is_completed:
            logger.info(f"Session {session_id} 已被標記為結束。")
            return {"status": "already_ended", "message": "Session was already marked as completed."}

        session_map.is_completed = True
        db.commit()
        logger.info(f"Session {session_id} 已成功標記為結束。")
        return {"status": "ended", "message": "Session marked as completed successfully."}
    except Exception as e:
        db.rollback()
        logger.error(f"標記 Session 結束時發生錯誤: {e}")
        raise HTTPException(status_code=500, detail="標記對話結束時發生內部錯誤。")
    finally:
        db.close()

@app.get("/audio/{audio_file}")
async def get_audio(audio_file: str):
    """獲取音頻檔案"""
    # 安全檢查
    if not audio_file.endswith(('.wav', '.mp3', '.ogg')):
        raise HTTPException(status_code=400, detail="不支援的音頻格式")
    
    audio_path = os.path.join(AUDIO_DIR, audio_file)
    
    if not os.path.exists(audio_path):
        logger.error(f"音頻檔案不存在: {audio_path}")
        raise HTTPException(status_code=404, detail="音頻檔案不存在")
    
    return FileResponse(
        audio_path, 
        media_type="audio/wav",
        headers={
            "Cache-Control": "public, max-age=3600",
            "Access-Control-Allow-Origin": "*"
        }
    )

@app.get("/health")
async def health_check():
    """詳細健康檢查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "edge_tts": True,
            "whisper_initialized": whisper_model is not None
        },
        "directories": {
            "audio_dir_exists": os.path.exists(AUDIO_DIR),
            "temp_dir_exists": os.path.exists(TEMP_DIR)
        },
        "available_voices": len(available_voices)
    }

@app.get("/agent_info", response_model=AgentInfo)
async def agent_info_find(
    level: str = Query(..., description="難度級別: 初級, 中級, 高級"),
    username: str = Query(..., description="使用者名稱")
):
    """
    根據難度級別隨機選擇並返回 AgentSettings 資料，同時創建 SessionUserMap 記錄
    """
    try:
        # 建立資料庫連接
        db = SessionLocal()
        
        # 根據難度級別映射到對應的字母前綴
        level_mapping = {
            "初級": "A",
            "中級": "B", 
            "高級": "C"
        }
        
        # 檢查難度級別是否有效
        if level not in level_mapping:
            raise HTTPException(
                status_code=400, 
                detail=f"無效的難度級別: {level}。請使用: 初級, 中級, 高級"
            )
        
        # 取得對應的字母前綴
        prefix = level_mapping[level]
        
        # 隨機選擇1-5之間的數字
        random_number = random.randint(1, 5)
        
        # 組合 agent_code
        agent_code = f"{prefix}{random_number}"
        
        logger.info(f"查詢 AgentSettings: level={level}, agent_code={agent_code}, username={username}")
        
        # 從資料庫查詢對應的 AgentSettings
        agent_setting = db.query(AgentSettings).filter(
            AgentSettings.agent_code == agent_code
        ).first()
        
        if not agent_setting:
            raise HTTPException(
                status_code=404, 
                detail=f"找不到 agent_code: {agent_code} 的資料"
            )
        
        
        # 組合回傳資料
        agent_info = AgentInfo(
            agent_code=agent_setting.agent_code,
            gender=agent_setting.gender or "",
            age=agent_setting.age or "",
            med_info=agent_setting.med_info or "",
            disease=agent_setting.disease or "",
            med_complexity=agent_setting.med_complexity or "",
            med_code=agent_setting.med_code or "",
            special_status=agent_setting.special_status or "",
            check_day=agent_setting.check_day,
            check_time=agent_setting.check_time or "",
            check_type=agent_setting.check_type or "",
            low_cost_med=agent_setting.low_cost_med or "",
            payment_fee=agent_setting.payment_fee or "",
            laxative_experience=agent_setting.laxative_experience or ""
        )
        
        logger.info(f"成功返回 AgentSettings 資料: {agent_code}")
        return agent_info
        
    except HTTPException:
        # 重新拋出 HTTP 異常
        raise
    except Exception as e:
        logger.error(f"查詢 AgentSettings 時發生錯誤: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"伺服器內部錯誤: {str(e)}"
        )
    finally:
        db.close()

# 新增：create_session API
@app.post("/create_session")
async def create_session(session_info: SessionInfo, db: Session = Depends(get_db)):
    """创建新的 session"""
    try:
        db = SessionLocal()
        
        # 检查是否已存在
        existing_session = db.query(SessionUserMap).filter(
            SessionUserMap.session_id == session_info.session_id
        ).first()
        
        if existing_session:
            logger.info(f"Session {session_info.session_id} 已存在")
            return {"message": "Session already exists", "session_id": session_info.session_id}
        
        # 创建新的 session 记录
        new_session = SessionUserMap(
            session_id=session_info.session_id,
            username=session_info.username,
            agent_code=session_info.agent_code,
            created_at=datetime.now()
        )
        
        db.add(new_session)
        db.commit()
        
        # --- 新增：觸發預計算 ---
        precompute_and_save_answers(session_info.session_id, session_info.agent_code, db)
        
        logger.info(f"创建新 session: {session_info.session_id}")
        
        return {
            "message": "Session created successfully",
            "session_id": session_info.session_id,
            "username": session_info.username,
            "agent_code": session_info.agent_code
        }
        
    except Exception as e:
        db.rollback() # 出錯時回滾
        logger.error(f"创建 session 错误: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"创建 session 失败: {str(e)}")
    finally:
        db.close()
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )