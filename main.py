# main.py

from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Query, Path, Depends, Form
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
import Scoring

from utils import generate_llm_response

# 導入資料庫相關模組
from databases import (
    get_conversation_history, 
    save_chat_message, 
    get_db,
    save_db_conversation_summary,
    get_latest_conversation_summary,
    get_user_message_count,
    SessionLocal,
    ChatLog,
    AnswerLog,
    UserLogin,
    Scores,
    AgentSettings,
    SessionUserMap,
    ScoringAttributionLog,
    get_latest_user_messages,
    get_latest_chat_history_for_scoring,
    PrecomputedSessionAnswer,
    SessionInteractionLog,
    get_module_id_by_session
)

from agentset import insert_agent_data
from module_manager import ModuleManager # 新增
from scoring_service_manager import ScoringServiceManager # 新增

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

# --- NEW: Module and Scoring Managers ---
# 現在是利用module_maneger去管理所有的教育模組
# 他會動態載入所需的模組
# 此模組只會在lifespan被實例化一次
module_manager: ModuleManager = None
scoring_service_manager: ScoringServiceManager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理"""
    global module_manager, scoring_service_manager # 聲明為全域變數
    
    # 啟動時執行
    logger.info("AI Voice Chat API 啟動中...")
    
    logger.info("正在檢查並插入 Agent 設定資料...")
    insert_agent_data()
    logger.info("Agent 設定資料檢查完畢。")
    
    # 初始化模型
    await init_models()
    
    # 創建無聲音頻檔案
    # 用途是避免當語音生成錯誤時導致整體系統無法運作
    create_silence_audio()
    
    # --- NEW: Initialize Module Manager and Scoring Service Manager ---
    logger.info("Initializing ModuleManager...")
    module_manager = ModuleManager()
    logger.info("ModuleManager initialized.")
    
    logger.info("Initializing ScoringServiceManager...")
    scoring_service_manager = ScoringServiceManager()
    Scoring.scoring_service = scoring_service_manager # 將實例賦值給 Scoring 模組的變數
    logger.info("ScoringServiceManager initialized and linked to Scoring module.")
    
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
        "http://127.0.0.1:5173",
        "https://3da8f686e88f.ngrok-free.app  "
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(history_router)
app.include_router(Scoring.score_router) # 引用 Scoring 模組中的 router


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
    session_id: str

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
    
# 修改 SessionInfo 模型，增加 module_id
class SessionInfo(BaseModel):
    session_id: str
    username: str
    agent_code: str
    module_id: str = "colonoscopy_bowklean" # 預設為現有模組

# --- New Pydantic Model for Ending Chat ---
class EndChatRequest(BaseModel):
    viewed_alltimes_ci: bool = False # 歷次清腸資訊
    viewed_chiachi_med: bool = False # 本院用藥
    viewed_med_allergy: bool = False # 藥物過敏史
    viewed_disease_diag: bool = False # 疾病診斷
    viewed_cloud_med: bool = False   # 雲端藥歷

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
    
   
    def save_conversation_summary(self, session_id: str, summary: str, module_id: str):
        """
        儲存對話總結到資料庫。
        """
        try:
            # --- MODIFIED (Requirement 3): 呼叫新的資料庫函數 ---
            save_db_conversation_summary(session_id, summary, module_id)
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
            await asyncio.sleep(3)
            
            logger.info(f"開始為 Session {session_id} 進行對話總結...")
            
            # 獲取模組 ID
            module_id = get_module_id_by_session(session_id)
            
            if not module_id:
                logger.error(f"Session {session_id} 沒有綁定模組 ID，無法進行總結。")
                return
            
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
                self.save_conversation_summary(session_id, summary, module_id)
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
            return "silence.wav"
        
        logger.info(f"edge-tts 音頻生成成功: {audio_filename} (語音: {selected_voice})")
        return audio_filename
        
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




def transcribe_audio(audio_path: str) -> dict:
    """使用 Whisper 轉錄音頻"""
    if whisper_model is None:
        return {"text": "", "confidence": 0.0, "language": "zh"}
    
    prompt_text = (
        "以下是關於大腸鏡衛教的醫學對話。關鍵字包含:大腸鏡檢查、清腸劑、清腸藥、保可淨"
        "低渣飲食、無渣流質飲食、瀉藥、麻醉、口服瀉藥錠劑、樂可舒"
        "抗凝血劑藥物、抗血小板藥物、降血糖藥物、高血壓藥物、抗癲癇藥物"
    )
    
    try:
        result = whisper_model.transcribe(
            audio_path, 
            language="zh",
            fp16=False,
            initial_prompt=prompt_text, # <--- 關鍵修改在這裡
            temperature=0.2, # 稍微降低隨機性，讓它更保守地依賴 prompt
            beam_size=5      # 增加搜索廣度，提高準確率
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
async def speech_to_text(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None) # <--- 接收 session_id
):
    """
    完整音頻轉文字 API。
    新職責：
    1. 接收音訊和可選的 session_id。
    2. 如果沒有 session_id，則創建一個新的。
    3. 永久儲存音訊檔案。
    4. 進行 STT。
    5. 將使用者的訊息（文字 + 音檔名）存入資料庫。
    6. 回傳辨識文字和 session_id 給前端。
    """
    logger.info(f"收到 STT 請求: {audio.filename}, Session: {session_id or 'New'}")
    
    # 1. 確定 session_id
    current_session_id = session_id
    
    # 2. 永久儲存音訊檔案
    saved_audio_filename = f"user_{current_session_id}_{uuid.uuid4().hex}.wav"
    saved_audio_path = os.path.join(AUDIO_DIR, saved_audio_filename)

    try:
        with open(saved_audio_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        if len(content) < 100:
            logger.warning("音頻檔案太小，可能是空檔案")
            return STTResponse(text="", confidence=0.0, session_id=current_session_id)
        
        # 3. 轉錄音頻
        loop = asyncio.get_event_loop()
        transcription = await loop.run_in_executor(executor, transcribe_audio, saved_audio_path)
        user_text = transcription["text"].strip()
        
        logger.info(f"轉錄結果 for session {current_session_id}: {user_text}")
        
        # 4. 從資料庫獲取 agent_code 和 module_id
        session_map = SessionLocal().query(SessionUserMap).filter(SessionUserMap.session_id == current_session_id).first()
        if not session_map:
             # 如果是第一次請求，可能還沒有 session_map，這是一個問題。
             # 前端必須在進入聊天頁面時就呼叫 /create_session 來確保 session_map 存在。
             # 這裡我們假設前端會先創建 session。
             raise HTTPException(status_code=400, detail=f"Session {current_session_id} not found. Please create a session first.")
        
        # 5. 儲存使用者訊息到 ChatLog
        if user_text:
            user_message_log = save_chat_message(
                role='user',
                text=user_text,
                session_id=current_session_id,
                agent_code=session_map.agent_code,
                module_id=session_map.module_id,
                audio_filename=saved_audio_filename, # 儲存音檔名
                return_obj=True
            )
            # 觸發背景評分
            asyncio.create_task(score_utterance_task(user_message_log.id, current_session_id))
        
        return STTResponse(
            text=user_text,
            confidence=transcription["confidence"],
            session_id=current_session_id # <--- 回傳 session_id
        )
        
    except Exception as e:
        logger.error(f"STT 處理錯誤: {e}")
        logger.error(traceback.format_exc())
        # 即使出錯，也回傳 session_id 讓前端可以繼續
        return STTResponse(text="", confidence=0.0, session_id=current_session_id)
    # 注意：這裡不再清理 TEMP_DIR 中的檔案，因為我們直接存到 AUDIO_DIR

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

# --- 新增：背景評分任務 ---
async def score_utterance_task(chat_log_id: int, session_id: str):
    """
    在背景運行的、針對單句話的評分任務。
    """
    await asyncio.sleep(2) # 稍微延遲，避免阻塞主要回應
    
    logger.info(f"[{session_id}] Starting background scoring task for chat_log_id: {chat_log_id}")
    db_task = SessionLocal()
    try:
        # 從 SessionUserMap 獲取 module_id
        session_map = db_task.query(SessionUserMap).filter(SessionUserMap.session_id == session_id).first()
        if not session_map:
            logger.error(f"[{session_id}] SessionUserMap not found for scoring task.")
            return
        module_id = session_map.module_id
        
        # 獲取當前這句話和它的上下文（例如，包括前4句話）
        recent_history = get_latest_chat_history_for_scoring(session_id, limit=5)
        if not recent_history:
            return

        # 呼叫評分服務，獲取新達成的評分項
        newly_passed_item_ids = await scoring_service_manager.process_user_inputs_for_scoring(
            session_id, module_id, recent_history, db_task
        )

        # 如果有新達成的項目，寫入歸因表 (ScoringAttributionLog)
        if newly_passed_item_ids:
            for item_id in newly_passed_item_ids:
                attribution_entry = ScoringAttributionLog(
                    session_id=session_id,
                    chat_log_id=chat_log_id,
                    scoring_item_id=item_id,
                    module_id=module_id
                )
                # 使用 merge 避免因時序問題導致的重複寫入錯誤
                db_task.merge(attribution_entry)
            db_task.commit()
            logger.info(f"[{session_id}] Attributed chat_log {chat_log_id} to items: {newly_passed_item_ids}")

    except Exception as e:
        logger.error(f"Background scoring task failed for chat_log_id {chat_log_id}: {e}")
        logger.error(traceback.format_exc())
        db_task.rollback()
    finally:
        db_task.close()

@app.post("/chat", response_model=ChatResponse)
async def chat_api(req: ChatRequest):
    """
    主要聊天 API。
    新職責：接收文字和 session_id，純粹地生成並回傳 AI 的下一句話。
    不再儲存使用者訊息（已由 /stt 處理）。
    """
    logger.info(f"收到聊天請求 for session {req.session_id}: {req.text[:50]}...")
    
    user_text = req.text.strip()
    session_id = req.session_id
    
    if not user_text or not session_id:
        raise HTTPException(status_code=400, detail="訊息內容和 session_id 皆不能為空")
    
    try:
        # 從資料庫獲取 session_id 對應的模組 ID 和 agent_code
        session_map = SessionLocal().query(SessionUserMap).filter(SessionUserMap.session_id == session_id).first()
        if not session_map:
            raise HTTPException(status_code=400, detail=f"Session {session_id} not found.")
        
        module_id = session_map.module_id
        agent_code = session_map.agent_code
        
        # --- 移除儲存使用者訊息的邏輯 ---
        # user_message_log = save_chat_message(...) <--- 刪除此行

        # 觸發背景總結任務 (這部分邏輯不變)
        if await conversation_manager.should_summarize(session_id):
            logger.info(f"觸發 Session {session_id} 的背景總結任務...")
            asyncio.create_task(conversation_manager.summarize_conversation(session_id))
        
        # --- 主要流程繼續 ---
        agent_settings = get_agent_settings_as_dict(agent_code)
        if not agent_settings:
            raise HTTPException(status_code=404, detail=f"找不到 agent_code: {agent_code} 的设定")
        
        history = await conversation_manager.get_formatted_history(session_id, limit=15)
        precomputed_data_obj = SessionLocal().query(PrecomputedSessionAnswer).filter(PrecomputedSessionAnswer.session_id == session_id).first()
        precomputed_data_dict = {c.key: getattr(precomputed_data_obj, c.key) for c in precomputed_data_obj.__table__.columns} if precomputed_data_obj else None
        
        patient_agent_builder = module_manager.get_patient_agent_builder(module_id)
        prompt = patient_agent_builder(user_text, history, agent_settings, precomputed_data_dict)
        
        module_config = module_manager.get_module_config(module_id)
        llm_response = await generate_llm_response(prompt, model_name=module_config.PATIENT_AGENT_MODEL_NAME)
               
        if llm_response.startswith("你："):
            llm_response = llm_response[5:].strip()
        
        logger.info(f"Patient 回應: {llm_response} for module {module_id}")
        
        # --- 修改儲存 AI 回應的邏輯 ---
        # 1. 先生成 TTS 音頻，以獲取檔名
        tts_audio_filename = await generate_tts_audio(llm_response, req.voice)
        
        # 2. 儲存 AI 回應到資料庫，同時傳入文字和音檔名
        save_chat_message(
            'patient', 
            llm_response, 
            session_id, 
            agent_code, 
            module_id,
            audio_filename=tts_audio_filename
        )
        
        # 3. 組合完整的 URL 給前端
        audio_url = f"/audio/{tts_audio_filename}"
        
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
async def end_chat_session(
    session_id: str = Path(..., description="要結束的對話 Session ID"),
    req: EndChatRequest = None 
):
    """
    標記一個對話 session 為已結束 (is_completed = True)。
    同時接收前端傳來的檢閱藥歷互動紀錄，並儲存到資料庫供後續評分使用。
    """
    db = SessionLocal()
    try:
        session_map = db.query(SessionUserMap).filter(SessionUserMap.session_id == session_id).first()
        if not session_map:
            raise HTTPException(status_code=404, detail="指定的 Session 不存在")

        # 1. 標記完成
        if not session_map.is_completed:
            session_map.is_completed = True
        
        # 2. 儲存 UI 互動紀錄 (不再這裡計算分數)
        if req:
            logger.info(f"Session {session_id} 接收到 UI 互動紀錄，正在儲存...")

            # 檢查是否已有紀錄，有則更新，無則新增
            interaction_record = db.query(SessionInteractionLog).filter(SessionInteractionLog.session_id == session_id).first()
            
            if not interaction_record:
                interaction_record = SessionInteractionLog(
                    session_id=session_id,
                    module_id=session_map.module_id,
                    viewed_alltimes_ci=req.viewed_alltimes_ci,
                    viewed_chiachi_med=req.viewed_chiachi_med,
                    viewed_med_allergy=req.viewed_med_allergy,
                    viewed_disease_diag=req.viewed_disease_diag,
                    viewed_cloud_med=req.viewed_cloud_med
                )
                db.add(interaction_record)
            else:
                # 更新現有紀錄
                interaction_record.viewed_alltimes_ci = req.viewed_alltimes_ci
                interaction_record.viewed_chiachi_med = req.viewed_chiachi_med
                interaction_record.viewed_med_allergy = req.viewed_med_allergy
                interaction_record.viewed_disease_diag = req.viewed_disease_diag
                interaction_record.viewed_cloud_med = req.viewed_cloud_med
        
        db.commit()
        logger.info(f"Session {session_id} 已成功標記為結束，UI 互動紀錄已儲存。")
        return {"status": "ended", "message": "Session marked as completed and interaction logged."}
        
    except Exception as e:
        db.rollback()
        logger.error(f"標記 Session 結束時發生錯誤: {e}")
        logger.error(traceback.format_exc())
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
    edu_type: str = Query(..., description="衛教類型: 清腸劑衛教, Warfarin抗凝衛教, DOAC抗凝衛教, 慢性病衛教"),
    chemical_type: str = Query(..., description="藥物種類: 保可淨(BowKlean), 腸見淨(Klean-prep)等"),
    level: str = Query(..., description="難度級別: 初級, 中級, 高級"),
    username: str = Query(..., description="使用者名稱")
):
    """
    根據難度級別隨機選擇並返回 AgentSettings 資料，同時創建 SessionUserMap 記錄
    Note: 此端點不再創建 SessionUserMap。該操作會在 `/create_session` 執行。
    """

    """
    找時間問嘉基這個
    目前用藥 印象中他們是希望從每個agent的藥庫裡面抽藥品種類出來 依照每個agent的藥品數量
    範例1 A1 有5種藥 -> 簡單模式只抽4種以下 -> 從1~4取隨機數量的藥品出來
    範例2 C2 有14種藥 -> 困難模式要抽10種以上 -> 從10~14取隨機數量的藥品出來
    範例3 B5 有7種藥 -> 普通模式要大於等於5 小於等於9 -> 從5~9 取隨機數量的藥品出來
    範例4 B4 有10種藥 -> 普通模式要大於等於5 小於等於9 -> 從5~9 取隨機數量的藥品出來
    藥品數量不會少於每種模式的最低需求藥品數量
    Note: 此端點不再創建 SessionUserMap。該操作會在 `/create_session` 執行。
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
        
        # 處理agent_code的號碼
        prefix = level_mapping[level]# 取得對應的字母前綴
        random_number = random.randint(1, 5) # 隨機選擇1-5之間的數字
        agent_code = f"{prefix}{random_number}" # 組合 agent_code
        
        logger.info(f"查詢 AgentSettings: level={level}, agent_code={agent_code}, username={username}")
        
        # 從資料庫查詢對應的 AgentSettings
        # 需要多加 對應的edu_type跟chemical_type
        agent_setting = db.query(AgentSettings).filter(
            AgentSettings.edu_type == edu_type,
            AgentSettings.chemical_type == chemical_type,
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
            logger.info(f"Session {session_info.session_id} 已存在，模組: {existing_session.module_id}")
            return {"message": "Session already exists", "session_id": session_info.session_id, "module_id": existing_session.module_id}
        
        # 檢查模組是否存在
        if session_info.module_id not in module_manager.modules:
            raise HTTPException(status_code=400, detail=f"指定的模組 '{session_info.module_id}' 不存在或未載入。")
        
        # 创建新的 session 记录
        new_session = SessionUserMap(
            session_id=session_info.session_id,
            username=session_info.username,
            agent_code=session_info.agent_code,
            module_id=session_info.module_id, # 新增 module_id
            created_at=datetime.now()
        )
        
        db.add(new_session)
        db.commit()
        
        # --- 新增：觸發預計算 ---
        precomputation_func = module_manager.get_precomputation_performer(session_info.module_id)
        await precomputation_func(session_info.session_id, session_info.agent_code, db)
        
        logger.info(f"创建新 session: {session_info.session_id}, 模組: {session_info.module_id}")
        
        return {
            "message": "Session created successfully",
            "session_id": session_info.session_id,
            "username": session_info.username,
            "agent_code": session_info.agent_code,
            "module_id": session_info.module_id # 回傳 module_id
        }
    except HTTPException:
        raise # 重新拋出已處理的 HTTP 異常    
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