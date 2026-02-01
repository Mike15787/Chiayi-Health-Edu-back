# tests/voice_replay_tester.py
import sys
import os
import asyncio
import logging
import uuid
import re
from datetime import datetime
import whisper
import torch

# --- 新增音訊處理套件 ---
import soundfile as sf
import librosa
import noisereduce as nr
from pydub import AudioSegment
from pydub.silence import split_on_silence

# --- 1. 環境設定 (必須在 import databases 前設定) ---
# 強制設定為真人測試環境，以讀取 human_test.db
os.environ["APP_ENV"] = "human"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from databases import (
    init_database,
    SessionLocal,
    ChatLog,
    SessionUserMap,
    PrecomputedSessionAnswer,
    SessionInteractionLog,
    Scores,
    AgentSettings,
)
from module_manager import ModuleManager
from scoring_service_manager import ScoringServiceManager

# 設定 Log
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VoiceReplayTester")

# 全域變數
module_manager = ModuleManager()
scoring_service_manager = ScoringServiceManager()
whisper_model = None  # 延遲載入

# 指定音檔存放目錄 (相對於專案根目錄)
AUDIO_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio"
)

# 指定暫存處理後的音檔目錄
TEMP_AUDIO_DIR = os.path.join(AUDIO_DIR, "processed_temp")
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

def preprocess_audio(input_path: str) -> str:
    """
    音訊前處理：
    1. 降噪 (Noise Reduction)
    2. 消除靜音 (Silence Removal)
    回傳處理後的暫存檔案路徑。
    """
    try:
        # --- 1. 降噪處理 (使用 noisereduce) ---
        # 使用 librosa 讀取 (轉為 float32 array, sr=取樣率)
        data, rate = librosa.load(input_path, sr=None)

        # 假設音檔前 0.5 秒是背景噪音 (若是對話很滿，可改用 stationary=True)
        # prop_decrease=0.8 表示降低 80% 的噪音，避免人聲失真
        reduced_noise_data = nr.reduce_noise(
            y=data, sr=rate, prop_decrease=0.8, stationary=True
        )

        # 暫存降噪後的檔案
        temp_denoised_path = os.path.join(
            TEMP_AUDIO_DIR, f"denoised_{os.path.basename(input_path)}"
        )
        sf.write(temp_denoised_path, reduced_noise_data, rate)

        # --- 2. 消除靜音 (使用 pydub) ---
        # 讀取剛剛降噪後的檔案
        sound = AudioSegment.from_file(temp_denoised_path)

        # split_on_silence 參數說明:
        # min_silence_len: 靜音超過多少毫秒就切斷 (700ms)
        # silence_thresh: 低於多少分貝視為靜音 (比平均音量低 16dB)
        # keep_silence: 切斷後保留多少毫秒的靜音，讓語句連接比較自然 (200ms)
        dBFS = sound.dBFS
        chunks = split_on_silence(
            sound, min_silence_len=700, silence_thresh=dBFS - 16, keep_silence=200
        )

        if not chunks:
            # 如果切完沒東西(全都是靜音)，回傳 None
            logger.warning(
                f"音檔 {os.path.basename(input_path)} 經處理後判定為全靜音。"
            )
            return None

        # 將切開的非靜音片段重新接起來
        processed_sound = sum(chunks)

        # 如果處理後太短 (小於 0.5 秒)，通常是雜訊，直接丟棄
        if len(processed_sound) < 500:
            logger.warning(
                f"音檔 {os.path.basename(input_path)} 處理後過短 (<0.5s)，視為無效。"
            )
            return None

        # 匯出最終檔案
        final_path = os.path.join(
            TEMP_AUDIO_DIR, f"clean_{os.path.basename(input_path)}"
        )
        processed_sound.export(final_path, format="wav")

        # 清理中間檔案
        if os.path.exists(temp_denoised_path):
            os.remove(temp_denoised_path)

        return final_path

    except Exception as e:
        logger.error(f"音訊前處理失敗: {e}")
        # 如果處理失敗，回傳原始路徑嘗試辨識
        return input_path


def clean_repetitive_text(text: str) -> str:
    """清洗重複字元"""
    if not text:
        return ""
    text = re.sub(r"(.)\1{4,}", r"\1", text)
    text = re.sub(r"(.{2})\1{3,}", r"\1", text)
    return text


def load_whisper_model():
    """載入 Whisper 模型 (使用 GPU 若可用)"""
    global whisper_model
    if whisper_model is None:
        logger.info("正在載入 Whisper 模型 (可能需要一點時間)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 建議使用 'base' 或 'small'，若 VRAM 足夠可用 'medium' 效果更好
        whisper_model = whisper.load_model("small", device=device)
        logger.info(f"Whisper 模型載入完成 (Device: {device})")
    return whisper_model


def optimized_transcribe(audio_path: str) -> str:
    """
    優化版轉錄 Ver 3：前處理 -> Whisper -> 後處理 -> 語速檢核 (Sanity Check)
    """
    if not os.path.exists(audio_path):
        return "[音檔遺失]"

    # --- 步驟 A: 音訊前處理 ---
    clean_audio_path = preprocess_audio(audio_path)

    if clean_audio_path is None:
        return "[無效語音]"

    target_path = clean_audio_path

    # --- 步驟 B: 計算音檔長度 (用於語速檢核) ---
    try:
        # 取得音檔秒數
        y, sr = librosa.load(target_path, sr=None)
        duration_sec = librosa.get_duration(y=y, sr=sr)
    except:
        duration_sec = 0.0

    # --- 步驟 C: Whisper 轉錄 ---
    model = load_whisper_model()
    initial_prompt = (
        "以下是關於大腸鏡衛教的醫學對話。關鍵字包含:大腸鏡檢查、清腸劑、清腸藥、保可淨、"
        "低渣飲食、無渣流質飲食、瀉藥、麻醉、口服瀉藥錠劑、樂可舒。"
    )

    try:
        result = model.transcribe(
            target_path,
            language="zh",
            fp16=False,
            initial_prompt=initial_prompt,
            temperature=0.2,
            beam_size=5,
            best_of=5,
            condition_on_previous_text=False,
            compression_ratio_threshold=1.8,  # [再調低] 更嚴格，稍微有重複就重試
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
        )

        raw_text = result["text"].strip()

        # --- 步驟 D: 文字後處理 ---
        cleaned_text = clean_repetitive_text(raw_text)

        # 清理暫存檔
        if target_path != audio_path and os.path.exists(target_path):
            os.remove(target_path)

        # --- 步驟 E: 語速合理性檢核 (Sanity Check) ---
        # 如果音檔很短 (< 1秒) 卻產出很多字 (> 10字)，這絕對是幻覺
        if duration_sec > 0:
            chars_per_sec = len(cleaned_text) / duration_sec

            # 正常說話約 3-5 字/秒，快嘴頂多 8-9 字/秒。
            # 如果超過 15 字/秒，或是音檔小於 1 秒卻超過 8 個字，判定為幻覺。
            if chars_per_sec > 12.0:
                logger.warning(
                    f"偵測到語速異常 (幻覺): {duration_sec:.2f}秒 產出 {len(cleaned_text)}字 -> '{cleaned_text}'"
                )
                return "[背景雜音]"

            if duration_sec < 1.0 and len(cleaned_text) > 8:
                logger.warning(
                    f"短音檔幻覺: {duration_sec:.2f}秒 產出 '{cleaned_text}'"
                )
                return "[背景雜音]"

        # 最後檢查：如果清洗後長度大幅縮減 (代表原本大部分都是重複)，且剩下內容極短
        if len(raw_text) > 20 and len(cleaned_text) < 5:
            return "[幻覺過濾]"

        return cleaned_text

    except Exception as e:
        logger.error(f"轉錄失敗: {e}")
        return "[轉錄失敗]"


def get_next_replay_session_id(db, original_session_id: str) -> str:
    """產生帶有 _replay 後綴的 Session ID"""
    base_id = original_session_id.split("_replay")[0]
    pattern = f"{base_id}_replay%"
    similar_sessions = (
        db.query(SessionUserMap.session_id)
        .filter(SessionUserMap.session_id.like(pattern))
        .all()
    )

    count = len(similar_sessions) + 1
    return f"{base_id}_replay_{count}"


async def run_voice_replay_test(target_session_ids: list):
    """
    主流程：讀取舊 Session -> 重轉錄音檔 -> 建立新 Session -> 評分
    """
    db = SessionLocal()

    try:
        for original_session_id in target_session_ids:
            logger.info(f"\n{'='*60}")
            logger.info(f"開始處理 Session: {original_session_id}")
            logger.info(f"{'='*60}")

            # 1. 取得原始 Session 資料
            src_map = (
                db.query(SessionUserMap)
                .filter(SessionUserMap.session_id == original_session_id)
                .first()
            )
            if not src_map:
                logger.error(f"❌ 資料庫找不到 Session: {original_session_id}")
                continue

            # 2. 建立新的 Replay Session ID
            new_session_id = get_next_replay_session_id(db, original_session_id)
            logger.info(f"建立新的測試 Session ID: {new_session_id}")

            # 3. 複製 SessionUserMap
            new_map = SessionUserMap(
                session_id=new_session_id,
                username=f"{src_map.username}_Replay",
                agent_code=src_map.agent_code,
                module_id=src_map.module_id,
                created_at=datetime.now(),
                is_completed=False,
            )
            db.add(new_map)

            # 4. 複製 PrecomputedSessionAnswer (答案卷)
            src_pre = (
                db.query(PrecomputedSessionAnswer)
                .filter(PrecomputedSessionAnswer.session_id == original_session_id)
                .first()
            )
            if src_pre:
                new_pre = PrecomputedSessionAnswer(
                    session_id=new_session_id,
                    module_id=src_pre.module_id,
                    exam_day=src_pre.exam_day,
                    prev_1d=src_pre.prev_1d,
                    prev_2d=src_pre.prev_2d,
                    prev_3d=src_pre.prev_3d,
                    second_dose_time=src_pre.second_dose_time,
                    npo_start_time=src_pre.npo_start_time,
                    actual_check_type=src_pre.actual_check_type,
                )
                db.add(new_pre)

            # 5. 複製 UI 互動紀錄 (保留 UI 操作分數)
            src_interact = (
                db.query(SessionInteractionLog)
                .filter(SessionInteractionLog.session_id == original_session_id)
                .first()
            )
            if src_interact:
                new_interact = SessionInteractionLog(
                    session_id=new_session_id,
                    module_id=src_interact.module_id,
                    viewed_alltimes_ci=src_interact.viewed_alltimes_ci,
                    viewed_chiachi_med=src_interact.viewed_chiachi_med,
                    viewed_med_allergy=src_interact.viewed_med_allergy,
                    viewed_disease_diag=src_interact.viewed_disease_diag,
                    viewed_cloud_med=src_interact.viewed_cloud_med,
                )
                db.add(new_interact)

            db.commit()

            # 6. 處理對話紀錄 (重轉錄與重建)
            original_logs = (
                db.query(ChatLog)
                .filter(ChatLog.session_id == original_session_id)
                .order_by(ChatLog.time)
                .all()
            )

            chat_history_list = []  # 用於評分的格式

            print(f"\n--- 對話重製與轉錄對比 ---")
            print(f"{'角色':<6} | {'原始文字 (Old)':<30} | {'重新轉錄 (New STT)'}")
            print("-" * 80)

            for log in original_logs:
                new_text = log.text  # 預設使用舊文字 (針對 AI 回應)

                # 如果是 User 且有音檔，進行重轉錄
                if log.role == "user" and log.audio_filename:
                    full_audio_path = os.path.join(AUDIO_DIR, log.audio_filename)
                    transcribed_text = optimized_transcribe(full_audio_path)

                    # 比對顯示
                    print(f"{'User':<6} | {log.text[:28]:<30} | {transcribed_text}")
                    new_text = transcribed_text
                elif log.role == "user":
                    print(f"{'User':<6} | {log.text[:28]:<30} | [無音檔]")
                else:
                    # AI 的話直接保留
                    pass

                new_chat_log = ChatLog(
                    session_id=new_session_id,
                    user_id=log.user_id,
                    agent_code=log.agent_code,
                    module_id=log.module_id,
                    role=log.role,
                    text=new_text,
                    audio_filename=log.audio_filename,  # 保留音檔連結
                    time=log.time,  # 保留相對時間順序
                )

                # [修改 2] 加入並 Flush，這樣 new_chat_log.id 才會有值
                db.add(new_chat_log)
                db.flush()  # <--- 關鍵！這會產生 id，但還沒 commit

                # 準備給評分用的 snippet
                chat_history_list.append({"role": log.role, "message": new_text})

                # 模擬即時評分 (雖然是回放，但為了確保邏輯一致，我們批次送入 User 的話)
                if log.role == "user":
                    snippet = chat_history_list[-5:]  # 取最近5句
                    await scoring_service_manager.process_user_inputs_for_scoring(
                        new_session_id,
                        src_map.module_id,
                        snippet,
                        db,
                        chat_log_id=new_chat_log.id,
                    )

            db.commit()

            # 7. 計算最終分數
            logger.info("正在計算最終分數...")
            final_scores = await scoring_service_manager.calculate_final_scores(
                new_session_id, src_map.module_id, db
            )

            # 8. 寫入分數與完成狀態
            new_score_record = Scores(
                session_id=new_session_id,
                module_id=src_map.module_id,
                **{
                    k: v
                    for k, v in final_scores.items()
                    if k in Scores.__table__.columns.keys()
                },
            )
            db.merge(new_score_record)

            # 更新 Session Map
            updated_map = (
                db.query(SessionUserMap)
                .filter(SessionUserMap.session_id == new_session_id)
                .first()
            )
            updated_map.score = final_scores.get("total_score", "0")
            updated_map.is_completed = True

            db.commit()

            # 9. 比較新舊分數
            old_score_record = (
                db.query(Scores)
                .filter(Scores.session_id == original_session_id)
                .first()
            )
            old_total = old_score_record.total_score if old_score_record else "N/A"
            new_total = final_scores.get("total_score", "N/A")

            print(f"\n📊 分數比較 (Agent: {src_map.agent_code})")
            print(f"   原始分數: {old_total}")
            print(f"   重測分數: {new_total}")
            print(f"   詳細結果請見資料庫 Session: {new_session_id}")

    except Exception as e:
        logger.error(f"測試過程發生錯誤: {e}", exc_info=True)
    finally:
        db.close()


if __name__ == "__main__":
    # 目標 Session IDs
    target_sessions = [
        "session_1769949647951_b0cin8d6b",
        "session_1769948170673_m4nq4emf1",
    ]

    print(f"🚀 啟動語音回放測試 (Voice Replay Tester)")
    print(f"📂 資料庫來源: human_test.db")
    print(f"🎯 目標 Session 數量: {len(target_sessions)}")

    # 初始化資料庫連接
    init_database()

    asyncio.run(run_voice_replay_test(target_sessions))
