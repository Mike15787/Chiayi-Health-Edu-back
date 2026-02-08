from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Text,
    Boolean,
    UniqueConstraint,
    ForeignKey,
    inspect,
    text,
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime, timezone, timedelta
import uuid
import secrets
from typing import Optional
import argparse
import os
import json

APP_ENV = os.getenv("APP_ENV", "dev")

if APP_ENV == "auto":
    DB_FILENAME = "auto_test.db"
    print(f"🤖 [系統提示] 模式：【自動化測試 (Auto Test)】 | 資料庫: {DB_FILENAME}")
elif APP_ENV == "human":
    DB_FILENAME = "human_test.db"
    print(f"👥 [系統提示] 模式：【真人測試 (Human/Ngrok)】 | 資料庫: {DB_FILENAME}")
else:
    # 預設 dev
    DB_FILENAME = "chatlog.db"
    print(f"🔧 [系統提示] 模式：【開發環境 (Dev)】 | 資料庫: {DB_FILENAME}")

# --- 資料庫設定 ---
DATABASE_URL = f"sqlite:///{DB_FILENAME}"
engine = create_engine(DATABASE_URL, echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

# 要再多紀錄時間
# 衛教時間 系統回饋時間 閱讀回饋時間

# 學習次數 學習成果 這個應該是記錄在個人資料中
# class PersonalData (username, uid,)
# class LearningRecord(uid, edu_type)
# 學習次數是會更新的東西 更新時間大概是使用者確定完成了一個session 之後的事
# 大概也需要分類

TAIPEI_TZ = timezone(timedelta(hours=8))

def get_taipei_now():
    """取得目前的台灣時間"""
    return datetime.now(TAIPEI_TZ)

# --- 資料庫模型 ---
class UserLogin(Base):
    __tablename__ = "user_login"

    id = Column(Integer, primary_key=True, autoincrement=True)  # 主鍵，自動增加
    username = Column(String(32), nullable=False, unique=True, comment="使用者名稱")
    uid = Column(String(36), unique=True, nullable=False, comment="用戶唯一ID (UUID)")
    email = Column(String(128), unique=True, nullable=False, comment="電子郵件")
    password = Column(String(128), nullable=False, comment="加密後的密碼")
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="建立時間",
    )
    login_stat = Column(
        Boolean, default=False, nullable=False, comment="登入狀態(True=已登入)"
    )

    def __repr__(self):
        return f"<UserLogin(username={self.username}, email={self.email}, login_stat={self.login_stat})>"


class ChatLog(Base):
    __tablename__ = "chatlog"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=True)
    session_id = Column(String, default=lambda: str(uuid.uuid4()))
    agent_code = Column(String, nullable=False)
    module_id = Column(
        String, nullable=False, default="default_module", comment="衛教模組ID"
    )  # 新增
    role = Column(String, nullable=False)
    text = Column(Text, nullable=False)
    audio_filename = Column(
        String(255), nullable=True, comment="使用者音檔以及AI回應音檔檔名"
    )
    time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    # [新增] 用來存 JSON 格式的除錯資訊 (例如：向量搜尋結果)
    # 在正式環境可以保持為 NULL
    debug_info = Column(Text, nullable=True, comment="JSON格式的除錯資訊")


# answerlog的用處就是在對話過程中 能夠知道哪些需要LLM評分的項目已經被評過分
# 以及在scoring logic結尾的部分去計算分數用的
class AnswerLog(Base):
    __tablename__ = "answer_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    module_id = Column(
        String, nullable=False, default="default_module", comment="衛教模組ID"
    )  # 新增
    scoring_item_id = Column(
        String, nullable=False, comment="來自 scoring_criteria.json 的 id"
    )
    score = Column(Integer, nullable=False, comment="評分結果 (1=達成, 0=未達成)")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    # 確保同一個 session 的同一個項目只有一筆紀錄
    __table_args__ = (
        UniqueConstraint("session_id", "scoring_item_id", name="_session_item_uc"),
    )


class AgentSettings(Base):
    __tablename__ = "agent_settings"
    # 10/27 新增欄位 edu_type 清腸、W抗凝、D抗凝、慢性
    #       新增欄位 chemical_type 保可淨、腸見淨、其他抗凝藥劑
    id = Column(Integer, primary_key=True, autoincrement=True)
    edu_type = Column(String)
    chemical_type = Column(String)
    agent_code = Column(String, unique=True, nullable=False)  # 例如：A1、B1
    gender = Column(String)
    age = Column(String)
    med_info = Column(Text)  # 藥物、用法
    disease = Column(Text)
    med_complexity = Column(String)
    med_code = Column(String)
    special_status = Column(Text)
    check_day = Column(Integer)
    check_time = Column(String)
    check_type = Column(String)
    low_cost_med = Column(String)
    payment_fee = Column(String, default="未提供", comment="繳交費用")
    laxative_experience = Column(String, default="未提供", comment="過去瀉藥經驗")
    drug_combination = Column(String, comment="組合一或組合二")
    low_residue_meal = Column(String, comment="低渣飲食代餐無或有")


# 修改 SessionUserMap 模型，添加評分功能
class SessionUserMap(Base):
    __tablename__ = "sessionid_user"
    session_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(32), nullable=False, comment="使用者名稱")
    agent_code = Column(String, nullable=False, comment="案例代碼")
    module_id = Column(
        String, nullable=False, default="default_module", comment="衛教模組ID"
    )  # 新增
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="對話建立時間",
    )
    score = Column(String(16), nullable=True, comment="對話評分(字串格式)")
    is_completed = Column(Boolean, default=False, comment="是否完成對話")


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(128), nullable=False, comment="EMAIL")
    token = Column(String(64), unique=True, nullable=False, comment="重設Token")
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="創建時間",
    )
    expires_at = Column(DateTime, nullable=False, comment="過期時間")
    is_used = Column(Boolean, default=False, nullable=False, comment="是否已使用")

    def __repr__(self):
        return f"<PasswordResetToken(email={self.email}, token={self.token[:8]}..., is_used={self.is_used})>"


class Scores(Base):
    __tablename__ = "sessionid_score"
    session_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    module_id = Column(String, nullable=False, default="default_module", comment="衛教模組ID")
    total_score = Column(String, nullable=False, comment="總分")
    review_med_history_score = Column(String, nullable=False, comment="檢閱藥歷分數")
    medical_interview_score = Column(String, nullable=False, comment="醫療面談分數")
    counseling_edu_score = Column(String, nullable=False, comment="諮商衛教分數")
    organization_efficiency_score = Column(String, nullable=False, comment="組織效率分數")
    clinical_judgment_score = Column(String, nullable=False, comment="臨床判斷分數")
    humanitarian_score = Column(String, nullable=False, comment="人道專業分數")
    overall_clinical_skills_score = Column(String, nullable=False, comment="整體臨床技能分數")
    # --- 紀錄模型的欄位 ---
    scoring_model = Column(String, nullable=True, comment="使用的評分LLM模型名稱")


class Summary(Base):
    __tablename__ = "sessionid_summary"
    session_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    module_id = Column(String, nullable=False, default="default_module", comment="衛教模組ID")
    total_summary = Column(String, nullable=False, comment="總結")
    review_med_history_summary = Column(String, nullable=False, comment="檢閱藥歷總結")
    medical_interview_summary = Column(String, nullable=False, comment="醫療面談總結")
    counseling_edu_summary = Column(String, nullable=False, comment="諮商衛教總結")
    organization_efficiency_summary = Column(String, nullable=False, comment="組織效率總結")
    clinical_judgment_summary = Column(String, nullable=False, comment="臨床判斷總結")
    humanitarian_summary = Column(String, nullable=False, default="無", comment="人道專業總結")
    organization_efficiency_summary = Column(String, nullable=False, default="無", comment="組織效率總結")
    overall_clinical_skills_summary = Column(String, nullable=False, default="無", comment="整體臨床技能總結")

class ConversationSummary(Base):
    __tablename__ = "conversation_summary"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)  # 加上索引以加快查詢
    module_id = Column(String, nullable=False, default="default_module", comment="衛教模組ID")
    summary = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# 新增一個table 用來存放預先計算好的這一個session的答案
class PrecomputedSessionAnswer(Base):
    __tablename__ = "precomputed_session_ans"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, unique=True, index=True)  # 加上索引以加快查詢
    module_id = Column(String, nullable=False, default="default_module", comment="衛教模組ID")
    exam_day = Column(String, nullable=False, comment="檢查日")
    prev_1d = Column(String, nullable=False, comment="檢查前一天")
    prev_2d = Column(String, nullable=False, comment="檢查前兩天")
    prev_3d = Column(String, nullable=False, comment="檢查前三天")  # ?月?號
    second_dose_time = Column(String, nullable=False, comment="第二包藥劑服用時間")  # 凌晨?點
    npo_start_time = Column(String, nullable=False, comment="禁水時間")  # 上午?點
    actual_check_type = Column(String, nullable=False, comment="實際檢查類型(一般/無痛)")


class ScoringPromptLog(Base):
    __tablename__ = "scoring_prompt_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    module_id = Column(String, nullable=False, default="default_module", comment="衛教模組ID")
    scoring_item_id = Column(String, nullable=False, index=True)
    prompt_text = Column(Text, nullable=False, comment="發送給 LLM 的完整 Prompt")
    llm_response = Column(Text, nullable=False, comment="LLM 返回的原始回應")
    final_score = Column(Integer, nullable=False, comment="解析後的最終分數 (0 或 1)")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), comment="紀錄建立時間")

    # [新增] 關聯 ID
    chat_log_id = Column(Integer, ForeignKey("chatlog.id"), nullable=True, index=True)


class ScoringAttributionLog(Base):
    __tablename__ = "scoring_attribution_log"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, index=True)
    chat_log_id = Column(Integer, ForeignKey("chatlog.id"))  # <-- 連結到觸發評分的那句話
    module_id = Column(String, nullable=False, default="default_module", comment="衛教模組ID")
    scoring_item_id = Column(String, index=True)  # <-- 連結到 scoring_criteria.json 的 id
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # 確保每個句子對應的評分項是唯一的
    __table_args__ = (
        UniqueConstraint("chat_log_id", "scoring_item_id", name="_chat_item_uc"),
    )


class SessionInteractionLog(Base):
    __tablename__ = "session_interaction_log"

    session_id = Column(String, primary_key=True, comment="對應的 Session ID")
    module_id = Column(String, nullable=False, default="default_module", comment="衛教模組ID")

    # 記錄各個按鈕是否被點擊 (True/False)
    viewed_alltimes_ci = Column(Boolean, default=False, comment="是否檢閱歷次清腸資訊")
    viewed_chiachi_med = Column(Boolean, default=False, comment="是否檢閱本院用藥")
    viewed_med_allergy = Column(Boolean, default=False, comment="是否檢閱藥物過敏史")
    viewed_disease_diag = Column(Boolean, default=False, comment="是否檢閱疾病診斷")
    viewed_cloud_med = Column(Boolean, default=False, comment="是否檢閱雲端藥歷")

    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


# --- 資料庫初始化函數 ---
def init_database():
    """初始化資料庫，創建所有表格"""
    Base.metadata.create_all(engine)
    sync_db_schema(engine)


def get_db():
    """獲取資料庫連接（用於依賴注入）"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- 資料庫操作輔助函數 ---
def get_conversation_history(session_id: str = None, limit: int = 20) -> str:
    """獲取對話歷史"""
    db = SessionLocal()
    try:
        query = db.query(ChatLog)

        if session_id:
            query = query.filter(ChatLog.session_id == session_id)

        history_rows = query.order_by(ChatLog.id.desc()).limit(limit).all()

        history_txt = ""
        for row in reversed(history_rows):
            prefix = "你：" if row.role == "user" else "ViVi："
            history_txt += f"{prefix}{row.text}\n"

        return history_txt
    except Exception as e:
        print(f"獲取對話歷史錯誤: {e}")
        return ""
    finally:
        db.close()


def get_latest_user_messages(session_id: str, count: int = 2) -> list[str]:
    """獲取指定 session 中使用者最新的幾條訊息"""
    db = SessionLocal()
    try:
        messages = (
            db.query(ChatLog.text)
            .filter(ChatLog.session_id == session_id, ChatLog.role == "user")
            .order_by(ChatLog.id.desc())
            .limit(count)
            .all()
        )
        # all() returns a list of tuples, e.g., [('message1',), ('message2',)]
        # We need to extract the first element of each tuple and reverse the list
        # so the messages are in chronological order.
        return [msg[0] for msg in reversed(messages)]
    except Exception as e:
        print(f"獲取使用者最新訊息錯誤: {e}")
        return []
    finally:
        db.close()


def get_latest_chat_history_for_scoring(session_id: str, limit: int = 4) -> list[dict]:
    """
    獲取指定 session 的最新對話歷史，包含 user 和 patient 的訊息，用於評分。
    返回一個字典列表，格式為 [{'role': 'user', 'message': '...'}, ...]。
    """
    db = SessionLocal()
    try:
        # 查詢最新的 limit 筆記錄
        logs = (
            db.query(ChatLog.role, ChatLog.text)
            .filter(ChatLog.session_id == session_id)
            .order_by(ChatLog.time.desc())
            .limit(limit)
            .all()
        )

        # 結果是從最新到最舊的，我們需要反轉它以保持時間順序
        history = [
            {"role": role, "message": message} for role, message in reversed(logs)
        ]
        return history
    finally:
        db.close()


def save_chat_message(
    role: str,
    text: str,
    session_id: str = None,
    agent_code: str = "default",
    module_id: str = "default_module",
    return_obj: bool = False,
    audio_filename: Optional[str] = None,
):
    """保存聊天訊息到資料庫"""
    db = SessionLocal()
    try:
        if session_id is None:
            session_id = str(uuid.uuid4())

        chat_log = ChatLog(
            role=role,
            text=text,
            session_id=session_id,
            agent_code=agent_code,
            module_id=module_id,  # 新增 module_id
            audio_filename=audio_filename,
        )
        db.add(chat_log)
        db.commit()
        db.refresh(chat_log)

        if return_obj:
            return chat_log

        return session_id
    except Exception as e:
        print(f"保存聊天訊息錯誤: {e}")
        db.rollback()
        return session_id or str(uuid.uuid4())
    finally:
        db.close()


# get_module_id_by_session 函數
def get_module_id_by_session(session_id: str) -> Optional[str]:
    """根據 session_id 獲取模組ID"""
    db = SessionLocal()
    try:
        session_map = (
            db.query(SessionUserMap)
            .filter(SessionUserMap.session_id == session_id)
            .first()
        )
        return session_map.module_id if session_map else None
    finally:
        db.close()


def save_answer_log(
    role: str, text: str, session_id: str = None, agent_code: str = "default"
):
    """保存回答記錄到資料庫"""
    db = SessionLocal()
    try:
        if session_id is None:
            session_id = str(uuid.uuid4())

        answer_log = AnswerLog(
            role=role, text=text, session_id=session_id, agent_code=agent_code
        )
        db.add(answer_log)
        db.commit()
        return session_id
    except Exception as e:
        print(f"保存回答記錄錯誤: {e}")
        db.rollback()
        return session_id or str(uuid.uuid4())
    finally:
        db.close()


def find_history(username: str):
    """
    查詢使用者的聊天歷史紀錄
    返回: 包含 sessionid、time、agent_code、gender、age 的列表
    """
    db = SessionLocal()
    try:
        # 查詢使用者的所有對話session
        user_sessions = (
            db.query(SessionUserMap)
            .filter(SessionUserMap.username == username)
            .order_by(SessionUserMap.created_at.desc())
            .all()
        )  # 按照創建時間排序

        if not user_sessions:
            return []

        history_list = []

        for session in user_sessions:
            # 獲取該session的agent設定
            agent_setting = (
                db.query(AgentSettings)
                .filter(AgentSettings.agent_code == session.agent_code)
                .first()
            )

            # 獲取該session的最新對話時間
            latest_chat = (
                db.query(ChatLog)
                .filter(ChatLog.session_id == session.session_id)
                .order_by(ChatLog.time.desc())
                .first()
            )

            # 組合歷史記錄資料
            history_item = {
                "session_id": session.session_id,
                "time": session.created_at,
                "agent_code": session.agent_code,
                "gender": agent_setting.gender if agent_setting else "未設定",
                "age": agent_setting.age if agent_setting else "未設定",
                "score": session.score,
                "is_completed": session.is_completed,
                # 新增 level 欄位，對應前端的 level
                "level": agent_setting.med_complexity if agent_setting else "未知",
            }

            history_list.append(history_item)

        # 按時間排序（最新的在前面）
        history_list.sort(key=lambda x: x["time"], reverse=True)

        return history_list

    except Exception as e:
        print(f"查詢歷史紀錄錯誤: {e}")
        return []
    finally:
        db.close()


def update_conversation_score(
    session_id: str, score: float, is_completed: bool = False
):
    """
    更新對話評分
    """
    db = SessionLocal()
    try:
        session = (
            db.query(SessionUserMap)
            .filter(SessionUserMap.session_id == session_id)
            .first()
        )

        if session:
            session.score = score
            session.is_completed = is_completed
            db.commit()
            return True
        else:
            return False

    except Exception as e:
        print(f"更新對話評分錯誤: {e}")
        db.rollback()
        return False
    finally:
        db.close()


def get_user_sessions_summary(username: str):
    """
    獲取使用者所有對話的簡要統計
    """
    db = SessionLocal()
    try:
        sessions = (
            db.query(SessionUserMap).filter(SessionUserMap.username == username).all()
        )

        total_sessions = len(sessions)
        completed_sessions = sum(1 for s in sessions if s.is_completed)
        avg_score = (
            sum(s.score for s in sessions if s.score is not None)
            / len([s for s in sessions if s.score is not None])
            if any(s.score is not None for s in sessions)
            else 0
        )

        return {
            "total_sessions": total_sessions,
            "completed_sessions": completed_sessions,
            "completion_rate": (
                completed_sessions / total_sessions * 100 if total_sessions > 0 else 0
            ),
            "average_score": round(avg_score, 2) if avg_score > 0 else 0,
        }

    except Exception as e:
        print(f"獲取使用者對話統計錯誤: {e}")
        return {
            "total_sessions": 0,
            "completed_sessions": 0,
            "completion_rate": 0,
            "average_score": 0,
        }
    finally:
        db.close()


# 密碼重設函數
def create_password_reset_token(email: str) -> str:
    """創建重設密碼token"""
    db = SessionLocal()
    try:
        # 生成安全的令牌
        token = secrets.token_urlsafe(32)

        # 設定過期時間（1小時）
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

        # 先刪除舊Token
        db.query(PasswordResetToken).filter(
            PasswordResetToken.email == email, PasswordResetToken.is_used == False
        ).delete()

        # 創建新token
        reset_token = PasswordResetToken(
            email=email, token=token, expires_at=expires_at
        )

        db.add(reset_token)
        db.commit()

        return token

    except Exception as e:
        print(f"創建密碼重設的token發生錯誤: {e}")
        db.rollback()
        return None
    finally:
        db.close()


def verify_password_reset_token(token: str) -> str:
    """驗證密碼重設Token，回傳對應的email"""
    db = SessionLocal()
    try:
        # 查找令牌
        reset_token = (
            db.query(PasswordResetToken)
            .filter(
                PasswordResetToken.token == token,
                PasswordResetToken.is_used == False,
                PasswordResetToken.expires_at > datetime.now(timezone.utc),
            )
            .first()
        )

        if reset_token:
            return reset_token.email
        else:
            return None

    except Exception as e:
        print(f"驗證密碼重設的token發生錯誤: {e}")
        return None
    finally:
        db.close()


def use_password_reset_token(token: str) -> bool:
    """標記Token為已使用過"""
    db = SessionLocal()
    try:
        reset_token = (
            db.query(PasswordResetToken)
            .filter(PasswordResetToken.token == token)
            .first()
        )

        if reset_token:
            reset_token.is_used = True
            db.commit()
            return True
        else:
            return False

    except Exception as e:
        print(f"使用密碼重設Token錯誤: {e}")
        db.rollback()
        return False
    finally:
        db.close()


def cleanup_expired_tokens():
    """清除過期Token"""
    db = SessionLocal()
    try:
        expired_tokens = (
            db.query(PasswordResetToken)
            .filter(PasswordResetToken.expires_at < datetime.now(timezone.utc))
            .delete()
        )

        db.commit()
        print(f"清理了 {expired_tokens} 過期Token")

    except Exception as e:
        print(f"清理過期Token錯誤: {e}")
        db.rollback()
    finally:
        db.close()


def save_db_conversation_summary(
    session_id: str, summary: str, module_id: str = "default_module"
):
    """將對話總結儲存到資料庫"""
    db = SessionLocal()
    try:
        summary_log = ConversationSummary(session_id=session_id, summary=summary)
        db.add(summary_log)
        db.commit()
        print(f"成功儲存 Session {session_id} (Module: {module_id}) 的總結")
    except Exception as e:
        print(f"儲存對話總結錯誤: {e}")
        db.rollback()
    finally:
        db.close()


def get_latest_conversation_summary(session_id: str):
    """從資料庫獲取最新的對話總結"""
    db = SessionLocal()
    try:
        latest_summary = (
            db.query(ConversationSummary)
            .filter(ConversationSummary.session_id == session_id)
            .order_by(ConversationSummary.created_at.desc())
            .first()
        )

        return latest_summary.summary if latest_summary else None
    except Exception as e:
        print(f"獲取最新對話總結錯誤: {e}")
        return None
    finally:
        db.close()


def get_user_message_count(session_id: str) -> int:
    """獲取指定 session 中使用者的訊息數量"""
    db = SessionLocal()
    try:
        count = (
            db.query(ChatLog)
            .filter(ChatLog.session_id == session_id, ChatLog.role == "user")
            .count()
        )
        return count
    except Exception as e:
        print(f"獲取使用者訊息數量錯誤: {e}")
        return 0
    finally:
        db.close()


def get_conversation_history_for_user(
    session_id: str = None, limit: int = 20
) -> list[str]:
    """只獲取使用者(user)的對話歷史，返回文本列表"""
    db = SessionLocal()
    try:
        query = db.query(ChatLog).filter(ChatLog.role == "user")

        if session_id:
            query = query.filter(ChatLog.session_id == session_id)

        history_rows = query.order_by(ChatLog.id.desc()).limit(limit).all()

        # 返回文本列表，最新的在前面
        return [row.text for row in history_rows]

    except Exception as e:
        print(f"獲取使用者對話歷史錯誤: {e}")
        return []
    finally:
        db.close()


# 這邊大概會加入一堆 我可能會常用的資料庫處理function
def delete_table(table_name: str):
    """
    刪除指定的資料表。
    請注意：這會永久刪除資料，請謹慎使用！
    """
    if table_name == "agentsettings":
        table_object = AgentSettings.__table__
    # 如果之後有其他需要透過指令刪除的資料表，可以在這裡擴展
    # elif table_name == 'another_table':
    #     table_object = AnotherTable.__table__
    else:
        print(f"錯誤：找不到名為 '{table_name}' 的資料表定義，或不支援透過此指令刪除。")
        return

    confirm = input(f"確認要永久刪除資料表 '{table_name}' 嗎？(輸入 'yes' 確認)：")
    if confirm.lower() == "yes":
        try:
            table_object.drop(engine)
            print(f"資料表 '{table_name}' 已成功刪除。")
        except Exception as e:
            print(f"刪除資料表 '{table_name}' 失敗: {e}")
    else:
        print(f"取消刪除資料表 '{table_name}'。")


# 3. [新增] 自動修補資料庫結構的函式 (神器)
def sync_db_schema(engine):
    """
    自動檢查資料庫，如果發現 Python Model 有新欄位但 DB 沒有，就自動加上去。
    解決手動遷移的痛苦。
    """
    inspector = inspect(engine)
    metadata = Base.metadata

    with engine.connect() as conn:
        for table_name, table_obj in metadata.tables.items():
            if not inspector.has_table(table_name):
                continue  # init_database 會負責建立新表

            # 找出 DB 裡現有的欄位
            existing_cols = [c["name"] for c in inspector.get_columns(table_name)]

            for column in table_obj.columns:
                if column.name not in existing_cols:
                    print(
                        f"🛠️ [Auto-Fix] 發現新欄位 '{table_name}.{column.name}'，正在自動新增..."
                    )
                    col_type = column.type.compile(engine.dialect)
                    try:
                        # SQLite 的 ADD COLUMN 語法
                        conn.execute(
                            text(
                                f"ALTER TABLE {table_name} ADD COLUMN {column.name} {col_type}"
                            )
                        )
                        print(f"✅ 成功新增欄位！")
                    except Exception as e:
                        print(f"⚠️ 自動新增欄位失敗 (可能需手動處理): {e}")


# 自動初始化資料庫
init_database()


# --- 新增：命令列解析與執行邏輯 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="管理資料庫表格。")
    parser.add_argument(
        "--del",
        dest="delete_table_name",
        choices=["agentsettings"],  # 這裡可以擴展其他可刪除的表格名稱
        help="刪除指定的資料表 (目前支援: agentsettings)。請謹慎使用，資料將永久刪除！",
    )

    args = parser.parse_args()

    if args.delete_table_name:
        print(f"準備刪除資料表: {args.delete_table_name}...")
        delete_table(args.delete_table_name)
    else:
        # 如果沒有指定任何命令，可以選擇做一些默認操作或者提示用戶
        print("請指定操作，例如：")
        print("  - 刪除 AgentSettings 資料表: python databases.py --del agentsettings")
        print("  - 啟動時默認調用 init_database()。")
