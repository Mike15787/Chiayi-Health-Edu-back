from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timezone
import uuid

# --- 資料庫設定 ---
DATABASE_URL = 'sqlite:///chatlog.db'
engine = create_engine(DATABASE_URL, echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

# --- 資料庫模型 ---
class UserLogin(Base):
    __tablename__ = "user_login"

    id = Column(Integer, primary_key=True, autoincrement=True)  # 主鍵，自動增加
    username = Column(String(32), nullable=False, unique=True, comment="使用者名稱")
    uid = Column(String(36), unique=True, nullable=False, comment="用戶唯一ID (UUID)")
    email = Column(String(128), unique=True, nullable=False, comment="電子郵件")
    password = Column(String(128), nullable=False, comment="加密後的密碼")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, comment="建立時間")
    login_stat = Column(Boolean, default=False, nullable=False, comment="登入狀態(True=已登入)")

    def __repr__(self):
        return f"<UserLogin(username={self.username}, email={self.email}, login_stat={self.login_stat})>"

class ChatLog(Base):
    __tablename__ = 'chatlog'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=True)
    session_id = Column(String, default=lambda: str(uuid.uuid4()))
    agent_code = Column(String, nullable=False)
    role = Column(String, nullable=False)
    text = Column(Text, nullable=False)
    time = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class AnswerLog(Base):
    __tablename__ = 'answer'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, default=lambda: str(uuid.uuid4()))
    agent_code = Column(String, nullable=False)
    role = Column(String, nullable=False)
    text = Column(Text, nullable=False)
    time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
class AgentSettings(Base):
    __tablename__ = 'agent_settings'
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_code = Column(String, unique=True, nullable=False)   # 例如：A1、B1
    gender = Column(String)
    age = Column(String)
    med_info = Column(Text)     # 藥物、用法
    disease = Column(Text)
    med_complexity = Column(String)
    med_code = Column(String)
    special_status = Column(Text)
    check_day = Column(String)
    check_time = Column(String)
    check_type = Column(String)
    low_cost_med = Column(String)


#還需要再一個資料表用來存放使用者的對話有哪些
#對話紀錄就單純用chatlog 然後還需要一張資料表去導向chat
#所以這張table 要放的 sessionid user
#反正我就是要透過sessionid去找到這個人他總共進行過哪些對話所以這張資料表只需要 user跟 sessionid
#在chatlog上實現是不行的 一定要另開一張

# 修改 SessionUserMap 模型，添加評分功能
class SessionUserMap(Base):
    __tablename__ = 'sessionid_user'
    session_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(32), nullable=False, comment="使用者名稱")
    agent_code = Column(String, nullable=False, comment="案例代碼")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, comment="對話建立時間")
    score = Column(Integer, nullable=True, comment="對話評分")
    is_completed = Column(Boolean, default=False, comment="是否完成對話")

# --- 資料庫初始化函數 ---
def init_database():
    """初始化資料庫，創建所有表格"""
    Base.metadata.create_all(engine)

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
            prefix = "你：" if row.role == 'user' else "ViVi："
            history_txt += f"{prefix}{row.text}\n"
        
        return history_txt
    except Exception as e:
        print(f"獲取對話歷史錯誤: {e}")
        return ""
    finally:
        db.close()

def save_chat_message(role: str, text: str, session_id: str = None, agent_code: str = "default"):
    """保存聊天訊息到資料庫"""
    db = SessionLocal()
    try:
        if session_id is None:
            session_id = str(uuid.uuid4())
            
        chat_log = ChatLog(
            role=role, 
            text=text, 
            session_id=session_id,
            agent_code=agent_code
        )
        db.add(chat_log)
        db.commit()
        return session_id
    except Exception as e:
        print(f"保存聊天訊息錯誤: {e}")
        db.rollback()
        return session_id or str(uuid.uuid4())
    finally:
        db.close()

def save_answer_log(role: str, text: str, session_id: str = None, agent_code: str = "default"):
    """保存回答記錄到資料庫"""
    db = SessionLocal()
    try:
        if session_id is None:
            session_id = str(uuid.uuid4())
            
        answer_log = AnswerLog(
            role=role,
            text=text,
            session_id=session_id,
            agent_code=agent_code
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
        user_sessions = db.query(SessionUserMap).filter(
            SessionUserMap.username == username
        ).all()
        
        if not user_sessions:
            return []
        
        history_list = []
        
        for session in user_sessions:
            # 獲取該session的agent設定
            agent_setting = db.query(AgentSettings).filter(
                AgentSettings.agent_code == session.agent_code
            ).first()
            
            # 獲取該session的最新對話時間
            latest_chat = db.query(ChatLog).filter(
                ChatLog.session_id == session.session_id
            ).order_by(ChatLog.time.desc()).first()
            
            # 組合歷史記錄資料
            history_item = {
                "session_id": session.session_id,
                "time": latest_chat.time if latest_chat else session.created_at,
                "agent_code": session.agent_code,
                "gender": agent_setting.gender if agent_setting else "未設定",
                "age": agent_setting.age if agent_setting else "未設定",
                "score": session.score,
                "is_completed": session.is_completed
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

def update_conversation_score(session_id: str, score: float, is_completed: bool = False):
    """
    更新對話評分
    """
    db = SessionLocal()
    try:
        session = db.query(SessionUserMap).filter(
            SessionUserMap.session_id == session_id
        ).first()
        
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

def create_user_session(username: str, agent_code: str, session_id: str = None):
    """
    建立使用者對話session
    """
    db = SessionLocal()
    try:
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # 檢查是否已存在
        existing_session = db.query(SessionUserMap).filter(
            SessionUserMap.session_id == session_id
        ).first()
        
        if not existing_session:
            user_session = SessionUserMap(
                session_id=session_id,
                username=username,
                agent_code=agent_code
            )
            db.add(user_session)
            db.commit()
        
        return session_id
        
    except Exception as e:
        print(f"建立使用者session錯誤: {e}")
        db.rollback()
        return session_id or str(uuid.uuid4())
    finally:
        db.close()

def get_user_sessions_summary(username: str):
    """
    獲取使用者所有對話的簡要統計
    """
    db = SessionLocal()
    try:
        sessions = db.query(SessionUserMap).filter(
            SessionUserMap.username == username
        ).all()
        
        total_sessions = len(sessions)
        completed_sessions = sum(1 for s in sessions if s.is_completed)
        avg_score = sum(s.score for s in sessions if s.score is not None) / len([s for s in sessions if s.score is not None]) if any(s.score is not None for s in sessions) else 0
        
        return {
            "total_sessions": total_sessions,
            "completed_sessions": completed_sessions,
            "completion_rate": completed_sessions / total_sessions * 100 if total_sessions > 0 else 0,
            "average_score": round(avg_score, 2) if avg_score > 0 else 0
        }
        
    except Exception as e:
        print(f"獲取使用者對話統計錯誤: {e}")
        return {
            "total_sessions": 0,
            "completed_sessions": 0,
            "completion_rate": 0,
            "average_score": 0
        }
    finally:
        db.close()


# 自動初始化資料庫
init_database()