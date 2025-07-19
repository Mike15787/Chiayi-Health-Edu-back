from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import Optional
import logging
import uuid
from databases import find_history

# 配置日誌
logger = logging.getLogger(__name__)

# 創建路由器
history_router = APIRouter(prefix="/history", tags=["History"])

# 添加的 Pydantic 模型
class HistoryResponse(BaseModel):
    session_id: str
    time: str
    agent_code: str
    gender: str
    age: str
    score: Optional[float] = None
    is_completed: bool = False

class ConversationDetailResponse(BaseModel):
    session_info: dict
    agent_setting: dict
    chat_logs: list
    answer_logs: list



# API 端點
@history_router.get("/{username}", response_model=list[HistoryResponse])
async def get_user_history(username: str):
    """
    獲取使用者的聊天歷史紀錄 (用於 AppSidebar)
    """
    try:
        history_data = find_history(username)
        
        formatted_history = []
        for item in history_data:
            formatted_history.append(HistoryResponse(
                session_id=item["session_id"],
                time=item["time"].isoformat() if isinstance(item["time"], datetime) else item["time"],
                agent_code=item["agent_code"],
                gender=item["gender"],
                age=item["age"],
                score=item["score"],
                is_completed=item["is_completed"]
            ))
        
        return formatted_history
        
    except Exception as e:
        logger.error(f"獲取使用者歷史紀錄錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"獲取歷史紀錄失敗: {str(e)}")

