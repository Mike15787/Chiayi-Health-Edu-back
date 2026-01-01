# modules/colonoscopy_GI_Klean/precomputation_logic.py
import logging
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from databases import AgentSettings, PrecomputedSessionAnswer
from scenarios.colonoscopy_GI_Klean.config import MODULE_ID  # 引入模組ID

logger = logging.getLogger(__name__)


async def perform_precomputation(session_id: str, agent_code: str, db: Session):
    """
    為 colonoscopy_GI_Klean 模組預先計算並儲存答案。
    """
    agent_settings = (
        db.query(AgentSettings).filter(AgentSettings.agent_code == agent_code).first()
    )
    if not agent_settings:
        logger.error(
            f"[{session_id}] Cannot precompute answers, agent_settings not found for {agent_code}"
        )
        return

    # 1. 日期計算 (現有邏輯)
    today = datetime.now()
    check_day_offset = agent_settings.check_day
    exam_date = today + timedelta(days=check_day_offset)

    def format_date(d):
        return f"{d.month}月{d.day}日"

    exam_day_str = format_date(exam_date)
    prev_1d_str = format_date(exam_date - timedelta(days=1))
    prev_2d_str = format_date(exam_date - timedelta(days=2))
    prev_3d_str = format_date(exam_date - timedelta(days=3))

    # 2. 判斷真實檢查類型 (現有邏輯)
    actual_check_type = "一般"
    if agent_settings.payment_fee and "4500" in agent_settings.payment_fee:
        actual_check_type = "無痛"

    # 3. 計算第二包藥劑服用時間 (現有邏輯)
    check_time_str = agent_settings.check_time
    # 確保 check_time_str 包含 '上午' 或 '下午' 以正確解析
    hour_str = check_time_str.split(":")[0]  # e.g., "上午08"
    if "上午" in hour_str:
        hour = int(hour_str.replace("上午", ""))
    elif "下午" in hour_str:
        hour = int(hour_str.replace("下午", "")) + 12  # 轉換為24小時制
    else:
        hour = int(hour_str)  # 如果沒有上午/下午，直接解析

    second_dose_time = "凌晨7點"  # 預設下午
    if "上午" in check_time_str:
        if hour < 10:
            second_dose_time = "凌晨3點"
        elif 10 <= hour < 12:
            second_dose_time = "凌晨4點"

    # 4. 計算禁水時間 (現有邏輯)
    npo_hours_before = 3 if actual_check_type == "無痛" else 2
    # 將 "上午11:20" 轉換為 datetime 物件
    # 確保時間字符串與格式匹配
    time_part = check_time_str.replace("上午", "").replace("下午", "")
    check_time_obj = datetime.strptime(
        f"{exam_date.strftime('%Y-%m-%d')} {time_part}", "%Y-%m-%d %H:%M"
    )
    # 如果原始時間是下午且小時不是12，需要手動加12小時
    if "下午" in check_time_str and int(time_part.split(":")[0]) != 12:
        check_time_obj += timedelta(hours=12)

    npo_start_obj = check_time_obj - timedelta(hours=npo_hours_before)
    # 格式化為 "上午8:20"，並移除前導零
    npo_start_time = npo_start_obj.strftime("%H:%M")
    if npo_start_obj.hour < 12:
        npo_start_time = f"上午{npo_start_obj.hour}:{npo_start_obj.minute:02d}"
    else:
        npo_start_time = f"下午{npo_start_obj.hour-12 if npo_start_obj.hour > 12 else 12}:{npo_start_obj.minute:02d}"

    # 5. 存入資料庫
    new_precomputed = PrecomputedSessionAnswer(
        session_id=session_id,
        module_id=MODULE_ID,  # 使用模組 ID
        exam_day=exam_day_str,
        prev_1d=prev_1d_str,
        prev_2d=prev_2d_str,
        prev_3d=prev_3d_str,
        second_dose_time=second_dose_time,
        npo_start_time=npo_start_time,
        actual_check_type=actual_check_type,
    )
    db.merge(new_precomputed)
    db.commit()
    logger.info(
        f"[{session_id}] Precomputed answers saved successfully for module {MODULE_ID}."
    )
