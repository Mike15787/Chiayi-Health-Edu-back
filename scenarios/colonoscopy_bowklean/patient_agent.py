# scenarios/colonoscopy_bowklean/patient_agent.py
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def build_patient_prompt(
    user_text: str,
    history: str,
    agent_settings_dict: Dict,
    precomputed_data_dict: Optional[Dict] = None,
) -> str:
    """
    構建優化的 prompt，包含 agent 設定。
    precomputed_data_dict 現在是一個字典，包含預先計算的值。
    """

    # --- 1. 計算實際檢查日期 ---
    if precomputed_data_dict:
        exam_day_str = precomputed_data_dict.get("exam_day")
        actual_check_type = precomputed_data_dict.get("actual_check_type")
    else:
        # Fallback for when precomputed data is not available (shouldn't happen in normal flow)
        logger.warning(
            "Precomputed data not available for patient agent prompt. Using dynamic calculation."
        )
        check_day_offset = agent_settings_dict.get("check_day", 0)
        today = datetime.now()
        exam_date = today + timedelta(days=check_day_offset)
        exam_day_str = f"{exam_date.month}月{exam_date.day}日"

        actual_check_type = agent_settings_dict.get("check_type", "未提供")

    check_day_offset = agent_settings_dict.get("check_day", 0)
    check_time_str = agent_settings_dict.get("check_time", "未提供")
    today_str = f"{datetime.now().month}月{datetime.now().day}日"

    # --- 2. 構建基礎角色和特殊指令 ---
    instructions = [
        f"你是一個病患，預計在 {check_day_offset} 天後（也就是 {exam_day_str}）要做大腸鏡檢查。今天（{today_str}）你剛拿到醫生開的清腸藥（保可淨），正要向護理師（使用者）請教如何使用及相關衛教事宜。",
        "你需要仔細聆聽護理師的衛教內容，並給予簡短、自然的回應",
        "在15個字以內簡潔回答",
    ]

    # 處理特殊狀況
    special_status = agent_settings_dict.get("special_status", "無")
    if "不知道檢查型態" in special_status:
        instructions.append(
            "【特殊任務】由於你不確定自己的檢查類型，如果對方詢問是做一般檢查還是無痛檢查，統一回答「我不知道我做哪種檢查」"
        )

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
- 預計檢查時間：{check_time_str}
- 檢查類型：{agent_settings_dict.get('check_type', '未提供')} (實際類型: {actual_check_type})
- 繳交費用：{agent_settings_dict.get('payment_fee', '未提供')}
- 過去使用瀉藥經驗：{agent_settings_dict.get('laxative_experience', '未提供')}
- 是否需要代購低渣飲食：{agent_settings_dict.get('low_cost_med', '未提供')}
---
"""
    system_prompt = base_role_prompt + personal_info

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
            f"{system_prompt}\n\n" "現在，對話開始。\n" f"護理師：{user_text}\n" "你："
        )

    return prompt
