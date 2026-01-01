# modules/colonoscopy_GI_Klean/patient_agent.py
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
        second_dose_time = precomputed_data_dict.get("second_dose_time")
        npo_start_time = precomputed_data_dict.get("npo_start_time")
        actual_check_type = precomputed_data_dict.get("actual_check_type")
        prev_1d_str = precomputed_data_dict.get("prev_1d")
        prev_2d_str = precomputed_data_dict.get("prev_2d")
        prev_3d_str = precomputed_data_dict.get("prev_3d")
    else:
        # Fallback for when precomputed data is not available (shouldn't happen in normal flow)
        logger.warning(
            "Precomputed data not available for patient agent prompt. Using dynamic calculation."
        )
        check_day_offset = agent_settings_dict.get("check_day", 0)
        today = datetime.now()
        exam_date = today + timedelta(days=check_day_offset)
        exam_day_str = f"{exam_date.month}月{exam_date.day}日"
        prev_1d_str = f"{(exam_date - timedelta(days=1)).month}月{(exam_date - timedelta(days=1)).day}日"
        prev_2d_str = f"{(exam_date - timedelta(days=2)).month}月{(exam_date - timedelta(days=2)).day}日"
        prev_3d_str = f"{(exam_date - timedelta(days=3)).month}月{(exam_date - timedelta(days=3)).day}日"

        # This fallback is incomplete, ideally precomputation always runs.
        # For simplicity, other precomputed values will just be '未提供' if not in dict.
        second_dose_time = "未提供"
        npo_start_time = "未提供"
        actual_check_type = agent_settings_dict.get("check_type", "未提供")

    check_day_offset = agent_settings_dict.get("check_day", 0)
    check_time_str = agent_settings_dict.get("check_time", "未提供")
    today_str = f"{datetime.now().month}月{datetime.now().day}日"

    # --- 2. 構建基礎角色和特殊指令 ---
    instructions = [
        f"你是一個病患，預計在 {check_day_offset} 天後（也就是 {exam_day_str}）要做大腸鏡檢查。今天（{today_str}）你剛拿到醫生開的清腸藥（保可淨），正要向護理師（使用者）請教如何使用及相關衛教事宜。",
        "你需要仔細聆聽護理師的衛教內容，並給予簡短、自然的回應，讓他知道你有在聽。",
        "你的回答應盡量簡潔，通常在10個字以內。",
    ]

    # 處理特殊狀況
    special_status = agent_settings_dict.get("special_status", "無")
    if "不知道檢查型態" in special_status:
        instructions.append(
            "【特殊任務】由於你不確定自己的檢查類型，你必須在對話中主動詢問護理師：『我的檢查是一般檢查還是無痛的？需要麻醉嗎？』"
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
- 檢查前三天 ({prev_3d_str}) 飲食：低渣
- 檢查前兩天 ({prev_2d_str}) 飲食：低渣
- 檢查前一天 ({prev_1d_str}) 飲食：無渣流質
- 檢查當天第二包藥服用時間：{second_dose_time}
- 檢查禁水開始時間：{npo_start_time}
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
