# modules/colonoscopy_bowklean/summary_logic.py
import logging
from typing import Dict, List
import re
from utils import generate_llm_response  # 需要引入

logger = logging.getLogger(__name__)


async def generate_summary_prompt(
    agent_settings: Dict, chat_history_dicts: List[Dict]
) -> str:
    """
    生成總結的 LLM prompt，涵蓋 7 大評分面向。
    """
    formatted_history = "\n".join(
        [
            f"{'學員' if e['role'] == 'user' else '病患'}: {e['text']}"
            for e in chat_history_dicts
        ]
    )

    # 判斷 agent_settings 是字典還是 SQLAlchemy 物件，以正確的方式讀取 med_info
    med_info = "無"
    if agent_settings:
        if isinstance(agent_settings, dict):
            # 如果是字典 (Dict)
            med_info = agent_settings.get("med_info", "無")
        else:
            # 如果是 SQLAlchemy 物件 (Object)
            med_info = getattr(agent_settings, "med_info", "無")

    summary_prompt = f"""
    你是一位專業的臨床衛教指導老師。請根據以下病患背景和完整的對話紀錄，為學員的表現提供具體評語。
    請嚴格按照下面的 JSON 格式輸出，對每一個項目提供簡潔、有建設性的評語。

    [病患背景資料]:
    {med_info}

    [對話紀錄]:
    {formatted_history}

    [輸出格式 (JSON)]:
    {{
        "total_summary": "一句話總評，點出學員表現最突出或最需改進的地方。",
        "review_med_history_summary": "針對「檢閱藥歷」的評語(是否有查閱過去經驗、院內外用藥)。",
        "medical_interview_summary": "針對「醫療面談」的評語(發問技巧、確認身分、引導對話)。",
        "counseling_edu_summary": "針對「諮商衛教」的評語(用藥指導、水分補充、飲食衛教)。",
        "humanitarian_summary": "針對「人道專業」的評語(尊重、同理心、態度)。",
        "organization_efficiency_summary": "針對「組織效率」的評語(流程順序正確性、時間控制、簡潔度)。",
        "clinical_judgment_summary": "針對「臨床判斷」的評語(判斷病人理解度、用藥合理性、禁水時間)。",
        "overall_clinical_skills_summary": "針對「整體臨床技能」的綜合評語(整合能力、整體有效性)。"
    }}
    """
    return summary_prompt


