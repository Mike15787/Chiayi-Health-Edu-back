# modules/colonoscopy_GI_Klean/summary_logic.py
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


async def calculate_organization_efficiency_score_llm(full_history: str) -> float:
    """
    呼叫 gemma3:12b (或 gemma3:4b) 來對對話流程進行 0-6 分的評分。
    """
    prompt = f"""
    你是一位資深的臨床衛教評分老師。你的任務是根據以下的完整對話紀錄，評估學員(醫生)的衛教流程是否清晰、有條理且遵循了標準順序。

    [正確的衛教順序標準]:
    1.  **醫療面談**: 開場問候、確認病人身份、確認檢查時間與類型。
    2.  **飲食管理**: 說明檢查前三天的低渣飲食原則。
    3.  **檢查前一天指導**: 說明無渣流質飲食、第一包清腸劑的服用時間與方法、水分補充。
    4.  **檢查當天指導**: 說明第二包清腸劑的服用時間與方法、禁水時間。
    5.  **額外補充**: 處理特殊用藥(如抗凝血劑)、說明排便理想狀態等。

    [評分指南]:
    - 6分: 完美遵循順序，衛教流暢，所有要點均在正確的階段提出。
    - 4-5分: 大致遵循順序，可能有少數項目順序顛倒，但不影響整體理解。
    - 2-3分: 順序較為混亂，重要衛教項目散落在對話各處，結構性不佳。
    - 0-1分: 完全沒有順序，對話混亂，讓病人難以理解準備流程。

    [對話紀錄]:
    {full_history}

    [你的評分]:
    請根據上述標準，給出一個 0 到 6 之間的整數分數。**請只輸出一個數字，不要有任何其他文字、符號或解釋。**
    """

    logger.info(f"正在請求 gemma3:12b (或 gemma3:4b) 進行組織效率評分 (0-6)...")
    # 這裡可以根據 config.py 中的 PATIENT_AGENT_MODEL_NAME 或專用模型來決定
    response_text = await generate_llm_response(
        prompt, "gemma3:12b"
    )  # 暫時使用 gemma3:12b

    try:
        match = re.search(r"\d+", response_text)
        if match:
            score = int(match.group(0))
            clamped_score = max(0, min(6, score))
            logger.info(f"LLM 原始分數: {score}，校正後分數: {clamped_score}")
            return float(clamped_score)
        else:
            logger.warning(
                f"無法從 LLM 回應中解析出組織效率分數: '{response_text}'。將預設為 0 分。"
            )
            return 0.0
    except Exception as e:
        logger.error(
            f"解析組織效率分數時發生錯誤: {e}。回應: '{response_text}'。將預設為 0 分。"
        )
        return 0.0
