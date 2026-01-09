# tests/simulated_user.py
import sys
import os
from datetime import datetime, timedelta
import logging

# 讓程式能找到上一層的模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import generate_llm_response
from tests.QA_script_content import QA_SCRIPT_PROMPT

logger = logging.getLogger(__name__)

# 目前這份程式的功能寫的不太好
# 我是想要有多種版本的測試資料
# 但現在就先做個小目標
# 目標呢 就是按照範例對話 的使用者輸入 按順序一句一句改 並且生成回應
# 所以我現在 只要針對這15個agent 各自執行完一個session要有的對話
# 這樣子執行完之後 我就有15個完整的對話紀錄以及評分紀錄
# 然後


class SimulatedUserAgent:
    def __init__(self, agent_settings, module_id="colonoscopy_bowklean"):
        self.agent_settings = agent_settings
        self.module_id = module_id
        self.context_data = self._calculate_dates()

    def _calculate_dates(self):
        """根據 AgentSettings 預先計算正確答案 (與 precomputation_logic 類似)"""
        today = datetime.now()
        check_day_offset = self.agent_settings.check_day
        exam_date = today + timedelta(days=check_day_offset)

        # 簡單推算邏輯 (需與 precomputation_logic.py 保持一致以確保正確性)
        check_type = "一般"
        if (
            self.agent_settings.payment_fee
            and "4500" in self.agent_settings.payment_fee
        ):
            check_type = "無痛"
        elif "一般" in str(self.agent_settings.check_type):
            check_type = "一般"
        elif "無痛" in str(self.agent_settings.check_type):
            check_type = "無痛"

        # 禁水時間推算
        exam_time_str = self.agent_settings.check_time  # e.g. "上午08:10"
        # 這裡簡化處理，實際專案可用 regex 解析
        npo_hours = 3 if check_type == "無痛" else 2
        # 為了 Prompt 生成，這裡做一個簡單的字串替換，實際應更嚴謹
        npo_time_hint = f"檢查前 {npo_hours} 小時"

        # 藥物組合
        drug_combo = "組合一"
        extra_meds = "無"
        if self.agent_settings.drug_combination == "組合二":
            drug_combo = "組合二"
            extra_meds = "樂可舒 (Dulcolax)"

        return {
            "exam_date": f"{exam_date.month}月{exam_date.day}日",
            "exam_time": exam_time_str,
            "check_in_time": "檢查前20分鐘",
            "check_type": check_type,
            "fee": self.agent_settings.payment_fee,
            "drug_combo": drug_combo,
            "extra_meds": extra_meds,
            "npo_time": npo_time_hint,  # 讓 LLM 自己根據時間算，或這裡算好
            "dose_1_time": f"{(exam_date - timedelta(days=1)).month}月{(exam_date - timedelta(days=1)).day}日 下午5點",
            "dose_2_time": "檢查當天 凌晨/早上 (依檢查時間而定)",
            "low_residue_start": f"{(exam_date - timedelta(days=3)).month}月{(exam_date - timedelta(days=3)).day}日",
            "liquid_diet_start": f"{(exam_date - timedelta(days=1)).month}月{(exam_date - timedelta(days=1)).day}日",
        }

    async def generate_response(self, chat_history_text: str) -> str:
        """根據對話歷史生成下一句使用者(藥師)的台詞"""

        # 填入變數到 Prompt
        system_prompt = QA_SCRIPT_PROMPT.format(**self.context_data)

        full_prompt = f"""
        {system_prompt}

        以下是目前的對話紀錄：
        {chat_history_text}

        請根據 SOP 流程，生成下一句「藥師」該說的話。
        請直接輸出對話內容，不要有任何引號或動作描述。
        藥師：
        """

        # 使用專案中的 LLM 工具 (假設使用較聰明的模型來扮演藥師)
        response = await generate_llm_response(
            full_prompt, model_name="gemma3:12b"
        )  # 或 config.SCORING_MODEL_NAME
        return response.strip()
