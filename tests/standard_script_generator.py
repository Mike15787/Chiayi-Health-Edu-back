# tests/golden_script_generator.py
import re
from typing import List, Dict, Optional
from databases import AgentSettings, PrecomputedSessionAnswer
from scenarios.colonoscopy_bowklean.config import MED_INSTRUCTIONS


class GoldenScriptGenerator:
    """
    根據 Agent 設定與預計算數據，生成符合滿分標準的「藥師(使用者)」對話劇本。
    """

    def __init__(
        self, agent_settings: AgentSettings, precomputed: PrecomputedSessionAnswer
    ):
        self.agent = agent_settings
        self.pre = precomputed
        self.script = []

    def generate(self) -> List[str]:
        """產生完整的對話列表"""
        self.script = []

        # 1. 開場與確認身分
        self._add_greeting()

        # 2. 詢問經驗與情緒安撫 (A2特殊處理)
        self._add_history_check()

        # 3. 確認檢查資訊 (時間、型態、費用)
        self._add_exam_info_check()

        # 4. 確認目前用藥
        self._add_current_meds_check()

        # 5. 藥物組合與開立目的說明
        self._add_drug_intro()

        # 6. 飲食衛教
        self._add_diet_instruction()

        # 7. 服藥方法與時間 (第一包、第二包、樂可舒)
        self._add_dosing_instructions()

        # 8. 水分補充與理想狀態
        self._add_hydration_and_result()

        # 9. 禁水說明
        self._add_npo_instruction()

        # 10. 特殊用藥衛教
        self._add_special_meds_instruction()

        # 11. 結尾
        self._add_closing()

        return self.script

    def _add_greeting(self):
        self.script.append("先生/小姐您好，請坐。")  # 問好 + 請坐
        self.script.append("請問今天是您本人要使用保可淨清腸藥嗎？")  # 確認本人

    def _add_history_check(self):
        self.script.append("請問您之前有使用過清腸劑或保可淨的經驗嗎？")  # 詢問經驗

        # 針對 Agent A2 (焦慮/第一次)，根據 PDF SOP 必須給予安撫
        if self.agent.agent_code == "A2":
            self.script.append(
                "知道您之前沒用過清腸藥，會擔心不知道如何正確使用，等一下會跟您說明清腸藥的使用方式，不用太擔心。"
            )  # 情緒回應

    def _add_exam_info_check(self):
        # 確認日期時間
        self.script.append(
            "跟您確認一下，醫師幫我們安排哪一天做大腸鏡檢查，日期和時間是？"
        )

        # 確認檢查型態
        if "不知道" in str(self.agent.special_status) or "不知道" in str(
            self.agent.check_type
        ):
            # 如果不知道，必須引導
            self.script.append(
                "請問您這次是做一般的還是無痛的檢查？如果不確定的話，請問剛剛繳費大約是多少錢？是800元還是4500元？"
            )
        else:
            self.script.append("確認一下，您這次是做一般的還是無痛的大腸鏡檢查？")

    def _add_current_meds_check(self):
        self.script.append(
            "請問目前還有沒有服用其他藥物？例如抗凝血劑、降血糖藥或是便秘藥？"
        )

    def _add_drug_intro(self):
        # 判斷藥物組合
        has_dulcolax = "組合二" in str(self.agent.drug_combination)

        if has_dulcolax:
            self.script.append(
                "好的，目前醫師開立的清腸藥是「保可淨」共一盒兩包，另外還有口服瀉藥錠劑「樂可舒」。目的是要把腸道清乾淨，讓醫師檢查得更清楚。"
            )
        else:
            self.script.append(
                "好的，目前醫師開立的清腸藥是「保可淨」共一盒兩包。目的是要把腸道清乾淨，讓醫師檢查得更清楚。"
            )

        self.script.append("幫您把服藥日期及時間註記在您的衛教單張上。")  # 註記時間

    def _add_diet_instruction(self):
        # 使用 Precomputed 的日期確保完全匹配
        self.script.append(
            f"除了吃藥，飲食控制很重要。檢查前三天({self.pre.prev_3d})和前兩天({self.pre.prev_2d})請吃低渣飲食，"
            f"少吃纖維多的蔬菜水果奶類。檢查前一天({self.pre.prev_1d})請改吃無渣流質飲食，例如無料的湯或運動飲料。"
        )

    def _add_dosing_instructions(self):
        has_dulcolax = "組合二" in str(self.agent.drug_combination)

        # 1. 樂可舒 (若有) - 通常在檢查前一天中午
        if has_dulcolax:
            self.script.append(
                f"首先，請在檢查前一天({self.pre.prev_1d})的中午12點，服用口服瀉藥錠劑樂可舒。"
            )

        # 2. 第一包保可淨 - 檢查前一天下午5點 (SOP標準時間)
        self.script.append(
            f"第一包保可淨請在檢查前一天({self.pre.prev_1d})的下午5點服用。"
            "泡法是：將一包藥粉倒入150c.c.的常溫水中，攪拌至少5分鐘至溶解，然後立即喝完。"
        )

        # 3. 第二包保可淨 - 依據檢查時間計算 (使用 Precomputed 的時間)
        self.script.append(
            f"第二包保可淨請在檢查當天({self.pre.exam_day})的{self.pre.second_dose_time}服用。"
            "泡法一樣是加入150c.c.水中攪拌溶解後喝下。"
        )

    def _add_hydration_and_result(self):
        has_dulcolax = "組合二" in str(self.agent.drug_combination)

        # 依據組合決定水量 (PDF 規則: 組合一約2000cc / 組合二約1000cc)
        # 注意：这里的逻辑需对应 scoring_criteria.json 的 hydration_and_goal.s1 (2000cc) 或 s2 (1000cc)
        if has_dulcolax:
            # 組合二 (搭配樂可舒) -> 水分較少
            self.script.append(
                "服藥後要多喝水。喝完藥後間隔半小時，請準備1000c.c.的水分，在1小時內分次慢慢喝完。"
                "可以喝白開水、運動飲料或清湯。"
            )
        else:
            # 組合一 (單純保可淨) -> 水分較多
            self.script.append(
                "服藥後要多喝水。喝完藥後間隔一小時，請準備2000c.c.的水分，在2-3小時內分次慢慢喝完。"
                "可以喝白開水、運動飲料或清湯，不要一次喝太快。"
            )

        # 理想狀態與作用時間
        self.script.append(
            "通常服藥2-3小時後會開始腹瀉。理想的清腸狀態是排出的糞水呈現「淡黃色清澈液體」，像尿液一樣沒有渣。"
        )

    def _add_npo_instruction(self):
        # 禁水時間
        self.script.append(
            f"最後要注意禁水時間。檢查當天{self.pre.npo_start_time}之後，就完全不能再喝水或進食了。"
        )

    def _add_special_meds_instruction(self):
        if not self.agent.med_code or self.agent.med_code == "無":
            # 若無特殊藥物，也需要給予一般指示 (N)
            instruction = MED_INSTRUCTIONS.get("N", "")
            self.script.append(f"關於您的其他常規藥物：{instruction}")
            return

        # 處理多重藥物代碼 (例如 "抗血小板藥物 S2; 高血壓藥物 X1")
        med_codes = self.agent.med_code.split(";")

        for code_str in med_codes:
            # 提取代碼 (S1, X1, etc.)
            match = re.search(r"(S\d|X\d|N)", code_str)
            if match:
                code_key = match.group(1)
                instruction = MED_INSTRUCTIONS.get(code_key)
                if instruction:
                    # 為了讓 AI 判斷這是在講哪個藥，最好加上藥物名稱
                    # 但這裡簡化處理，直接講指令，通常 scoring 只要抓到關鍵字(停藥/不停藥)就會給分
                    self.script.append(f"針對您的{code_str}：{instruction}")

    def _add_closing(self):
        self.script.append("請問對於剛剛的說明，還有哪裡不清楚或想問的嗎？")
        self.script.append("不客氣，有問題可以再詢問我們，謝謝。")
