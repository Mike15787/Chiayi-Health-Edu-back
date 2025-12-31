# scenarios/colonoscopy_bowklean/scoring_logic.py
import json
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from datetime import datetime
from typing import List, Dict, Optional, Any
import re
from utils import generate_llm_response
from databases import (
    AnswerLog,
    PrecomputedSessionAnswer,
    ScoringPromptLog,
    SessionUserMap,
    AgentSettings,
    ChatLog,
    Scores,
    SessionInteractionLog,
    ScoringAttributionLog,
)
from scenarios.colonoscopy_bowklean.config import (
    MODULE_ID,
    SCORING_CRITERIA_FILE,
    SCORING_MODEL_NAME,
    STRONGER_SCORING_MODEL_NAME,
    MED_INSTRUCTIONS,
    CATEGORY_TO_FIELD_MAP,
    COMPOSITE_SUB_ITEM_IDS,
)
from scenarios.colonoscopy_bowklean.summary_logic import (
    calculate_organization_efficiency_score_llm,
)

logger = logging.getLogger(__name__)


class ColonoscopyBowkleanScoringLogic:
    def __init__(self):
        logger.info(f"Initializing Scoring Logic for module: {MODULE_ID}...")
        self.criteria = self._load_criteria(SCORING_CRITERIA_FILE)
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.index, self.criteria_id_list = self._build_vector_index()
        logger.info(f"Scoring Logic for module: {MODULE_ID} initialized successfully.")

    def _load_criteria(self, file_path):
        # file_path is relative to the project root, so adjust if needed
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_vector_index(self):
        documents_to_encode = []
        criteria_id_list = []

        logger.info(f"Building vector index for module {MODULE_ID} from criteria...")
        for criterion in self.criteria:
            item_description = criterion.get("item", "")
            examples = " ".join(criterion.get("example_answer", []))
            combined_text = f"{item_description} {examples}"

            documents_to_encode.append(combined_text)
            criteria_id_list.append(criterion["id"])

        logger.info(
            f"Encoding {len(documents_to_encode)} combined criteria documents for module {MODULE_ID}..."
        )
        embeddings = self.model.encode(documents_to_encode, convert_to_tensor=False)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        logger.info(f"Vector index for module {MODULE_ID} built successfully.")
        return index, criteria_id_list

    def find_relevant_criteria_ids(self, user_input: str, top_k: int = 5):
        """根據使用者輸入，找到最相關的評分項目ID"""
        input_embedding = self.model.encode([user_input])
        distances, I = self.index.search(input_embedding, top_k)

        relevant_ids = set()
        for idx in I[0]:
            if idx != -1:
                relevant_ids.add(self.criteria_id_list[idx])

        logger.info(
            f"Found relevant criteria IDs for module {MODULE_ID}: {list(relevant_ids)}"
        )
        return list(relevant_ids)

    async def _call_llm_and_log(
        self, session_id: str, item_id: str, prompt: str, model_name: str, db: Session
    ) -> int:
        """呼叫 LLM 進行評分，並將 prompt 和 response 記錄到資料庫。"""
        try:
            response_text = await generate_llm_response(prompt, model_name)
            raw_response = response_text.strip()
            score = 1 if "1" in raw_response else 0

            log_entry = ScoringPromptLog(
                session_id=session_id,
                module_id=MODULE_ID,  # 使用模組專屬 ID
                scoring_item_id=item_id,
                prompt_text=prompt,
                llm_response=raw_response,
                final_score=score,
            )
            db.add(log_entry)

            logger.info(
                f"[{session_id}] Scored item '{item_id}' for module {MODULE_ID} with score {score}. Raw response: '{raw_response[:50]}...'"
            )
            return score
        except Exception as e:
            logger.error(
                f"[{session_id}] Error during LLM call or logging for item '{item_id}' (module {MODULE_ID}): {e}"
            )
            return 0

    async def process_user_inputs_for_scoring(
        self, session_id: str, chat_snippet: List[Dict], db: Session
    ) -> List[str]:
        """處理多個使用者輸入並進行評分"""
        if not chat_snippet:
            return []

        def format_snippet(snippet: List[Dict]) -> str:
            formatted_lines = []
            for item in snippet:
                role = "學員" if item["role"] == "user" else "病患"
                formatted_lines.append(f"{role}: {item['message']}")
            return "\n".join(formatted_lines)

        formatted_conversation = format_snippet(chat_snippet)

        logger.info(
            f"[{session_id}] Scoring conversation snippet for module {MODULE_ID}: {formatted_conversation[:150]}..."
        )

        relevant_ids = self.find_relevant_criteria_ids(formatted_conversation, top_k=5)
        if not relevant_ids:
            logger.info(
                f"[{session_id}] No relevant scoring criteria found for module {MODULE_ID}."
            )
            return []

        already_passed_stmt = db.query(AnswerLog.scoring_item_id).filter(
            AnswerLog.session_id == session_id, AnswerLog.score == 1
        )
        passed_ids = {row.scoring_item_id for row in already_passed_stmt.all()}

        precomputed_data = (
            db.query(PrecomputedSessionAnswer)
            .filter(PrecomputedSessionAnswer.session_id == session_id)
            .first()
        )

        newly_passed_ids = []

        for item_id in relevant_ids:
            if item_id in passed_ids:
                logger.info(
                    f"[{session_id}] Skipping already passed item: {item_id} for module {MODULE_ID}."
                )
                continue

            criterion = next(
                (item for item in self.criteria if item["id"] == item_id), None
            )
            if criterion:
                score = await self.score_item(
                    session_id, formatted_conversation, criterion, db, precomputed_data
                )

                if score == 1:
                    newly_passed_ids.append(item_id)

        return newly_passed_ids

    async def score_item(
        self,
        session_id: str,
        conversation_context: str,
        criterion: dict,
        db: Session,
        precomputed_data: Optional[PrecomputedSessionAnswer],
    ):
        """對單個評分項目進行評分並儲存結果"""
        item_id = criterion["id"]
        item_type = criterion.get("type", "RAG")

        score = 0
        try:
            if item_type == "RAG":
                score = await self._check_rag_basic(
                    session_id, conversation_context, criterion, db
                )
            elif item_type == "RAG_4b":
                score = await self._check_rag_strong(
                    session_id, conversation_context, criterion, db
                )
            elif item_type == "RAG_confirm_med_time":
                score = await self._check_confirm_med_time(
                    session_id, conversation_context, criterion, precomputed_data, db
                )
            elif item_type == "RAG_confirm_npo":
                score = await self._check_npo_time(
                    session_id, conversation_context, precomputed_data, db
                )
            elif item_type == "RAG_diet_logic":
                score = await self._check_diet_logic(
                    session_id, conversation_context, precomputed_data, db
                )
            elif item_type == "RAG_med_usage_method":
                score = await self._check_med_usage_method(
                    session_id, conversation_context, criterion, db
                )
            elif item_type == "RAG_special_meds":
                score = await self._check_special_meds(
                    session_id, conversation_context, db
                )
            elif item_type == "non_term":
                score = await self._check_non_term(
                    session_id, conversation_context, criterion, db
                )
            elif item_type == "RAG_emo_response":
                score = await self._check_emo_response(
                    session_id, conversation_context, criterion, db
                )
            elif item_type == "RAG_review_med_history":
                score = await self._check_review_med_history(
                    session_id, conversation_context, criterion, db
                )
            elif item_type == "first_med_mix_method":
                # 對應: 說明保可淨使用方式 (泡製)
                score = await self._check_med_mix_method(
                    session_id, conversation_context, criterion, db
                )
            elif item_type == "first_med_mix_time":
                # 對應: 說明樂可舒使用方式 (錠劑時間)
                score = await self._check_first_med_mix_time(
                    session_id, conversation_context, criterion, db
                )
            elif item_type == "hydration_method":
                # [修正] 區分 s1 (2000cc) 與 s2 (1000cc)
                score = await self._check_hydration_method(
                    session_id, conversation_context, criterion, db
                )
            elif item_type == "ideal_intestinal":
                score = await self._check_ideal_intestinal(
                    session_id, conversation_context, criterion, db
                )
            elif item_type == "RAG_explain_npo":
                # 這是單純檢查「有無提到禁水」，不同於 RAG_confirm_npo (檢查時間點準確性)
                score = await self._check_explain_npo_mention(
                    session_id, conversation_context, criterion, db
                )
            else:  # 對於未定義的類型，使用基礎 RAG 邏輯
                score = await self._check_rag_basic(
                    session_id, conversation_context, criterion, db
                )

            stmt = sqlite_insert(AnswerLog).values(
                session_id=session_id,
                module_id=MODULE_ID,  # 新增 module_id
                scoring_item_id=item_id,
                score=score,
                created_at=datetime.now(),
            )
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=["session_id", "scoring_item_id"], set_=dict(score=score)
            )
            db.execute(on_conflict_stmt)
            db.commit()

            return score

        except Exception as e:
            logger.error(
                f"[{session_id}] Error scoring item '{item_id}' for module {MODULE_ID}: {e}"
            )
            db.rollback()
            return 0

    async def _check_rag_basic(
        self, session_id: str, conversation_context: str, criterion: dict, db: Session
    ) -> int:
        """使用優化後的 LLM 進行基礎 RAG 評分"""
        prompt = f"""
        你是一個嚴謹的對話評分員。你的任務是判斷一段對話中，學員是否完成了特定的溝通任務。請嚴格按照指令操作。

        [溝通任務]: {criterion['任務說明']}
        [任務說明/參考範例]: {" / ".join(criterion['example_answer'])}

        [學員與病人的對話紀錄]:
        ---
        {conversation_context}
        ---

        [你的判斷任務]:
        請仔細閱讀上面的[學員與病人的對話紀錄]，判斷其中是否**包含**了符合[溝通任務]的內容。你不需要判斷整段對話是否完美，只需要確認這個特定的任務有沒有被完成即可。

        - 如果對話中**有**任何一句話或一個片段的語意符合[溝通任務]，請只輸出 "1"。
        - 如果遍覽整段對話後，**都找不到**符合[溝通任務]的內容，請只輸出 "0"。

        [你的判斷 (只輸出 1 或 0)]:
        """
        return await self._call_llm_and_log(
            session_id, criterion["id"], prompt, SCORING_MODEL_NAME, db
        )

    async def _check_rag_strong(
        self, session_id: str, conversation_context: str, criterion: dict, db: Session
    ) -> int:
        """使用更強的 gemma:4b 模型進行評分"""
        prompt = f"""
        你是一個資深護理師。根據學員的回答，以及參考答案，判斷是否達成了評分項目。
        判斷的標準為語意表達正確、所講的內容正確。
        如果達成，只輸出 "1"。如果未達成，只輸出 "0"。不要有任何其他文字。

        [評分項目]: {criterion['item']}
        [參考答案]: {" / ".join(criterion['example_answer'])}
        [學員回答]:
        {conversation_context}

        [你的判斷 (1 或 0)]:
        """
        return await self._call_llm_and_log(
            session_id, criterion["id"], prompt, STRONGER_SCORING_MODEL_NAME, db
        )

    # 12/04檢查 OK
    async def _check_confirm_med_time(
        self,
        session_id: str,
        conversation_context: str,
        criterion: dict,
        precomputed_data: PrecomputedSessionAnswer,
        db: Session,
    ) -> int:
        """檢查學員是否正確說明了第一包和第二包清腸劑的服用時間。"""
        if not precomputed_data:
            logger.warning(
                f"[{session_id}] Precomputed data is not available for _check_confirm_med_time in module {MODULE_ID}."
            )
            return 0

        prompt = ""
        if criterion["id"] == "clinical_med_timing_1":
            prompt = f"""
            你是一個資深護理師，請判斷學員的衛教內容是否正確。
            [情境] 病人應該在檢查前一天 ({precomputed_data.prev_1d}) 的下午5點服用第一包藥。
            [學員回答]: {conversation_context}
            
            學員是否有清楚且正確地告知病人第一包藥的服用日期和時間？如果正確，只輸出 "1"。如果不正確，只輸出 "0"。
            """
        elif criterion["id"] == "clinical_med_timing_2":
            prompt = f"""
            你是一個資深護理師，請判斷學員的衛教內容是否正確。
            [情境] 病人應該在檢查當天 ({precomputed_data.exam_day}) 的 {precomputed_data.second_dose_time} 服用第二包藥。
            [學員回答]: {conversation_context}
            
            學員是否有清楚且正確地告知病人第二包藥的服用日期和時間？如果正確，只輸出 "1"。如果不正確，只輸出 "0"。
            """
        else:
            return 0

        return await self._call_llm_and_log(
            session_id, criterion["id"], prompt, SCORING_MODEL_NAME, db
        )

    async def _check_npo_time(
        self,
        session_id: str,
        conversation_context: str,
        precomputed_data: PrecomputedSessionAnswer,
        db: Session,
    ) -> int:
        """檢查學員是否清楚說明禁水時間點，並連帶給予 npo_mention 分數。"""
        if not precomputed_data:
            logger.warning(
                f"[{session_id}] Precomputed data is not available for _check_npo_time in module {MODULE_ID}."
            )
            return 0

        # 根據實際檢查類型判斷禁水前應停止多久
        npo_hours_before = 3 if precomputed_data.actual_check_type == "無痛" else 2

        prompt = f"""
        你是一個資深護理師，請判斷學員的衛教內容是否正確。
        [情境] 病人正確的禁水開始時間是 {precomputed_data.npo_start_time} (在 {precomputed_data.actual_check_type} 檢查前 {npo_hours_before} 小時)。
        [學員回答]: {conversation_context}
        
        學員是否有清楚告知病人，從「{precomputed_data.npo_start_time}」這個時間點開始就完全不能再喝任何東西（包含水）？
        如果學員清楚說明了這個具體的時間點和禁水要求，只輸出 "1"。否則輸出 "0"。
        """
        score = await self._call_llm_and_log(
            session_id, "clinical_npo_timing", prompt, SCORING_MODEL_NAME, db
        )

        if score == 1:
            logger.info(
                f"[{session_id}] _check_npo_time passed, automatically scoring npo_mention for module {MODULE_ID}."
            )
            try:
                stmt = sqlite_insert(AnswerLog).values(
                    session_id=session_id,
                    module_id=MODULE_ID,  # 新增 module_id
                    scoring_item_id="npo_mention",
                    score=1,
                    created_at=datetime.now(),
                )
                on_conflict_stmt = stmt.on_conflict_do_update(
                    index_elements=["session_id", "scoring_item_id"], set_=dict(score=1)
                )
                db.execute(on_conflict_stmt)
            except Exception as e:
                logger.error(
                    f"[{session_id}] Failed to auto-score 'npo_mention' for module {MODULE_ID}: {e}"
                )

        return score

    async def _check_med_usage_method(
        self, session_id: str, conversation_context: str, criterion: dict, db: Session
    ) -> int:
        """檢查學員是否正確說明清腸劑的泡製方法。"""
        prompt = f"""
        你是一個資深護理師，請判斷學員是否正確說明了清腸劑的泡製與服用方式。
        
        [正確方法參考]: 將一包藥粉倒入150c.c.的常溫水中，攪拌至完全溶解後立即喝完。
        
        [學員的對話]:
        {conversation_context}
        
        學員是否有提到類似上述的泡製和服用方法（水量、攪拌、喝完）？只要有清楚說明過一次即可，無論是針對第一包還是第二包。
        如果說明正確，只輸出 "1"。如果說明不完整或不正確，只輸出 "0"。
        """
        return await self._call_llm_and_log(
            session_id, criterion["id"], prompt, SCORING_MODEL_NAME, db
        )

    async def _check_special_meds(
        self, session_id: str, conversation_context: str, db: Session
    ) -> int:
        """根據病人的特殊用藥，判斷學員的衛教是否正確。"""
        session_map = (
            db.query(SessionUserMap)
            .filter(SessionUserMap.session_id == session_id)
            .first()
        )
        if not session_map:
            logger.warning(
                f"[{session_id}] Cannot find session map for _check_special_meds in module {MODULE_ID}."
            )
            return 0

        agent_settings = (
            db.query(AgentSettings)
            .filter(AgentSettings.agent_code == session_map.agent_code)
            .first()
        )
        if not agent_settings or not agent_settings.med_code:
            logger.info(
                f"[{session_id}] Agent has no special meds to check for module {MODULE_ID}. Skipping."
            )
            return 0

        med_codes = agent_settings.med_code.strip().split(
            ";"
        )  # 假設 med_code 可能是多個，用分號分隔

        # 遍歷所有可能的 med_code 類型，檢查是否有匹配
        for med_code_part in med_codes:
            # 提取代碼部分，例如從 "抗凝血劑 S1" 提取 "S1"
            code_match = re.search(r"(S\d|X\d|N)", med_code_part)
            if code_match:
                med_key = code_match.group(0)
                if med_key in MED_INSTRUCTIONS:
                    correct_instruction = MED_INSTRUCTIONS.get(med_key)

                    prompt = f"""
                    你是一位資深的臨床藥師，請根據病人的用藥情況和學員的衛教內容進行評分。
                    
                    [病人用藥情境]:
                    病人正在服用特殊藥物：{med_code_part}。
                    
                    [此藥物正確的衛教指令]:
                    "{correct_instruction}"
                    
                    [學員與病人的對話紀錄]:
                    {conversation_context}
                    
                    [評分任務]:
                    請判斷學員是否向病人提供了與上述「正確的衛教指令」語意相符的說明？
                    如果學員的說明核心內容正確（例如，指出了藥物 {med_code_part} 需要停藥或繼續服用），請只輸出 "1"。
                    如果學員的說明錯誤、不完整，或完全沒有提到如何處理藥物 {med_code_part}，請輸出 "0"。
                    """
                    # 只要有一個特殊藥物衛教正確，就給予 1 分
                    score = await self._call_llm_and_log(
                        session_id,
                        "specify_special_meds",
                        prompt,
                        SCORING_MODEL_NAME,
                        db,
                    )
                    if score == 1:
                        return 1  # 只要一個通過就返回1

        logger.warning(
            f"[{session_id}] No matching special med instruction found or all checks failed for module {MODULE_ID}."
        )
        return 0  # 如果沒有匹配的指令或所有檢查都失敗，則返回 0

    async def _check_diet_logic(
        self,
        session_id: str,
        conversation_context: str,
        precomputed_data: PrecomputedSessionAnswer,
        db: Session,
    ) -> int:
        """檢查學員是否正確說明了飲食衛教。"""
        if not precomputed_data:
            logger.warning(
                f"[{session_id}] Precomputed data is not available for _check_diet_logic in module {MODULE_ID}."
            )
            return 0
        prompt = f"""
        你是一個資深護理師，請判斷學員的衛教內容是否正確。
        [情境] 檢查日期是 {precomputed_data.exam_day}。
        - 檢查前三天 ({precomputed_data.prev_3d}) 應該要低渣飲食。
        - 檢查前兩天 ({precomputed_data.prev_2d}) 應該要低渣飲食。
        - 檢查前一天 ({precomputed_data.prev_1d}) 應該要無渣流質飲食。
        [學員回答]: {conversation_context}
        
        學員是否給出了正確的飲食衛教？如果正確，只輸出 "1"。如果不正確，只輸出 "0"。
        """
        return await self._call_llm_and_log(
            session_id, "diet_basic", prompt, SCORING_MODEL_NAME, db
        )

    async def _check_med_mix_method(
        self, session_id: str, conversation_context: str, criterion: dict, db: Session
    ) -> int:
        """檢查學員是否正確說明了特定一包清腸劑的泡製方法。"""

        # 從 criterion ID 判斷是第一包還是第二包
        dose_number_text = "第一包" if "first" in criterion["id"] else "第二包"

        prompt = f"""
        你是一位嚴謹的臨床藥師，請判斷學員是否正確地說明了「{dose_number_text}」清腸劑的泡製與服用方式。

        [正確的泡製方法參考]:
        將一包「保可淨」倒入裝有150c.c.常溫水的杯中，攪拌至完全溶解後立即喝完。

        [學員與病人的對話紀錄]:
        ---
        {conversation_context}
        ---

        [你的判斷任務]:
        請判斷學員在對話中，是否有針對「{dose_number_text}」藥劑，清楚說明了類似上述的泡製方法？
        你需要檢查的關鍵點包含：「150c.c.的水量」、「攪拌/溶解」、「立即喝完」。
        只要語意正確即可，不需逐字比對。

        - 如果學員對「{dose_number_text}」的說明包含上述所有關鍵點，請只輸出 "1"。
        - 如果說明不完整、不正確或完全沒有提到，請只輸出 "0"。

        [你的判斷 (只輸出 1 或 0)]:
        """
        return await self._call_llm_and_log(
            session_id, criterion["id"], prompt, SCORING_MODEL_NAME, db
        )

    async def _check_first_med_mix_time(
        self, session_id: str, conversation_context: str, criterion: dict, db: Session
    ) -> int:
        prompt = f"""
        你是一位藥師。請檢查學員是否有針對「口服瀉藥錠劑」(例如：樂可舒/Dulcolax/粉紅色藥丸) 進行衛教。
        [對話紀錄]: {conversation_context}
        學員是否有提到除了喝的保可淨之外，還有「藥丸/錠劑」要吃？
        並且是否有說明大概什麼時候吃 (例如：檢查前一天、中午、或搭配第一包)？
        有提到錠劑及其服用時機輸出 "1"，否則 "0"。
        """
        return await self._call_llm_and_log(
            session_id, criterion["id"], prompt, SCORING_MODEL_NAME, db
        )

    # 12/3 從non_term開始修正 好的 從non_term跳過XD 我這邊沒有那個專業用語集
    async def _check_non_term(
        self, session_id: str, conversation_context: str, criterion: dict, db: Session
    ) -> int:
        """
        檢查學員是否避免使用專業術語 (non_term)。
        邏輯：如果學員使用了 exclude_term 中的詞彙，則為 0 分；若使用通俗語言解釋，則為 1 分。
        """
        exclude_terms = criterion.get("exclude_term", [])
        terms_str = "、".join(exclude_terms)

        prompt = f"""
        你是一個嚴格的醫學溝通評分員。你的任務是檢查學員在對話中是否使用了病人聽不懂的「專業術語」。
        
        [禁止使用的專業術語列表]: {terms_str}
        
        [學員與病人的對話紀錄]:
        ---
        {conversation_context}
        ---
        
        [評分標準]:
        1. 如果學員在對話中**直接使用**了上述禁止列表中的任何一個詞彙（例如直接說「你有高血壓嗎」而沒有用通俗說法），請輸出 "0"。
        2. 如果學員完全沒有使用上述術語，或者使用了通俗的說法（例如將「高血壓」說成「血壓比較高」、「糖尿病」說成「血糖的問題」），請輸出 "1"。
        
        [你的判斷 (只輸出 1 或 0)]:
        """
        return await self._call_llm_and_log(
            session_id, criterion["id"], prompt, SCORING_MODEL_NAME, db
        )

    # 1207 修正
    async def _check_emo_response(
        self, session_id: str, conversation_context: str, criterion: dict, db: Session
    ) -> int:
        """檢查學員是否對病人的情緒有適當回應 (同理心)。"""
        prompt = f"""
        你是一個資深護理導師。請觀察學員是否展現出「同理心」或對病人的情緒/焦慮做出適當回應。
        
        [任務說明]: {criterion['任務說明']}
        [參考範例]: {" / ".join(criterion['example_answer'])}
        
        [對話紀錄]:
        {conversation_context}
        
        [評分重點]:
        如果病人表達了焦慮、擔心、不懂或是沉默，學員是否有進行安撫、鼓勵，或表示願意協助？
        - 有展現關懷或安撫：輸出 "1"
        - 忽視病人情緒，只顧講自己的衛教內容：輸出 "0"
        """
        return await self._call_llm_and_log(
            session_id, criterion["id"], prompt, SCORING_MODEL_NAME, db
        )

    # 1207 修正
    async def _check_review_med_history(
        self, session_id: str, conversation_context: str, criterion: dict, db: Session
    ) -> int:
        """檢查學員是否詢問過往用藥經驗。"""
        prompt = f"""
        請判斷學員是否詢問了病人關於「過去使用清腸藥物」的經驗。
        
        [對話紀錄]:
        {conversation_context}
        
        [評分標準]:
        學員是否有問類似以下的問題：
        - "您以前有做過大腸鏡嗎？"
        - "您之前有喝過保可淨/清腸藥嗎？"
        - "您知道這個藥怎麼喝嗎？" (隱含詢問經驗)
        
        如果有詢問經驗，請輸出 "1"，否則輸出 "0"。
        """
        return await self._call_llm_and_log(
            session_id, criterion["id"], prompt, SCORING_MODEL_NAME, db
        )

    # 1207 修正中
    async def _check_hydration_method(
        self, session_id: str, conversation_context: str, criterion: dict, db: Session
    ) -> int:
        item_id = criterion["id"]

        if item_id == "hydration_and_goal.s1":
            correct_volume = "2000c.c.以上 (或 8 杯)"
            correct_timing_concept = "2-3小時內陸續分次喝完"
        elif item_id == "hydration_and_goal.s2":
            correct_volume = "1000c.c. (或 4 杯)"
            correct_timing_concept = "1小時內陸續分次喝完"
        else:
            logger.error(f"Unknown hydration criteria ID: {item_id}")
            return 0

        prompt = f"""
        你是一位藥師，請檢查學員對於「水分補充」的衛教是否正確且完整。
        [正確標準]: 
        1. 總水量約 {correct_volume}。
        2. 需要「分次」喝完 (例如：{correct_timing_concept})，不可牛飲。
        [對話紀錄]: {conversation_context}
        學員是否提到了「{correct_volume}」以及「分次喝/慢慢喝」這兩個關鍵概念？
        符合輸出 "1"，不符合或未提及輸出 "0"。
        """
        return await self._call_llm_and_log(
            session_id, item_id, prompt, SCORING_MODEL_NAME, db
        )

    async def _check_ideal_intestinal(
        self, session_id: str, conversation_context: str, criterion: dict, db: Session
    ) -> int:
        """檢查是否說明清腸理想狀態 (淡黃清澈)。"""
        prompt = f"""
        請檢查學員是否描述了清腸成功的「理想糞便狀態」。
        
        [正確描述]: 糞便應該要是「淡黃色」、「清澈」、「像尿液一樣」的液體，沒有渣。
        
        [對話紀錄]:
        {conversation_context}
        
        [判斷]:
        學員是否有提到上述任何關於糞便顏色或性狀的正確描述？
        有提到 (例如：拉到水水的像尿一樣) -> 輸出 "1"
        沒提到 -> 輸出 "0"
        """
        return await self._call_llm_and_log(
            session_id, criterion["id"], prompt, SCORING_MODEL_NAME, db
        )

    async def _check_explain_npo_mention(
        self, session_id: str, conversation_context: str, criterion: dict, db: Session
    ) -> int:
        """
        單純檢查學員是否有提到「禁水」這個動作 (RAG_explain_npo)。
        註：不同於 RAG_confirm_npo (那是檢查時間點計算是否正確)，這裡只檢查有無衛教病人「要禁水」。
        """
        prompt = f"""
        請檢查學員是否有告知病人檢查前需要「禁水」(不能喝水)。
        
        [對話紀錄]:
        {conversation_context}
        
        [判斷]:
        學員是否有提到「檢查前幾小時不能喝水」、「要禁食禁水」或類似的指令？
        這裡不需要判斷時間點是否計算精確，只需要確認學員有**傳達禁水這個指令**即可。
        
        有提到禁水 -> 輸出 "1"
        完全沒提到 -> 輸出 "0"
        """
        return await self._call_llm_and_log(
            session_id, criterion["id"], prompt, SCORING_MODEL_NAME, db
        )

    def _check_organization_sequence_by_time(
        self, session_id: str, has_dulcolax: bool, has_special_meds: bool, db: Session
    ) -> float:
        """
        [組織效率] 透過資料庫紀錄的時間戳記，檢查衛教順序。

        標準順序 (類別代號)：
        1. 飲食衛教 (diet_basic)
        2. 口服瀉藥錠劑 (med_mix_method_and_time.s2) -> 僅在 has_dulcolax=True 時檢查
        3. 清腸粉劑 (med_mix_method_and_time.s1)
        4. 禁水時間 (npo_mention)
        5. 其他用藥 (specify_special_meds) -> 僅在 has_special_meds=True 時檢查

        評分邏輯：
        - 3.0 分：所有「應說明項目」皆存在，且時間順序符合 1 -> 2 -> 3 -> 4 -> 5
        - 1.0 分：有缺漏項目，或順序顛倒
        """

        # 1. 定義要檢查的項目 ID
        target_items = ["diet_basic", "med_mix_method_and_time.s1", "npo_mention"]

        if has_dulcolax:
            target_items.append("med_mix_method_and_time.s2")

        if has_special_meds:
            target_items.append("specify_special_meds")

        # 2. 從 AnswerLog 撈取這些項目「第一次」拿到 1 分的時間
        # 注意：我們只關心 score=1 的紀錄
        logs = (
            db.query(AnswerLog.scoring_item_id, AnswerLog.created_at)
            .filter(
                AnswerLog.session_id == session_id,
                AnswerLog.score == 1,
                AnswerLog.scoring_item_id.in_(target_items),
            )
            .all()
        )

        # 將結果轉為 Dict: {item_id: timestamp}
        # 如果同一個項目有多筆 (雖然不應發生)，取最早的一筆
        passed_times = {}
        for item_id, created_at in logs:
            if item_id not in passed_times or created_at < passed_times[item_id]:
                passed_times[item_id] = created_at

        # 3. 檢查是否有缺漏 (Missing Items)
        # 只要有一個「應說明項目」不在 passed_times 裡，就視為缺漏 -> 1分
        for item in target_items:
            if item not in passed_times:
                logger.info(f"[{session_id}] 組織效率扣分: 缺少必要項目 {item}")
                return 1.0

        # 4. 檢查時間順序 (Chronological Order)
        # 順序：Diet -> [Tablet] -> Powder -> NPO -> [Special]

        t_diet = passed_times["diet_basic"]
        t_powder = passed_times["med_mix_method_and_time.s1"]
        t_npo = passed_times["npo_mention"]

        # 基礎檢查: 飲食 -> 粉劑 -> 禁水
        if not (t_diet < t_powder < t_npo):
            logger.info(
                f"[{session_id}] 組織效率扣分: 基礎順序錯誤 (Diet->Powder->NPO)"
            )
            return 1.0

        # 若有錠劑 (Dulcolax)，順序應為: 飲食 -> 錠劑 -> 粉劑
        # (通常錠劑是中午吃，粉劑是下午/晚上喝，故衛教順序通常先講錠劑)
        if has_dulcolax:
            t_tablet = passed_times["med_mix_method_and_time.s2"]
            if not (t_diet < t_tablet < t_powder):
                logger.info(f"[{session_id}] 組織效率扣分: 錠劑順序錯誤")
                return 1.0

        # 若有特殊用藥，順序應為: 禁水 -> 特殊用藥
        # (通常講完禁水限制後，會補充說明還有哪些藥要停、哪些藥可以喝水配服)
        if has_special_meds:
            t_special = passed_times["specify_special_meds"]
            if not (t_npo < t_special):
                logger.info(
                    f"[{session_id}] 組織效率扣分: 特殊用藥順序錯誤 (應在禁水後)"
                )
                return 1.0

        logger.info(f"[{session_id}] 組織效率滿分: 順序正確且無缺漏")
        return 3.0

    # --- 新增：最終分數計算邏輯 ---
    async def calculate_final_scores(
        self, session_id: str, db: Session
    ) -> Dict[str, str]:
        """
        計算該 Session 的最終分數。
        包含標準項目的累加以及特殊複合規則（如 S1-S4 藥物衛教）的計算。
        """
        logger.info(
            f"[{session_id}] Calculating final scores using module logic: {MODULE_ID}"
        )

        # 1. 取得基本資料
        session_map = (
            db.query(SessionUserMap)
            .filter(SessionUserMap.session_id == session_id)
            .first()
        )
        agent_code = session_map.agent_code if session_map else "unknown"
        agent_settings = (
            db.query(AgentSettings)
            .filter(AgentSettings.agent_code == agent_code)
            .first()
        )

        # 判斷是否為特殊教案 A2 (保可淨)
        is_case_A2 = agent_code == "A2"

        # 判斷藥物組合 (E: 保可淨 Only, F: 保可淨 + 樂可舒)
        # 根據 PDF 藥物組合 E/F 的說明：如果有 Dulcolax 則是 F，否則 E
        # 這裡簡易判斷：若 med_code 或 med_info 包含 Dulcolax/樂可舒 則為 F
        has_dulcolax = False
        if agent_settings:
            # 優先使用明確的 drug_combination 欄位判斷
            if agent_settings.drug_combination == "組合二":
                has_dulcolax = True

        # [新增] 判斷是否有特殊用藥 (has_special_meds)
        # 用來決定組織效率評分時，是否必須檢查 "specify_special_meds" 這一項
        has_special_meds = False
        if agent_settings and agent_settings.med_code:
            # 如果 med_code 欄位有值且不是 "無"，則視為有特殊用藥
            if (
                agent_settings.med_code.strip()
                and agent_settings.med_code.strip() != "無"
            ):
                has_special_meds = True

        # 1. 獲取所有已得分項目 ID
        passed_items_query = (
            db.query(AnswerLog.scoring_item_id)
            .filter(AnswerLog.session_id == session_id, AnswerLog.score == 1)
            .all()
        )
        passed_item_ids = {item.scoring_item_id for item in passed_items_query}

        # 2. 獲取 UI 互動產生的檢閱藥歷分數 (從 Scores 表讀取暫存值)
        #    注意：main.py 在 end_chat 時會先存一次 UI 產生的 review_med_history_score
        # ui_score_record = db.query(Scores).filter(Scores.session_id == session_id).first()
        # ui_review_score = float(ui_score_record.review_med_history_score) if ui_score_record else 0.0

        # 3. 獲取對話紀錄 (用於計算時間和順序)
        chat_logs = (
            db.query(ChatLog)
            .filter(ChatLog.session_id == session_id)
            .order_by(ChatLog.time.asc())
            .all()
        )

        # --- 初始化各類別分數 ---
        scores = {key: 0.0 for key in CATEGORY_TO_FIELD_MAP.values()}

        # 輔助函式：檢查項目是否通過
        def is_passed(item_id):
            return 1 if item_id in passed_item_ids else 0

        # 現在確實感覺這樣子 把每個類別的項目都抓出來處理比較好 這樣子雖然程式比較長 但是好處是好除錯 改項目也方便

        # ==========================================
        # 1. 檢閱藥歷 (Review Med History) - 總分 9
        # ==========================================

        # --- (修改處開始) 計算 UI 互動產生的分數 ---
        # 讀取 SessionInteractionLog 表
        interaction_log = (
            db.query(SessionInteractionLog)
            .filter(SessionInteractionLog.session_id == session_id)
            .first()
        )

        ui_review_score = 0.0
        if interaction_log:
            # 依據 PDF 權重計算
            if interaction_log.viewed_alltimes_ci:
                ui_review_score += 2.0  # 歷次清腸
            if interaction_log.viewed_chiachi_med:
                ui_review_score += 3.0  # 本院用藥
            if interaction_log.viewed_med_allergy:
                ui_review_score += 1.0  # 過敏史
            if interaction_log.viewed_disease_diag:
                ui_review_score += 1.0  # 疾病診斷
            if interaction_log.viewed_cloud_med:
                ui_review_score += 2.0  # 雲端藥歷

        logger.info(
            f"[{session_id}] UI Interaction Score Calculated: {ui_review_score}"
        )

        # 檢閱藥歷 = UI分數
        scores["review_med_history_score"] = ui_review_score

        # ==========================================
        # 2. 醫療面談 (Medical Interview) - 總分 9
        # ==========================================

        # 2-1. 問好 (+1, A2為0.5)
        score_hello = 0.5 if is_case_A2 else 1.0
        val_hello = is_passed("greeting_hello") * score_hello

        # 2-2. 請坐 (+1, A2為0.5)
        score_sit = 0.5 if is_case_A2 else 1.0
        val_sit = is_passed("invite_to_sit") * score_sit

        # 2-3. 適切發問 (+4)
        # 規則: S2 和 S3 是 OR 閘 (共1分)，S1, S4, S5 各 1 分
        pg_s1 = is_passed("proper_guidance_s1")
        pg_s2 = is_passed("proper_guidance_s2")
        pg_s3 = is_passed("proper_guidance_s3")
        pg_s4 = is_passed("proper_guidance_s4")
        pg_s5 = is_passed("proper_guidance_s5")

        pg_s2_s3_score = 1 if (pg_s2 + pg_s3) > 0 else 0
        proper_guidance_total = pg_s1 + pg_s4 + pg_s5 + pg_s2_s3_score  # Max 4

        # 2-4. 確認本人 (+1)
        val_confirm_self = is_passed("confirm_self_use") * 1.0

        # 5. 詢問經驗 (+1) -> Note: 這裡 PDF 醫療面談裡也有一項 "詢問是否有使用經驗" (+1)
        # 注意：這與檢閱藥歷的 +2 是分開算的。scoring_criteria 裡我們只有一個 item ID。
        # 如果 review_med_history_1 通過，這裡也算通過。
        val_ask_exp = is_passed("review_med_history_1") * 1.0

        # 7. 無專業術語 (+1)
        val_no_term = is_passed("no_use_term") * 1.0

        # 8. 情緒回應 (+0, A2為1)
        val_emo = (is_passed("emo_response") * 1.0) if is_case_A2 else 0.0

        scores["medical_interview_score"] = (
            val_hello
            + val_sit
            + proper_guidance_total
            + val_confirm_self
            + val_ask_exp
            + val_no_term
            + val_emo
        )

        # ==========================================
        # 3. 諮商衛教 (Counseling) - 總分 9
        # ==========================================

        # 3-1. 開立目的 (+0.5)
        # 這裡有 s1 (保可淨) 和 s2 (保+樂)，看 passed 哪個
        val_purpose = (
            is_passed("explain_med_purpose.s1") or is_passed("explain_med_purpose.s2")
        ) * 0.5

        # 3-2. 註記時間 (+0.5)
        val_note_time = is_passed("note_have_med_time") * 0.5

        # 3-3. 藥物使用時機及方式 (+2) -> 組合 E vs F
        # 組合 E (Only Bowklean): med_mix_method_and_time.s1
        # 組合 F (Bowklean + Dulcolax): med_mix_method_and_time.s1 + s2
        val_med_method = 0.0
        if has_dulcolax:  # 組合 F
            # PDF: "說明藥物使用時機及方式(+2)"。
            # 這裡簡單處理：各佔 1 分
            val_med_method = (
                is_passed("med_mix_method_and_time.s1") * 1.0
                + is_passed("med_mix_method_and_time.s2") * 1.0
            )
        else:  # 組合 E
            # 只有保可淨，佔 2 分
            val_med_method = is_passed("med_mix_method_and_time.s1") * 2.0

        # 3-4. 水分補充 (+1)
        # s1 (2000cc) or s2 (1000cc). 兩者 ID 不同，視為同一分
        val_hydro = (
            is_passed("hydration_and_goal.s1") or is_passed("hydration_and_goal.s2")
        ) * 1.0

        # 3-5. 清腸理想狀態 (+1)
        val_ideal = is_passed("ideal_intestinal") * 1.0

        # 3-6. 作用時間 (+0.5)
        val_onset = is_passed("med_onset_duration") * 0.5

        # 3-7. 無痛確認 (+1)
        val_pain_check = pg_s2_s3_score  # 重複使用 因為這不需要額外再判斷一次

        # 3-8. 禁水時間 (+1)
        val_npo_explain = is_passed("npo_mention") * 1.0

        # 3-9. 特殊用藥 (+1)
        val_special_med = is_passed("specify_special_meds") * 1.0

        # 3-10. 簡易飲食 (+0.5)
        val_diet = is_passed("diet_basic") * 0.5

        scores["counseling_edu_score"] = (
            val_purpose
            + val_note_time
            + val_med_method
            + val_hydro
            + val_ideal
            + val_onset
            + val_pain_check
            + val_npo_explain
            + val_special_med
            + val_diet
        )

        # ==========================================
        # 4. 人道專業 (Humanitarian) - 總分 9
        # ==========================================

        # 4-1. 表現尊重 (+2 or +3)
        # 公式: 若 (Hello + Sit + Guidance A) 都得分 ->
        # A2: +2分. 非A2: +3分.
        has_respect_basis = val_hello > 0 and val_sit > 0 and pg_s1 > 0
        score_respect = 0.0
        if has_respect_basis:
            score_respect = 2.0 if is_case_A2 else 3.0

        # 4-2. 需求滿足 (+4 or +3)
        # 來自 satisfy_patient_infomation (JSON weight 3).
        # PDF: A2 (+4), others (+3).
        # 我們用 is_passed * PDF分數
        score_satisfy_weight = 4.0 if is_case_A2 else 3.0
        val_satisfy = is_passed("satisfy_patient_infomation") * score_satisfy_weight

        # 4-3. 同理心 (+1 or +2)
        # 公式: (No Term + Ask Exp) ->
        # A2: +1. Others: +2.
        has_empathy_basis = val_no_term > 0 and val_ask_exp > 0
        score_empathy = 0.0
        if has_empathy_basis:
            score_empathy = 1.0 if is_case_A2 else 2.0

        # 4-4. 信賴感 (+1) 標準回答 "不客氣，有問題可以再詢問我們。"
        val_trust = is_passed("great_relationship_trust") * 1.0

        # 4-5. 舒適守密 (+, A2為0)
        # emo_response
        val_comfort = (is_passed("emo_response") * 1.0) if is_case_A2 else 0.0
        # PDF 寫: "對病人情緒...回應 (+1)" 在醫療面談。
        # 人道裡有 "注意舒適 (+1)". 暫時略過或設為 0

        scores["humanitarian_score"] = (
            score_respect + val_satisfy + score_empathy + val_trust + val_comfort
        )

        # ==========================================
        # 5. 組織效率 (Organization Efficiency) - 總分 9
        # ==========================================

        # 5-1. 優先順序 (+3 or +1)
        # 呼叫 LLM 判斷 (Diet -> Oral -> Powder -> NPO -> Others)
        val_sequence = self._check_organization_sequence_by_time(
            session_id, has_dulcolax, has_special_meds, db
        )

        # 5-2. 及時且適時 (+1.5 or +0.5) 這部分之後改成用使用者輸入音檔長度+tts生成音檔長度的時間加總
        # 時間控制: 5-9 分鐘 -> 1.5, 否則 0.5
        val_time = 0.5
        if chat_logs:
            start_time = chat_logs[0].time
            end_time = chat_logs[-1].time
            duration_minutes = (end_time - start_time).total_seconds() / 60.0
            if 5 <= duration_minutes <= 9:
                val_time = 1.5
            logger.info(f"衛教時間: {duration_minutes:.2f} 分鐘, 得分: {val_time}")

        # 5-3. 歷練而簡潔 (+3.5) - 複雜公式
        # 公式: (諮商衛教總分/6 + 醫療面談3項/5 + 檢閱藥歷3項/7) / 3 * 3.5
        # 醫療面談3項: (Confirm Self + Guidance A + Guidance B) = 1+1+3 = 5
        mi_3_sub = val_confirm_self + proper_guidance_total

        # 檢閱藥歷3項: (Exp + UI_In + UI_Out) -> 假設 Exp(+2) + 5 (UI Max?) = 7
        rmh_3_sub = 0.0
        if interaction_log:
            # 依據 PDF 權重計算
            if interaction_log.viewed_alltimes_ci:
                rmh_3_sub += 2.0  # 歷次清腸
            if interaction_log.viewed_chiachi_med:
                rmh_3_sub += 3.0  # 本院用藥
            if interaction_log.viewed_cloud_med:
                rmh_3_sub += 2.0  # 雲端藥歷

        concise_part = (
            (
                (scores["counseling_edu_score"] / 6.0)
                + (mi_3_sub / 5.0)
                + (rmh_3_sub / 7.0)
            )
            / 3.0
            * 3.5
        )

        scores["organization_efficiency_score"] = val_sequence + val_time + concise_part

        # ==========================================
        # 6. 臨床判斷 (Clinical Judgment) - 總分 9
        # ==========================================
        # 6-1. 特殊藥物有無 (+2) -> identify_special_meds
        val_has_special = is_passed("identify_special_meds") * 2.0

        # 6-2. 判斷服藥時間 (+1) -> clinical_med_timing_1 (第一包)
        val_judge_time1 = is_passed("clinical_med_timing_1") * 1.0

        # 6-3. 判斷早上服藥點 (+1) -> clinical_med_timing_2 (第二包)
        val_judge_time2 = is_passed("clinical_med_timing_2") * 1.0

        # 6-4. 判斷禁水時間 (+1) -> clinical_npo_timing
        val_judge_npo = is_passed("clinical_npo_timing") * 1.0

        # 6-5. 判斷特殊藥停用 (+2) -> specify_special_meds
        val_judge_stop = is_passed("specify_special_meds") * 2.0

        # 6-6. 判斷理解程度 (+1) -> satisfy_patient_infomation / 3 or 4 * 1 ?
        # PDF 公式: 人道專業中 需求滿足得分 / 3 or 4
        if score_satisfy_weight > 0:
            val_judge_understand = val_satisfy / score_satisfy_weight * 1.0
        else:
            val_judge_understand = 0.0

        # 6-7. 判斷開立合理 (+1) -> Ask Exp / 2
        # PDF 公式: 檢閱藥歷中 詢問經驗得分 / 2
        val_judge_reasonable = (val_ask_exp / 2.0) * 1.0  # 2/2 = 1

        scores["clinical_judgment_score"] = (
            val_has_special
            + val_judge_time1
            + val_judge_time2
            + val_judge_npo
            + val_judge_stop
            + val_judge_understand
            + val_judge_reasonable
        )

        # ==========================================
        # 7. 整體臨床技能 (Overall) - 總分 9
        # ==========================================
        # 1. 態度 (愛心同理) (+3)
        # 公式: (醫療面談/6 + 人道專業/6) / 2 * 3
        val_attitude = (
            (
                (scores["medical_interview_score"] / 6.0)
                + (scores["humanitarian_score"] / 6.0)
            )
            / 2.0
            * 3.0
        )

        # 2. 整合能力 (+4)
        # 公式: (檢閱藥歷/6 + 醫療面談4項/6 + 臨床判斷/6) / 3 * 4
        # 醫療面談3項: (Confirm + proper_guidance_total + NoTerm) = 1+1+3+1 = 6
        mi_4_sub = val_confirm_self + proper_guidance_total + val_no_term
        val_integration = (
            (
                (scores["review_med_history_score"] / 6.0)
                + (mi_4_sub / 6.0)
                + (scores["clinical_judgment_score"] / 6.0)
            )
            / 3.0
            * 4.0
        )

        # 3. 整體有效性 (+2)
        # 公式: [6大類得分 / 9 的總和] / 3 ? (PDF 公式較模糊，依照之前的推導)
        # (Sum(Score/9) for all 6 cats) / 3 * 2 ?
        # 假設: ((Sum of all 6 scores) / 54) * ? -> PDF: divide by 3
        # 讓我們用平均達成率的概念:
        sum_ratios = (
            scores["review_med_history_score"] / 9.0
            + scores["medical_interview_score"] / 9.0
            + scores["counseling_edu_score"] / 9.0
            + scores["humanitarian_score"] / 9.0
            + scores["organization_efficiency_score"] / 9.0
            + scores["clinical_judgment_score"] / 9.0
        )
        val_effectiveness = sum_ratios / 3.0  # max = 6/3 = 2.

        scores["overall_clinical_skills_score"] = (
            val_attitude + val_integration + val_effectiveness
        )

        # --- 10. 計算總分並格式化 ---
        # 總分為 7 項類別的加總 (滿分 63)
        real_total = (
            scores["review_med_history_score"]
            + scores["medical_interview_score"]
            + scores["counseling_edu_score"]
            + scores["humanitarian_score"]
            + scores["organization_efficiency_score"]
            + scores["clinical_judgment_score"]
            + scores["overall_clinical_skills_score"]
        )

        # 將所有數值轉為字串格式 (保留兩位小數)
        result = {key: str(round(value, 2)) for key, value in scores.items()}
        result["total_score"] = str(round(real_total, 2))

        logger.info(f"[{session_id}] Final Scores Calculated: {result}")
        return result

    async def get_detailed_scores(self, session_id: str, db: Session) -> Dict[str, Any]:
        """
        生成此模組 (保可淨衛教) 的詳細評分清單。
        包含: Standard LLM items, UI items, Logic items.
        回傳格式需符合 Scoring.py 定義的 CategoryDetail 結構。
        """
        logger.info(f"[{session_id}] Generating detailed scores for module {MODULE_ID}")

        # 1. 準備基礎資料
        session_map = (
            db.query(SessionUserMap)
            .filter(SessionUserMap.session_id == session_id)
            .first()
        )
        agent_code = session_map.agent_code
        agent_settings = (
            db.query(AgentSettings)
            .filter(AgentSettings.agent_code == agent_code)
            .first()
        )

        # 判斷特殊條件
        is_case_A2 = agent_code == "A2"
        has_dulcolax = agent_settings and agent_settings.drug_combination == "組合二"
        has_special_meds = False
        if (
            agent_settings
            and agent_settings.med_code
            and agent_settings.med_code.strip() != "無"
        ):
            has_special_meds = True

        # 取得得分紀錄
        answer_logs = (
            db.query(AnswerLog).filter(AnswerLog.session_id == session_id).all()
        )
        user_scores = {log.scoring_item_id: log.score for log in answer_logs}

        # 取得 UI 互動紀錄
        interaction_log = (
            db.query(SessionInteractionLog)
            .filter(SessionInteractionLog.session_id == session_id)
            .first()
        )

        # 取得對話紀錄 (用於歸因顯示)
        attributions = (
            db.query(ScoringAttributionLog, ChatLog.text)
            .join(ChatLog, ScoringAttributionLog.chat_log_id == ChatLog.id)
            .filter(ScoringAttributionLog.session_id == session_id)
            .all()
        )

        dialogue_map = {}
        for attr, text in attributions:
            if attr.scoring_item_id not in dialogue_map:
                dialogue_map[attr.scoring_item_id] = []
            dialogue_map[attr.scoring_item_id].append(text)

        # 輔助函式
        def is_passed(item_id):
            return 1 if user_scores.get(item_id, 0) == 1 else 0

        # 初始化分類
        categories = [
            "檢閱藥歷",
            "醫療面談",
            "諮商衛教",
            "人道專業",
            "組織效率",
            "臨床判斷",
            "整體臨床技能",
        ]
        # 這裡我們手動構建字典結構，對應 Scoring.py 的 Pydantic Model
        grouped_details = {
            cat: {"category_name": cat, "items": []} for cat in categories
        }

        # ====================================================
        # A. 處理 JSON 定義的標準項目 (Standard Items)
        # ====================================================
        # 定義要合併顯示的適切發問子項目 ID
        proper_guidance_ids = [
            "proper_guidance_s1",
            "proper_guidance_s2",
            "proper_guidance_s3",
            "proper_guidance_s4",
            "proper_guidance_s5",
        ]

        for criterion in self.criteria:
            item_id = criterion["id"]
            category = criterion.get("category", "其他")

            # --- [修改重點 1] 過濾掉要合併的「適切發問」子項目 ---
            if item_id in proper_guidance_ids:
                continue

            # --- [修改重點 1] 過濾「說明藥物開立目的」 ---
            # 邏輯：組合一 (has_dulcolax=False) -> 顯示 s1
            #       組合二 (has_dulcolax=True)  -> 顯示 s2
            if item_id == "explain_med_purpose.s1" and has_dulcolax:
                continue  # 如果是組合二，跳過 s1
            if item_id == "explain_med_purpose.s2" and not has_dulcolax:
                continue  # 如果是組合一，跳過 s2

            # --- [修改重點 2] 過濾「說明樂可舒使用方式」 ---
            # 邏輯：組合一沒有樂可舒，直接隱藏 s2
            if item_id == "med_mix_method_and_time.s2" and not has_dulcolax:
                continue

            # --- [修改重點 3] 過濾「水分補充」 (如果 s1/s2 也是對應不同組合的話) ---
            # 假設 s1 對應組合一，s2 對應組合二 (根據您的 JSON 判斷)
            if item_id == "hydration_and_goal.s1" and has_dulcolax:
                continue
            if item_id == "hydration_and_goal.s2" and not has_dulcolax:
                continue

            if category not in grouped_details:
                grouped_details[category] = {"category_name": category, "items": []}

            score_val = user_scores.get(item_id, 0)
            base_weight = criterion.get("weight", 1.0)

            # 處理動態權重 (與 calculate_final_scores 邏輯一致)
            final_weight = base_weight
            final_score = 0.0

            if item_id == "med_mix_method_and_time.s1":
                # 說明保可淨使用方式
                # 組合二 (有樂可舒): 保可淨佔 1 分 (因為樂可舒 s2 也會出現佔 1 分)
                # 組合一 (無樂可舒): 保可淨佔 2 分
                final_weight = 1.0 if has_dulcolax else 2.0
                final_score = final_weight if score_val else 0
            elif item_id in ["greeting_hello", "invite_to_sit"]:
                final_weight = 0.5 if is_case_A2 else 1.0
                final_score = final_weight if score_val else 0
            elif item_id == "emo_response":
                final_weight = 1.0 if is_case_A2 else 0.0
                final_score = final_weight if score_val else 0
            elif item_id == "satisfy_patient_infomation":
                final_weight = 4.0 if is_case_A2 else 3.0
                final_score = final_weight if score_val else 0
            elif item_id == "med_mix_method_and_time.s1":
                final_weight = 1.0 if has_dulcolax else 2.0
                final_score = final_weight if score_val else 0
            else:
                final_score = final_weight if score_val else 0

            # 建立 Item 結構
            item_detail = {
                "item_id": item_id,
                "item_name": criterion.get("item", item_id),
                "description": criterion.get("任務說明", "無說明"),
                "weight": final_weight,
                "user_score": final_score,
                "scoring_type": criterion.get("type", "Standard"),
                "relevant_dialogues": dialogue_map.get(item_id, []),
            }
            grouped_details[category]["items"].append(item_detail)

        # ====================================================
        # [新增] 手動加入「適切發問及引導」的合併項目
        # ====================================================

        # 1. 計算各子項得分狀態
        s1 = user_scores.get("proper_guidance_s1", 0)
        s2 = user_scores.get("proper_guidance_s2", 0)
        s3 = user_scores.get("proper_guidance_s3", 0)
        s4 = user_scores.get("proper_guidance_s4", 0)
        s5 = user_scores.get("proper_guidance_s5", 0)

        # 2. 計算邏輯分數
        # S2 和 S3 只要有一個達成，該分項就算分 (1分)
        s23_score = 1 if (s2 or s3) else 0

        # 總得分 (滿分 4)
        pg_total_score = s1 + s23_score + s4 + s5
        pg_weight = 4.0

        # 3. 組合說明文字 (使用 \n 換行，前端 CSS 需支援)
        # 這裡將邏輯清楚列出
        pg_description = (
            f"• 是否引導病人進到衛教對話 (1分)：{'✅' if s1 else '❌'}\n"
            f"• 確認檢查型態 或 引導判斷 (1分)：{'✅' if s23_score else '❌'}\n"
            f"• 是否確認病人檢查時間 (1分)：{'✅' if s4 else '❌'}\n"
            f"• 是否確認病人的額外用藥 (1分)：{'✅' if s5 else '❌'}"
        )

        # 4. 收集所有相關對話
        pg_dialogues = []
        for pid in proper_guidance_ids:
            pg_dialogues.extend(dialogue_map.get(pid, []))
        # 去重 (雖然對話可能重複被歸因，但在顯示時通常沒關係，若要乾淨可做 set)
        pg_dialogues = list(set(pg_dialogues))

        # 5. 加入到「醫療面談」類別
        if "醫療面談" in grouped_details:
            grouped_details["醫療面談"]["items"].append(
                {
                    "item_id": "proper_guidance_combined",
                    "item_name": "適切發問及引導以獲得正確且足夠的訊息",
                    "description": pg_description,
                    "weight": pg_weight,
                    "user_score": float(pg_total_score),
                    "scoring_type": "Composite Logic",
                    "relevant_dialogues": pg_dialogues,
                }
            )

        # ====================================================
        # B. 注入邏輯計算項目 (Logic/Computed Items)
        # 這是此模組特有的，如果是別的教案，會有完全不同的實作
        # ====================================================

        # --- 1. 檢閱藥歷 (Review Med History) - UI Items ---
        ui_items = [
            ("viewed_alltimes_ci", "檢閱「歷次清腸資訊」", 2.0),
            ("viewed_chiachi_med", "檢閱「本院用藥」", 3.0),
            ("viewed_med_allergy", "檢閱「藥物過敏史」", 1.0),
            ("viewed_disease_diag", "檢閱「疾病診斷」", 1.0),
            ("viewed_cloud_med", "檢閱「雲端藥歷」", 2.0),
        ]

        for field, name, w in ui_items:
            passed = (
                getattr(interaction_log, field, False) if interaction_log else False
            )
            score = w if passed else 0.0
            grouped_details["檢閱藥歷"]["items"].append(
                {
                    "item_id": field,
                    "item_name": name,
                    "description": "透過介面點擊檢閱",
                    "weight": w,
                    "user_score": score,
                    "scoring_type": "UI Interaction",
                    "relevant_dialogues": [],
                }
            )

        # --- 2. 人道專業 (Logic Formulas) ---
        # 表現尊重
        val_hello = is_passed("greeting_hello")
        val_sit = is_passed("invite_to_sit")
        pg_s1 = is_passed("proper_guidance_s1")
        has_respect = val_hello > 0 and val_sit > 0 and pg_s1 > 0
        respect_w = 2.0 if is_case_A2 else 3.0
        grouped_details["人道專業"]["items"].append(
            {
                "item_id": "logic_respect",
                "item_name": "表現尊重",
                "description": "同時達成：問好、請坐、引導衛教",
                "weight": respect_w,
                "user_score": respect_w if has_respect else 0.0,
                "scoring_type": "Logic Formula",
                "relevant_dialogues": [],
            }
        )

        # 同理心
        val_no_term = is_passed("no_use_term")
        val_ask_exp = is_passed("review_med_history_1")
        has_empathy = val_no_term > 0 and val_ask_exp > 0
        empathy_w = 1.0 if is_case_A2 else 2.0
        grouped_details["人道專業"]["items"].append(
            {
                "item_id": "logic_empathy",
                "item_name": "同理心(感同身受)",
                "description": "同時達成：無專業術語、詢問過往經驗",
                "weight": empathy_w,
                "user_score": empathy_w if has_empathy else 0.0,
                "scoring_type": "Logic Formula",
                "relevant_dialogues": [],
            }
        )

        # --- 3. 組織效率 ---
        # 優先順序
        val_sequence = self._check_organization_sequence_by_time(
            session_id, has_dulcolax, has_special_meds, db
        )
        grouped_details["組織效率"]["items"].append(
            {
                "item_id": "logic_sequence",
                "item_name": "按優先順序處置",
                "description": "飲食 -> 口服藥(若有) -> 清腸粉劑 -> 禁水 -> 其他用藥",
                "weight": 3.0,
                "user_score": val_sequence,
                "scoring_type": "Timestamp Logic",
                "relevant_dialogues": [],
            }
        )

        # 及時且適時
        chat_logs_q = (
            db.query(ChatLog)
            .filter(ChatLog.session_id == session_id)
            .order_by(ChatLog.time.asc())
            .all()
        )
        duration_minutes = 0
        if chat_logs_q:
            duration_minutes = (
                chat_logs_q[-1].time - chat_logs_q[0].time
            ).total_seconds() / 60.0
        val_time = 1.5 if (5 <= duration_minutes <= 9) else 0.5
        grouped_details["組織效率"]["items"].append(
            {
                "item_id": "logic_time",
                "item_name": "及時且適時",
                "description": f"總衛教時間：{duration_minutes:.1f} 分鐘 (目標 5-9 分鐘)",
                "weight": 1.5,
                "user_score": val_time,
                "scoring_type": "Time Logic",
                "relevant_dialogues": [],
            }
        )

        # 歷練而簡潔 (透過總分倒推)
        score_record = db.query(Scores).filter(Scores.session_id == session_id).first()
        final_org_score = (
            float(score_record.organization_efficiency_score) if score_record else 0.0
        )
        val_concise = max(0, final_org_score - val_sequence - val_time)
        grouped_details["組織效率"]["items"].append(
            {
                "item_id": "logic_concise",
                "item_name": "歷練而簡潔",
                "description": "基於諮商衛教、醫療面談、檢閱藥歷的綜合表現計算",
                "weight": 3.5,
                "user_score": val_concise,
                "scoring_type": "Complex Formula",
                "relevant_dialogues": [],
            }
        )

        # --- 4. 臨床判斷 ---
        # 理解程度
        val_satisfy = is_passed("satisfy_patient_infomation") * (
            4.0 if is_case_A2 else 3.0
        )
        satisfy_weight = 4.0 if is_case_A2 else 3.0
        val_judge_understand = (
            (val_satisfy / satisfy_weight * 1.0) if satisfy_weight > 0 else 0.0
        )
        grouped_details["臨床判斷"]["items"].append(
            {
                "item_id": "logic_judge_understand",
                "item_name": "能依病人狀況判斷對說明內容理解程度",
                "description": "基於「人道專業-需求滿足」得分計算",
                "weight": 1.0,
                "user_score": val_judge_understand,
                "scoring_type": "Derived Formula",
                "relevant_dialogues": [],
            }
        )

        # 開立合理
        val_ask_exp = is_passed("review_med_history_1") * 1.0
        val_judge_reasonable = val_ask_exp / 2.0
        grouped_details["臨床判斷"]["items"].append(
            {
                "item_id": "logic_judge_reasonable",
                "item_name": "能判斷清腸藥物開立是否合理",
                "description": "基於「詢問過往經驗」得分計算",
                "weight": 1.0,
                "user_score": val_judge_reasonable,
                "scoring_type": "Derived Formula",
                "relevant_dialogues": [],
            }
        )

        # --- 5. 整體臨床技能 ---
        overall_score = (
            float(score_record.overall_clinical_skills_score) if score_record else 0.0
        )
        grouped_details["整體臨床技能"]["items"].append(
            {
                "item_id": "overall_total_calc",
                "item_name": "整體臨床技能總評",
                "description": "包含：態度(3分)、整合能力(4分)、整體有效性(2分)之綜合計算",
                "weight": 9.0,
                "user_score": overall_score,
                "scoring_type": "Complex Formula",
                "relevant_dialogues": [],
            }
        )

        return grouped_details
