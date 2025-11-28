# scenarios/colonoscopy_bowklean/scoring_logic.py
import json
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from datetime import datetime
from typing import List, Dict, Optional

from utils import generate_llm_response
from databases import AnswerLog, PrecomputedSessionAnswer, ScoringPromptLog, SessionUserMap, AgentSettings, ChatLog, Scores
from scenarios.colonoscopy_bowklean.config import (
    MODULE_ID, SCORING_CRITERIA_FILE, SCORING_MODEL_NAME, STRONGER_SCORING_MODEL_NAME,
    MED_INSTRUCTIONS, CATEGORY_TO_FIELD_MAP, COMPOSITE_SUB_ITEM_IDS
)
from scenarios.colonoscopy_bowklean.summary_logic import calculate_organization_efficiency_score_llm

logger = logging.getLogger(__name__)

class ColonoscopyBowkleanScoringLogic:
    def __init__(self):
        logger.info(f"Initializing Scoring Logic for module: {MODULE_ID}...")
        self.criteria = self._load_criteria(SCORING_CRITERIA_FILE)
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.index, self.criteria_id_list = self._build_vector_index()
        logger.info(f"Scoring Logic for module: {MODULE_ID} initialized successfully.")

    def _load_criteria(self, file_path):
        # file_path is relative to the project root, so adjust if needed
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_vector_index(self):
        documents_to_encode = [] 
        criteria_id_list = []

        logger.info(f"Building vector index for module {MODULE_ID} from criteria...")
        for criterion in self.criteria:
            item_description = criterion.get('item', '')
            examples = " ".join(criterion.get('example_answer', []))
            combined_text = f"{item_description} {examples}"
            
            documents_to_encode.append(combined_text)
            criteria_id_list.append(criterion['id'])

        logger.info(f"Encoding {len(documents_to_encode)} combined criteria documents for module {MODULE_ID}...")
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
        
        logger.info(f"Found relevant criteria IDs for module {MODULE_ID}: {list(relevant_ids)}")
        return list(relevant_ids)
    
    async def _call_llm_and_log(self, session_id: str, item_id: str, prompt: str, model_name: str, db: Session) -> int:
        """呼叫 LLM 進行評分，並將 prompt 和 response 記錄到資料庫。"""
        try:
            response_text = await generate_llm_response(prompt, model_name)
            raw_response = response_text.strip()
            score = 1 if "1" in raw_response else 0

            log_entry = ScoringPromptLog(
                session_id=session_id,
                module_id=MODULE_ID, # 使用模組專屬 ID
                scoring_item_id=item_id,
                prompt_text=prompt,
                llm_response=raw_response,
                final_score=score
            )
            db.add(log_entry)
            
            logger.info(f"[{session_id}] Scored item '{item_id}' for module {MODULE_ID} with score {score}. Raw response: '{raw_response[:50]}...'")
            return score
        except Exception as e:
            logger.error(f"[{session_id}] Error during LLM call or logging for item '{item_id}' (module {MODULE_ID}): {e}")
            return 0
    
    async def process_user_inputs_for_scoring(self, session_id: str, chat_snippet: List[Dict], db: Session) -> List[str]:
        """處理多個使用者輸入並進行評分"""
        if not chat_snippet:
            return []

        def format_snippet(snippet: List[Dict]) -> str:
            formatted_lines = []
            for item in snippet:
                role = "學員" if item['role'] == 'user' else "病患"
                formatted_lines.append(f"{role}: {item['message']}")
            return "\n".join(formatted_lines)

        formatted_conversation = format_snippet(chat_snippet)
        
        logger.info(f"[{session_id}] Scoring conversation snippet for module {MODULE_ID}: {formatted_conversation[:150]}...")
        
        relevant_ids = self.find_relevant_criteria_ids(formatted_conversation, top_k=5)
        if not relevant_ids:
            logger.info(f"[{session_id}] No relevant scoring criteria found for module {MODULE_ID}.")
            return []

        already_passed_stmt = db.query(AnswerLog.scoring_item_id).filter(
            AnswerLog.session_id == session_id,
            AnswerLog.score == 1
        )
        passed_ids = {row.scoring_item_id for row in already_passed_stmt.all()}
        
        precomputed_data = db.query(PrecomputedSessionAnswer).filter(PrecomputedSessionAnswer.session_id == session_id).first()
        
        newly_passed_ids = []

        for item_id in relevant_ids:
            if item_id in passed_ids:
                logger.info(f"[{session_id}] Skipping already passed item: {item_id} for module {MODULE_ID}.")
                continue
            
            criterion = next((item for item in self.criteria if item['id'] == item_id), None)
            if criterion:
                score = await self.score_item(session_id, formatted_conversation, criterion, db, precomputed_data)
                
                if score == 1:
                    newly_passed_ids.append(item_id)
                    
        return newly_passed_ids

    async def score_item(self, session_id: str, conversation_context: str, criterion: dict, db: Session, precomputed_data: Optional[PrecomputedSessionAnswer]):
        """對單個評分項目進行評分並儲存結果"""
        item_id = criterion['id']
        item_type = criterion.get('type', 'RAG')
        
        score = 0
        try:
            if item_type == "RAG":
                score = await self._check_rag_basic(session_id, conversation_context, criterion, db)
            elif item_type == "RAG_4b":
                score = await self._check_rag_strong(session_id, conversation_context, criterion, db)
            elif item_type == "RAG_confirm_med_time":
                score = await self._check_confirm_med_time(session_id, conversation_context, criterion, precomputed_data, db)
            elif item_type == "RAG_confirm_npo":
                 score = await self._check_npo_time(session_id, conversation_context, precomputed_data, db)
            elif item_type == "RAG_diet_logic":
                score = await self._check_diet_logic(session_id, conversation_context, precomputed_data, db)
            elif item_type == "RAG_med_usage_method":
                score = await self._check_med_usage_method(session_id, conversation_context, criterion, db)
            elif item_type == "RAG_special_meds":
                score = await self._check_special_meds(session_id, conversation_context, db)
            else: # 對於未定義的類型，使用基礎 RAG 邏輯
                score = await self._check_rag_basic(session_id, conversation_context, criterion, db)
            
            stmt = sqlite_insert(AnswerLog).values(
                session_id=session_id,
                module_id=MODULE_ID, # 新增 module_id
                scoring_item_id=item_id,
                score=score,
                created_at=datetime.now()
            )
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=['session_id', 'scoring_item_id'],
                set_=dict(score=score)
            )
            db.execute(on_conflict_stmt)
            db.commit()
            
            return score

        except Exception as e:
            logger.error(f"[{session_id}] Error scoring item '{item_id}' for module {MODULE_ID}: {e}")
            db.rollback()
            return 0

    async def _check_rag_basic(self, session_id: str, conversation_context: str, criterion: dict, db: Session) -> int:
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
        return await self._call_llm_and_log(session_id, criterion['id'], prompt, SCORING_MODEL_NAME, db)
    
    async def _check_rag_strong(self, session_id: str, conversation_context: str, criterion: dict, db: Session) -> int:
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
        return await self._call_llm_and_log(session_id, criterion['id'], prompt, STRONGER_SCORING_MODEL_NAME, db)
    
    async def _check_confirm_med_time(self, session_id: str, conversation_context: str, criterion: dict, precomputed_data: PrecomputedSessionAnswer, db: Session) -> int:
        """檢查學員是否正確說明了第一包和第二包清腸劑的服用時間。"""
        if not precomputed_data:
            logger.warning(f"[{session_id}] Precomputed data is not available for _check_confirm_med_time in module {MODULE_ID}.")
            return 0
        
        prompt = ""
        if criterion['id'] == 'clinical_med_timing_1':
            prompt = f"""
            你是一個資深護理師，請判斷學員的衛教內容是否正確。
            [情境] 病人應該在檢查前一天 ({precomputed_data.prev_1d}) 的下午5點服用第一包藥。
            [學員回答]: {conversation_context}
            
            學員是否有清楚且正確地告知病人第一包藥的服用日期和時間？如果正確，只輸出 "1"。如果不正確，只輸出 "0"。
            """
        elif criterion['id'] == 'clinical_med_timing_2':
            prompt = f"""
            你是一個資深護理師，請判斷學員的衛教內容是否正確。
            [情境] 病人應該在檢查當天 ({precomputed_data.exam_day}) 的 {precomputed_data.second_dose_time} 服用第二包藥。
            [學員回答]: {conversation_context}
            
            學員是否有清楚且正確地告知病人第二包藥的服用日期和時間？如果正確，只輸出 "1"。如果不正確，只輸出 "0"。
            """
        else:
            return 0

        return await self._call_llm_and_log(session_id, criterion['id'], prompt, SCORING_MODEL_NAME, db)

    async def _check_npo_time(self, session_id: str, conversation_context: str, precomputed_data: PrecomputedSessionAnswer, db: Session) -> int:
        """檢查學員是否清楚說明禁水時間點，並連帶給予 npo_mention 分數。"""
        if not precomputed_data:
            logger.warning(f"[{session_id}] Precomputed data is not available for _check_npo_time in module {MODULE_ID}.")
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
        score = await self._call_llm_and_log(session_id, 'clinical_npo_timing', prompt, SCORING_MODEL_NAME, db)
        
        if score == 1:
            logger.info(f"[{session_id}] _check_npo_time passed, automatically scoring npo_mention for module {MODULE_ID}.")
            try:
                stmt = sqlite_insert(AnswerLog).values(
                    session_id=session_id,
                    module_id=MODULE_ID, # 新增 module_id
                    scoring_item_id='npo_mention',
                    score=1,
                    created_at=datetime.now()
                )
                on_conflict_stmt = stmt.on_conflict_do_update(
                    index_elements=['session_id', 'scoring_item_id'],
                    set_=dict(score=1)
                )
                db.execute(on_conflict_stmt)
            except Exception as e:
                logger.error(f"[{session_id}] Failed to auto-score 'npo_mention' for module {MODULE_ID}: {e}")
        
        return score

    async def _check_med_usage_method(self, session_id: str, conversation_context: str, criterion: dict, db: Session) -> int:
        """檢查學員是否正確說明清腸劑的泡製方法。"""
        prompt = f"""
        你是一個資深護理師，請判斷學員是否正確說明了清腸劑的泡製與服用方式。
        
        [正確方法參考]: 將一包藥粉倒入150c.c.的常溫水中，攪拌至完全溶解後立即喝完。
        
        [學員的對話]:
        {conversation_context}
        
        學員是否有提到類似上述的泡製和服用方法（水量、攪拌、喝完）？只要有清楚說明過一次即可，無論是針對第一包還是第二包。
        如果說明正確，只輸出 "1"。如果說明不完整或不正確，只輸出 "0"。
        """
        return await self._call_llm_and_log(session_id, criterion['id'], prompt, SCORING_MODEL_NAME, db)

    async def _check_special_meds(self, session_id: str, conversation_context: str, db: Session) -> int:
        """根據病人的特殊用藥，判斷學員的衛教是否正確。"""
        session_map = db.query(SessionUserMap).filter(SessionUserMap.session_id == session_id).first()
        if not session_map:
            logger.warning(f"[{session_id}] Cannot find session map for _check_special_meds in module {MODULE_ID}.")
            return 0
            
        agent_settings = db.query(AgentSettings).filter(AgentSettings.agent_code == session_map.agent_code).first()
        if not agent_settings or not agent_settings.med_code:
            logger.info(f"[{session_id}] Agent has no special meds to check for module {MODULE_ID}. Skipping.")
            return 0
            
        med_codes = agent_settings.med_code.strip().split(';') # 假設 med_code 可能是多個，用分號分隔
        
        # 遍歷所有可能的 med_code 類型，檢查是否有匹配
        for med_code_part in med_codes:
            # 提取代碼部分，例如從 "抗凝血劑 S1" 提取 "S1"
            code_match = re.search(r'(S\d|X\d|N)', med_code_part)
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
                    score = await self._call_llm_and_log(session_id, 'specify_special_meds', prompt, SCORING_MODEL_NAME, db)
                    if score == 1:
                        return 1 # 只要一個通過就返回1

        logger.warning(f"[{session_id}] No matching special med instruction found or all checks failed for module {MODULE_ID}.")
        return 0 # 如果沒有匹配的指令或所有檢查都失敗，則返回 0


    async def _check_diet_logic(self, session_id: str, conversation_context: str, precomputed_data: PrecomputedSessionAnswer, db: Session) -> int:
        """檢查學員是否正確說明了飲食衛教。"""
        if not precomputed_data:
            logger.warning(f"[{session_id}] Precomputed data is not available for _check_diet_logic in module {MODULE_ID}.")
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
        return await self._call_llm_and_log(session_id, 'diet_basic', prompt, SCORING_MODEL_NAME, db)
    
    async def _check_med_mix_method(self, session_id: str, conversation_context: str, criterion: dict, db: Session) -> int:
        """檢查學員是否正確說明了特定一包清腸劑的泡製方法。"""
    
        # 從 criterion ID 判斷是第一包還是第二包
        dose_number_text = "第一包" if "first" in criterion['id'] else "第二包"
    
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
        return await self._call_llm_and_log(session_id, criterion['id'], prompt, SCORING_MODEL_NAME, db)
    
    async def _check_first_med_mix_time(self, session_id: str, conversation_context: str, criterion: dict, db: Session) -> int:
        """檢查學員是否說明了口服瀉藥錠劑（如樂可舒）的服用方法。"""
        
        prompt = f"""
        你是一位臨床藥師，請判斷學員的衛教內容是否完整。

        [情境與任務]:
        在某些清腸療程中，除了喝「保可淨」粉劑外，還需要搭配服用「口服瀉藥錠劑」，例如 '樂可舒(Dulcolax)'。
        你的任務是判斷學員是否有提到關於服用這種「錠劑」的指示。

        [學員與病人的對話紀錄]:
        ---
        {conversation_context}
        ---

        [你的判斷任務]:
        請閱讀以上對話，判斷學員是否有提到需要服用藥丸或錠劑型式的瀉藥（不只是喝的粉劑）？

        - 如果學員有提到關於「錠劑」的服用說明，請只輸出 "1"。
        - 如果學員的衛教內容只提到了用喝的「保可淨」粉劑，完全沒提到任何錠劑或藥丸，請只輸出 "0"。

        [你的判斷 (只輸出 1 或 0)]:
        """
        return await self._call_llm_and_log(session_id, criterion['id'], prompt, SCORING_MODEL_NAME, db)
    
    # --- 新增：最終分數計算邏輯 ---
    async def calculate_final_scores(self, session_id: str, db: Session) -> Dict[str, str]:
        """
        計算該 Session 的最終分數。
        包含標準項目的累加以及特殊複合規則（如 S1-S4 藥物衛教）的計算。
        """
        logger.info(f"[{session_id}] Calculating final scores using module logic: {MODULE_ID}")
        
        # 1. 獲取所有已得分項目 ID
        passed_items_query = db.query(AnswerLog.scoring_item_id).filter(
            AnswerLog.session_id == session_id,
            AnswerLog.score == 1
        ).all()
        passed_item_ids = {item.scoring_item_id for item in passed_items_query}
        
        # 2. 獲取 UI 互動產生的檢閱藥歷分數 (從 Scores 表讀取暫存值)
        #    注意：main.py 在 end_chat 時會先存一次 UI 產生的 review_med_history_score
        ui_score_record = db.query(Scores).filter(Scores.session_id == session_id).first()
        ui_review_score = float(ui_score_record.review_med_history_score) if ui_score_record else 0.0
        
        # 3. 獲取對話紀錄 (用於計算時間和順序)
        chat_logs = db.query(ChatLog).filter(ChatLog.session_id == session_id).order_by(ChatLog.time.asc()).all()
        
        # --- 初始化各類別分數 ---
        scores = {key: 0.0 for key in CATEGORY_TO_FIELD_MAP.values()}
        # 檢閱藥歷 = UI分數
        scores["review_med_history_score"] = ui_review_score
        
        # --- 輔助函式：檢查是否通過 ---
        def is_passed(item_id):
            return 1 if item_id in passed_item_ids else 0

        # 3. 建立 ID 到 Criterion 的快速查找表
        criteria_map = {item['id']: item for item in self.criteria}

        # 4. 計算【標準項目】的分數 (排除複合規則子項目)
        for item_id in passed_item_ids:
            if item_id in COMPOSITE_SUB_ITEM_IDS:
                continue # 跳過複合規則的子項目，稍後單獨計算
            
            criterion = criteria_map.get(item_id)
            if criterion:
                weight = criterion.get('weight', 0.0)
                category = criterion.get('category')
                if category in CATEGORY_TO_FIELD_MAP:
                    field_name = CATEGORY_TO_FIELD_MAP[category]
                    scores[field_name] += weight

        # 5. 根據規則計算【複合項目】的分數 (這是此模組特有的邏輯)
        
        # (A) 適切發問及引導 (Medical Interview) - 總分 4 分
        # 規則: S2 和 S3 是 OR 閘 (共1分)，S1, S4, S5 各 1 分
        pg_s1 = is_passed('proper_guidance_s1')
        pg_s2 = is_passed('proper_guidance_s2')
        pg_s3 = is_passed('proper_guidance_s3')
        pg_s4 = is_passed('proper_guidance_s4')
        pg_s5 = is_passed('proper_guidance_s5')
        
        pg_s2_s3_score = 1 if (pg_s2 + pg_s3) > 0 else 0
        proper_guidance_total = pg_s1 + pg_s4 + pg_s5 + pg_s2_s3_score # Max 4

        # 加回醫療面談分數
        scores["medical_interview_score"] += proper_guidance_total
        logger.info(f"適切發問及引導得分: {proper_guidance_total}")
               
        # (B) 藥物使用時機及方式 (Counseling)
        # 規則: First(S1, S2) + Second(S1) -> 這裡 criteria 似乎有點變化，依 JSON 為準
        # 假設 JSON 定義了 first_med...s1, s2 和 second...s1
        med_mix_s1 = is_passed('first_med_mix_method_and_time.s1')
        med_mix_s2 = is_passed('first_med_mix_method_and_time.s2')
        med_mix_s3 = is_passed('second_med_mix_method_and_time.s1')
        # 根據 JSON權重加總 (各0.5)
        scores["counseling_edu_score"] += (med_mix_s1 * 0.5 + med_mix_s2 * 0.5 + med_mix_s3 * 0.5)

        # (C) 水分補充與理想狀態 (Counseling)
        hydro_s1 = is_passed('hydration_and_goal.s1')
        hydro_s2 = is_passed('hydration_and_goal.s2')
        scores["counseling_edu_score"] += (hydro_s1 * 1.0 + hydro_s2 * 1.0)

        # --- 6. 計算【人道專業 (Humanitarian)】 (依賴其他項目) ---
        # 規則 1: 表現尊重 (3分) -> 問好(1) + 坐下(1) + (適切發問(4)正規化為1)
        respect_score = (
            is_passed('greeting_hello') * 1 + 
            is_passed('invite_to_sit') * 1 + 
            (proper_guidance_total / 4.0) * 1  # 正規化
        )
        
        # 規則 2: 同理心 (2分) -> 無術語(1) + 詢問經驗(1)
        # 注意: review_med_history_1 在 JSON 是 "詢問經驗"，權重2，但在這裡是同理心+1
        empathy_score = (
            is_passed('no_use_term') * 1 +
            is_passed('review_med_history_1') * 1
        )
        
        scores["humanitarian_score"] += (respect_score + empathy_score)


        # --- 7. 計算【組織效率 (Organization Efficiency)】 (複雜邏輯) ---
        
        # (A) 按優先順序處置 (3分)
        # 使用 LLM 判斷順序 (飲食->口服->粉劑->禁水->其他)
        full_history_text = "\n".join([f"{'學員' if l.role=='user' else '病患'}: {l.text}" for l in chat_logs])
        # 呼叫 summary_logic 中的函數 (回傳 0-6 分)
        seq_llm_score = await calculate_organization_efficiency_score_llm(full_history_text)
        # 轉換規則: 如果順序大致正確(>4)得3分，否則1分
        sequence_score = 3.0 if seq_llm_score >= 4 else 1.0
        
        # (B) 即時且適時 (1.5分)
        time_score = 0.5
        if chat_logs:
            start_time = chat_logs[0].time
            end_time = chat_logs[-1].time
            duration_minutes = (end_time - start_time).total_seconds() / 60.0
            if 5 <= duration_minutes <= 9:
                time_score = 1.5
            logger.info(f"衛教時間: {duration_minutes:.2f} 分鐘, 得分: {time_score}")
        
        # (C) 歷練而簡潔 (3.5分)
        # 公式: 諮商衛教/6 + 醫療面談子項/6 + 檢閱藥歷子項/7
        
        # C-1: 諮商衛教總分
        counseling_part = scores["counseling_edu_score"] / 6.0
        
        # C-2: 醫療面談子項 (確認本人+1, 適切發問+5) -> 這裡 Proper Guidance 算 5 分?
        # 依 Prompt: "適切發問及引導...(+5分)"。我們先前算它是4分。這裡依公式需求視為5分(滿分)
        # 假設如果 proper_guidance_total 滿分4分，這裡就給5分；按比例放大
        pg_scaled_score = (proper_guidance_total / 4.0) * 5.0
        interview_subset_score = is_passed('confirm_self_use') * 1 + pg_scaled_score
        interview_part = interview_subset_score / 6.0
        
        # C-3: 檢閱藥歷子項 (查閱經驗+2, 查閱院內+3, 查閱院外+2) -> 總分7
        # 假設 ui_review_score 已經包含了 UI 查閱的分數 (院內3+院外2=5)
        # review_med_history_1 是對話查閱經驗 (+2)
        # 我們將 UI分數 + 對話分數 視為分子
        # (注意: review_med_history_1 在 JSON 權重是 2)
        review_history_subset = ui_review_score + (is_passed('review_med_history_1') * 2)
        # 這裡上限可能超過7 (如果UI有點滿)，做個 clamp 比較保險
        review_history_subset = min(review_history_subset, 7.0)
        review_part = review_history_subset / 7.0
        
        concise_score = (counseling_part + interview_part + review_part) * 3.5 # 這裡是否還要乘係數?
        # Prompt 寫: "歷練而簡潔 +3.5分 (公式...)"，這通常表示公式的結果正規化後佔 3.5 分
        # 公式內的除法已經是正規化 (例如 /6)，所以三個正規化值相加 (最大3.0)，再乘?
        # 讓我們假設公式是: (Score1/6 + Score2/6 + Score3/7) / 3 * 3.5
        # 或是直接加總? 如果各項滿分都是1，加起來是3。為了讓總分是3.5，需要 * (3.5/3)
        concise_score = (counseling_part + interview_part + review_part) / 3.0 * 3.5
        
        scores["organization_efficiency_score"] += (sequence_score + time_score + concise_score)


        # --- 8. 計算【臨床判斷 (Clinical Judgment)】 (額外計算) ---
        # 規則 1: 判斷理解程度 (1分) -> `satisfy_patient_infomation` (JSON權重3) / 3
        judge_understand_score = (is_passed('satisfy_patient_infomation') * 3.0) / 3.0
        
        # 規則 2: 判斷開立合理 (1分) -> `review_med_history_1` (JSON權重2) / 2
        judge_reasonable_score = (is_passed('review_med_history_1') * 2.0) / 2.0
        
        scores["clinical_judgment_score"] += (judge_understand_score + judge_reasonable_score)


        # --- 9. 計算【整體臨床技能 (Overall Clinical Skills)】 ---
        
        # (A) 態度 (3分): 醫療面談/6 + 人道專業/6
        attitude_score = (scores["medical_interview_score"] / 6.0 + scores["humanitarian_score"] / 6.0) / 2.0 * 3.0
        # 這裡假設兩個正規化分數平均後，佔 3 分
        
        # (B) 整合能力 (4分): 檢閱藥歷/6 + 醫療面談子項2/6 + 臨床判斷/6
        # 檢閱藥歷總分: scores["review_med_history_score"] (含UI+對話)
        # 醫療面談子項2: 確認本人(1) + 適切發問(4) + 無術語(1) = 6
        interview_subset2 = is_passed('confirm_self_use') + proper_guidance_total + is_passed('no_use_term')
        
        integration_score = (
            (scores["review_med_history_score"] / 6.0) + 
            (interview_subset2 / 6.0) + 
            (scores["clinical_judgment_score"] / 6.0)
        ) / 3.0 * 4.0
        
        # (C) 整體有效性 (2分): 6大類平均 / 3 ?? 
        # Prompt: [6類得分/9 的總和] / 3 ? 
        # 假設意思是：將前6大類的分數都正規化 (除以9，假設滿分是9?)，然後平均，再佔2分
        # 讓我們假設各類別滿分約為 6~9 分。
        sum_normalized = (
            scores["review_med_history_score"] +
            scores["medical_interview_score"] +
            scores["counseling_edu_score"] +
            scores["humanitarian_score"] +
            scores["organization_efficiency_score"] +
            scores["clinical_judgment_score"]
        ) / 9.0 # 這裡除以9可能是因為有些類別滿分接近9
        
        effectiveness_score = (sum_normalized / 6.0) * 2.0 # 平均後佔2分
        # 修正: Prompt 寫 [Sum of (Score/9)] / 3。我們先照 Prompt 寫
        effectiveness_score = sum_normalized / 3.0 # 這樣如果滿分，結果會是 6/3 = 2。符合。
        
        scores["overall_clinical_skills_score"] += (attitude_score + integration_score + effectiveness_score)

        # --- 10. 計算總分並格式化 ---
        total_score = sum(scores.values())
        
        # 將所有數值轉為字串格式
        result = {key: str(round(value, 2)) for key, value in scores.items()}
        # 加上總分 (注意: overall 算在總分裡嗎? 通常 Overall 是獨立指標，不疊加進 Total)
        # 如果 Overall 是包含在 Total 裡的，那就重複計算了。
        # 通常 Total Score 是前 6 項的總和。Overall 是另外的評價。
        # 根據一般邏輯，Total 不應包含 Overall。
        real_total = (
            scores["review_med_history_score"] +
            scores["medical_interview_score"] +
            scores["counseling_edu_score"] +
            scores["humanitarian_score"] +
            scores["organization_efficiency_score"] +
            scores["clinical_judgment_score"]
        )
        result['total_score'] = str(round(real_total, 2))
        
        logger.info(f"[{session_id}] Final Scores: {result}")
        return result