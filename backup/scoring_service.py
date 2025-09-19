# --- START OF FILE scoring_service.py ---
# basic prompt 跟 strong prompt 應該要分為 強參考 以及 弱參考
# 強參考代表 所說內容 語意 都要相符參考答案
# 弱參考代表 只要內容有表達出來就可以給過 先等等好了 我現在是在於 評分項目有沒有搜索到的問題 判錯的問題之後再解決


import json
import logging
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime
from typing import List, Dict # <--- 新增導入
from databases import AnswerLog, PrecomputedSessionAnswer, ScoringPromptLog, SessionUserMap, AgentSettings
from utils import generate_llm_response

logger = logging.getLogger(__name__)

SCORING_MODEL_NAME = "gemma3:4b"
STRONGER_SCORING_MODEL_NAME = "gemma3:4b" 

# 特殊藥物衛教指令對照表
MED_INSTRUCTIONS = {
    "KKK": "緩解便秘藥物(KKK)請勿停藥，應繼續服用；其他藥品應於服用清腸劑前2小時或清腸後6小時服用。",
    "RRR": "抗凝血劑(RRR)請依醫師指示停藥；其他藥品，應於服用清腸劑前2小時或清腸後6小時服用。",
    "QQQ": "抗血小板藥物(QQQ)請依醫師指示停藥；其他藥品，應於服用清腸劑前2小時或清腸後6小時服用。",
    "PPP": "降血糖藥物(PPP)請依醫師指示停藥；其他藥品，應於服用清腸劑前2小時或清腸後6小時服用。",
    "AAA": "降血糖藥物(AAA)請依醫師指示停藥；其他藥品，應於服用清腸劑前2小時或清腸後6小時服用。"
}

class ScoringService:
    def __init__(self, criteria_file='scoring_criteria.json'):
        logger.info("Initializing Scoring Service...")
        self.criteria = self._load_criteria(criteria_file)
        # 選擇一個適合中文的多語言模型
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.index, self.criteria_id_list = self._build_vector_index()
        logger.info("Scoring Service initialized successfully.")

    def _load_criteria(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_vector_index(self):
        """
        為每個評分項目(criterion)建立一個單一的、語意豐富的向量。
        """
        # documents_to_encode 列表儲存要被向量化的文本
        documents_to_encode = []
        # criteria_id_list 儲存與 documents_to_encode 相對應的評分項目ID
        # 這個列表的索引將作為 FAISS 中的 ID
        criteria_id_list = []

        logger.info("Building a richer vector index from criteria...")
        for criterion in self.criteria:
            # 將 item 描述和所有 example_answer 合併成一個字符串
            # 這樣可以创建一个包含該評分項所有核心概念的綜合性文件
            item_description = criterion.get('item', '')
            examples = " ".join(criterion.get('example_answer', []))
            
            # 合併後的文本，例如 "能做簡易飲食衛教 不可以吃奶製品... 檢查前三天..."
            combined_text = f"{item_description} {examples}"
            
            documents_to_encode.append(combined_text)
            criteria_id_list.append(criterion['id'])

        logger.info(f"Encoding {len(documents_to_encode)} combined criteria documents for vector index...")
        embeddings = self.model.encode(documents_to_encode, convert_to_tensor=False)
        
        # 建立 FAISS 索引
        dimension = embeddings.shape[1]
        # 我們現在可以直接用列表的索引 (0, 1, 2, ...) 作為 FAISS 的 ID
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        logger.info("Vector index built successfully.")
        # 返回索引本身和一個 ID 映射列表
        # 透過這個列表，我們可以將 FAISS 返回的數字索引轉換回評分項的字串 ID
        return index, criteria_id_list

    def find_relevant_criteria_ids(self, user_input: str, top_k: int = 5): # 建議稍微提高 top_k
        """根據使用者輸入，找到最相關的評分項目ID"""
        input_embedding = self.model.encode([user_input])
        # search 返回的 I 是 FAISS 的數字索引
        distances, I = self.index.search(input_embedding, top_k)
        
        relevant_ids = set()
        for idx in I[0]:
            if idx != -1:
                # 使用 self.criteria_id_list 將數字索引轉換為字串 ID
                relevant_ids.add(self.criteria_id_list[idx])
        
        logger.info(f"Found relevant criteria IDs: {list(relevant_ids)}")
        return list(relevant_ids)
    
    # <--- 新增輔助函式，用於集中處理 LLM 呼叫與記錄 --->
    async def _call_llm_and_log(self, session_id: str, item_id: str, prompt: str, model_name: str, db: Session) -> int:
        """呼叫 LLM 進行評分，並將 prompt 和 response 記錄到資料庫。"""
        try:
            response_text = await generate_llm_response(prompt, model_name)
            raw_response = response_text.strip()
            score = 1 if "1" in raw_response else 0

            log_entry = ScoringPromptLog(
                session_id=session_id,
                scoring_item_id=item_id,
                prompt_text=prompt,
                llm_response=raw_response,
                final_score=score
            )
            db.add(log_entry)
            
            logger.info(f"[{session_id}] Scored item '{item_id}' with score {score}. Raw response: '{raw_response[:50]}...'")
            return score
        except Exception as e:
            logger.error(f"[{session_id}] Error during LLM call or logging for item '{item_id}': {e}")
            # 發生錯誤時，返回 0 且不寫入資料庫，讓外層的 rollback 機制處理
            return 0
    
    async def process_user_inputs_for_scoring(self, session_id: str, chat_snippet: List[Dict], db: Session) -> List[str]:
        """處理多個使用者輸入並進行評分"""
        if not chat_snippet:
            return

        # 將對話片段格式化為單一字串，以提供上下文
        def format_snippet(snippet: List[Dict]) -> str:
            formatted_lines = []
            for item in snippet:
                role = "學員" if item['role'] == 'user' else "病患"
                formatted_lines.append(f"{role}: {item['message']}")
            return "\n".join(formatted_lines)

        # 格式化後的對話紀錄
        formatted_conversation = format_snippet(chat_snippet)
        
        logger.info(f"[{session_id}] Scoring conversation snippet: {formatted_conversation[:150]}...")
        
        # Vector Search 仍然可以使用合併後的字串來尋找相關項目
        relevant_ids = self.find_relevant_criteria_ids(formatted_conversation, top_k=5)
        if not relevant_ids:
            logger.info(f"[{session_id}] No relevant scoring criteria found.")
            return

        # 獲取已評分且通過的項目，避免重複評分 (邏輯不變)
        already_passed_stmt = db.query(AnswerLog.scoring_item_id).filter(
            AnswerLog.session_id == session_id,
            AnswerLog.score == 1
        )
        passed_ids = {row.scoring_item_id for row in already_passed_stmt.all()}
        
        # 獲取預計算的答案 (邏輯不變)
        precomputed_data = db.query(PrecomputedSessionAnswer).filter(PrecomputedSessionAnswer.session_id == session_id).first()
        
        newly_passed_ids = []

        for item_id in relevant_ids:
            if item_id in passed_ids:
                logger.info(f"[{session_id}] Skipping already passed item: {item_id}")
                continue
            
            criterion = next((item for item in self.criteria if item['id'] == item_id), None)
            if criterion:
                # score_item 內部會評分並更新 AnswerLog
                score = await self.score_item(session_id, formatted_conversation, criterion, db, precomputed_data)
                
                # 如果這次評分通過了，就加到「新通過」列表中
                if score == 1:
                    newly_passed_ids.append(item_id)
                    
        return newly_passed_ids

    async def score_item(self, session_id: str, conversation_context: str, criterion: dict, db: Session, precomputed_data):
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
            
            #logger.info(f"[{session_id}] Scoring result for '{item_id}': {score}")

            # 使用 SQLAlchemy Core 的 Insert on conflict update，更適合高併發
            stmt = sqlite_insert(AnswerLog).values(
                session_id=session_id,
                scoring_item_id=item_id,
                score=score,
                created_at=datetime.now()
            )
            # 如果發生唯一約束衝突 (session_id, scoring_item_id)，則更新 score
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=['session_id', 'scoring_item_id'],
                set_=dict(score=score)
            )
            db.execute(on_conflict_stmt)
            db.commit()
            
            return score

        except Exception as e:
            logger.error(f"[{session_id}] Error scoring item '{item_id}': {e}")
            db.rollback()
            return 0 # 出錯時返回 0

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
            logger.warning(f"[{session_id}] Precomputed data is not available for _check_confirm_med_time.")
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
            logger.warning(f"[{session_id}] Precomputed data is not available for _check_npo_time.")
            return 0
        
        prompt = f"""
        你是一個資深護理師，請判斷學員的衛教內容是否正確。
        [情境] 病人正確的禁水開始時間是 {precomputed_data.npo_start_time}。
        [學員回答]: {conversation_context}
        
        學員是否有清楚告知病人，從「{precomputed_data.npo_start_time}」這個時間點開始就完全不能再喝任何東西（包含水）？
        如果學員清楚說明了這個具體的時間點和禁水要求，只輸出 "1"。否則輸出 "0"。
        """
        score = await self._call_llm_and_log(session_id, 'clinical_npo_timing', prompt, SCORING_MODEL_NAME, db)
        
        if score == 1:
            logger.info(f"[{session_id}] _check_npo_time passed, automatically scoring npo_mention.")
            # 注意：這裡的聯動評分不會記錄 prompt，因為它是被動觸發的。
            # 如果需要記錄，則需要為 npo_mention 設計一個獨立的 prompt 邏輯。
            try:
                stmt = sqlite_insert(AnswerLog).values(
                    session_id=session_id,
                    scoring_item_id='npo_mention',
                    score=1,
                    created_at=datetime.now()
                )
                on_conflict_stmt = stmt.on_conflict_do_update(
                    index_elements=['session_id', 'scoring_item_id'],
                    set_=dict(score=1)
                )
                db.execute(on_conflict_stmt)
                # commit 會在外層的 score_item 統一處理
            except Exception as e:
                logger.error(f"[{session_id}] Failed to auto-score 'npo_mention': {e}")
                # 不需要 rollback，讓外層的 score_item 處理
        
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
            logger.warning(f"[{session_id}] Cannot find session map for _check_special_meds.")
            return 0
            
        agent_settings = db.query(AgentSettings).filter(AgentSettings.agent_code == session_map.agent_code).first()
        if not agent_settings or not agent_settings.med_code:
            logger.info(f"[{session_id}] Agent has no special meds to check. Skipping.")
            return 0
            
        med_code = agent_settings.med_code.strip()
        correct_instruction = MED_INSTRUCTIONS.get(med_code)
        
        if not correct_instruction:
            logger.warning(f"[{session_id}] Unknown med_code '{med_code}' in MED_INSTRUCTIONS map.")
            return 0

        prompt = f"""
        你是一位資深的臨床藥師，請根據病人的用藥情況和學員的衛教內容進行評分。
        
        [病人用藥情境]:
        病人正在服用特殊藥物：{med_code}。
        
        [此藥物正確的衛教指令]:
        "{correct_instruction}"
        
        [學員與病人的對話紀錄]:
        {conversation_context}
        
        [評分任務]:
        請判斷學員是否向病人提供了與上述「正確的衛教指令」語意相符的說明？
        如果學員的說明核心內容正確（例如，指出了藥物 {med_code} 需要停藥或繼續服用），請只輸出 "1"。
        如果學員的說明錯誤、不完整，或完全沒有提到如何處理藥物 {med_code}，請輸出 "0"。
        """
        return await self._call_llm_and_log(session_id, 'specify_special_meds', prompt, SCORING_MODEL_NAME, db)


    async def _check_diet_logic(self, session_id: str, conversation_context: str, precomputed_data: PrecomputedSessionAnswer, db: Session) -> int:
        if not precomputed_data:
            logger.warning(f"[{session_id}] Precomputed data is not available for _check_diet_logic.")
            return 0
        prompt = f"""
        你是一個資深護理師，請判斷學員的衛教內容是否正確。
        [情境] 檢查日期是 {precomputed_data.exam_day}。
        - 檢查前三天 ({precomputed_data.prev_3d}) 和前兩天 ({precomputed_data.prev_2d}) 應該要低渣飲食。
        - 檢查前一天 ({precomputed_data.prev_1d}) 應該要無渣流質飲食。
        [學員回答]: {conversation_context}
        
        學員是否給出了正確的飲食衛教？如果正確，只輸出 "1"。如果不正確，只輸出 "0"。
        """
        return await self._call_llm_and_log(session_id, 'diet_basic', prompt, SCORING_MODEL_NAME, db)

# 在 main.py 中實例化
# scoring_service = ScoringService()