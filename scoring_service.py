# --- START OF FILE scoring_service.py ---

import json
import logging
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime
from typing import List, Dict # <--- 新增導入
from databases import AnswerLog, PrecomputedSessionAnswer
from utils import generate_llm_response

logger = logging.getLogger(__name__)

SCORING_MODEL_NAME = "gemma3:1b-it-fp16"
STRONGER_SCORING_MODEL_NAME = "gemma:4b" 

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
        self.index, self.id_map = self._build_vector_index()
        logger.info("Scoring Service initialized successfully.")

    def _load_criteria(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_vector_index(self):
        # 將所有 example_answer 扁平化並編碼
        all_examples = []
        id_map = {}
        idx_counter = 0
        for item in self.criteria:
            for example in item.get('example_answer', []):
                all_examples.append(example)
                id_map[idx_counter] = item['id']
                idx_counter += 1
        
        logger.info(f"Encoding {len(all_examples)} example answers for vector index...")
        embeddings = self.model.encode(all_examples, convert_to_tensor=False)
        
        # 建立 FAISS 索引
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIDMap(index)
        ids = np.array(range(len(all_examples))).astype('int64')
        index.add_with_ids(embeddings, ids)
        
        logger.info("Vector index built.")
        return index, id_map

    def find_relevant_criteria_ids(self, user_input: str, top_k: int = 3):
        """根據使用者輸入，找到最相關的評分項目ID"""
        input_embedding = self.model.encode([user_input])
        _, I = self.index.search(input_embedding, top_k)
        
        # 獲取唯一的評分項目 ID
        relevant_ids = set()
        for idx in I[0]:
            if idx != -1: # FAISS might return -1 if not enough neighbors are found
                relevant_ids.add(self.id_map[idx])
        return list(relevant_ids)
    
    async def process_user_inputs_for_scoring(self, session_id: str, chat_snippet: List[Dict], db: Session):
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

        for item_id in relevant_ids:
            if item_id in passed_ids:
                logger.info(f"[{session_id}] Skipping already passed item: {item_id}")
                continue
            
            criterion = next((item for item in self.criteria if item['id'] == item_id), None)
            if criterion:
                # 傳遞格式化後的完整對話
                await self.score_item(session_id, formatted_conversation, criterion, db, precomputed_data)

    async def score_item(self, session_id: str, conversation_context: str, criterion: dict, db: Session, precomputed_data):
        """對單個評分項目進行評分並儲存結果"""
        item_id = criterion['id']
        item_type = criterion.get('type', 'RAG')
        
        score = 0
        try:
            if item_type == "RAG":
                score = await self._check_rag_basic(conversation_context, criterion)
            elif item_type == "RAG_4b":
                score = await self._check_rag_strong(conversation_context, criterion)
            elif item_type == "RAG_confirm_med_time":
                score = await self._check_confirm_med_time(conversation_context, criterion, precomputed_data)
            elif item_type == "RAG_confirm_npo":
                 score = await self._check_npo_time(session_id, conversation_context, precomputed_data, db)
            elif item_type == "RAG_diet_logic":
                score = await self._check_diet_logic(conversation_context, precomputed_data)
            elif item_type == "RAG_med_usage_method":
                score = await self._check_med_usage_method(conversation_context, criterion)
            elif item_type == "RAG_special_meds":
                score = await self._check_special_meds(session_id, conversation_context, db)
            else: # 對於未定義的類型，使用基礎 RAG 邏輯
                score = await self._check_rag_basic(conversation_context, criterion)
            
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

        except Exception as e:
            logger.error(f"[{session_id}] Error scoring item '{item_id}': {e}")
            db.rollback()

    async def _check_rag_basic(self, conversation_context: str, criterion: dict) -> int:
        """使用 LLM 進行基礎 RAG 評分"""
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
        response = await generate_llm_response(prompt, SCORING_MODEL_NAME)
        #要有log輸出
        logger.info(f"評分結果 - 項目: '{criterion['item']}', 原始回應: {response.strip()}")
        
        return 1 if "1" in response.strip() else 0
    
    async def _check_rag_strong(self, conversation_context: str, criterion: dict) -> int:
        """
        NEW: 使用更強的 gemma:4b 模型進行評分，功能與 RAG 相同。
        """
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
        # 使用更強的模型進行評分
        response = await generate_llm_response(prompt, STRONGER_SCORING_MODEL_NAME)
        logger.info(f"強力評分結果 - 項目: '{criterion['item']}', 原始回應: {response.strip()}")
        return 1 if "1" in response.strip() else 0
    
    async def _check_confirm_med_time(self, conversation_context: str, criterion: dict, precomputed_data: PrecomputedSessionAnswer) -> int:
        """
        NEW: 檢查學員是否正確說明了第一包和第二包清腸劑的服用時間。
        """
        logger.info(f"Running custom check for: RAG_confirm_med_time ({criterion['id']})")
        if not precomputed_data:
            logger.warning("Precomputed data is not available for _check_confirm_med_time.")
            return 0
        
        prompt = ""
        if criterion['id'] == 'clinical_med_timing_1':
            correct_answer = f"第一包清腸劑(藥)服藥時間為 {precomputed_data.prev_1d}，下午5點。"
            prompt = f"""
            你是一個資深護理師，請判斷學員的衛教內容是否正確。
            [情境] 病人應該在檢查前一天 ({precomputed_data.prev_1d}) 的下午5點服用第一包藥。
            [學員回答]: {conversation_context}
            
            學員是否有清楚且正確地告知病人第一包藥的服用日期和時間？如果正確，只輸出 "1"。如果不正確，只輸出 "0"。
            """
        elif criterion['id'] == 'clinical_med_timing_2':
            correct_answer = f"第二包清腸劑(藥)服藥時間為 {precomputed_data.exam_day}，{precomputed_data.second_dose_time}。"
            prompt = f"""
            你是一個資深護理師，請判斷學員的衛教內容是否正確。
            [情境] 病人應該在檢查當天 ({precomputed_data.exam_day}) 的 {precomputed_data.second_dose_time} 服用第二包藥。
            [學員回答]: {conversation_context}
            
            學員是否有清楚且正確地告知病人第二包藥的服用日期和時間？如果正確，只輸出 "1"。如果不正確，只輸出 "0"。
            """
        else:
            return 0 # 不屬於此函式處理的項目

        response = await generate_llm_response(prompt, SCORING_MODEL_NAME)
        logger.info(f"服藥時間評分 - 項目: '{criterion['id']}', 正確答案: '{correct_answer}', 原始回應: {response.strip()}")
        return 1 if "1" in response.strip() else 0

    async def _check_npo_time(self, session_id: str, conversation_context: str, precomputed_data: PrecomputedSessionAnswer, db: Session) -> int:
        """
        UPDATED: 檢查學員是否清楚說明禁水時間點，並連帶給予 npo_mention 分數。
        """
        logger.info("Running custom check for: RAG_confirm_npo")
        if not precomputed_data:
            logger.warning("Precomputed data is not available for _check_npo_time.")
            return 0
        
        prompt = f"""
        你是一個資深護理師，請判斷學員的衛教內容是否正確。
        [情境] 病人正確的禁水開始時間是 {precomputed_data.npo_start_time}。
        [學員回答]: {conversation_context}
        
        學員是否有清楚告知病人，從「{precomputed_data.npo_start_time}」這個時間點開始就完全不能再喝任何東西（包含水）？
        如果學員清楚說明了這個具體的時間點和禁水要求，只輸出 "1"。否則輸出 "0"。
        """
        response = await generate_llm_response(prompt, SCORING_MODEL_NAME)
        score = 1 if "1" in response.strip() else 0
        
        # 聯動評分：如果此項達成，代表「npo_mention」也達成了
        if score == 1:
            logger.info(f"[{session_id}] _check_npo_time passed, automatically scoring npo_mention.")
            try:
                stmt = sqlite_insert(AnswerLog).values(
                    session_id=session_id,
                    scoring_item_id='npo_mention', # 這是 scoring_criteria.json 中對應的 id
                    score=1,
                    created_at=datetime.now()
                )
                on_conflict_stmt = stmt.on_conflict_do_update(
                    index_elements=['session_id', 'scoring_item_id'],
                    set_=dict(score=1)
                )
                db.execute(on_conflict_stmt)
                db.commit()
            except Exception as e:
                logger.error(f"[{session_id}] Failed to auto-score 'npo_mention': {e}")
                db.rollback()
        
        logger.info(f"禁水時間評分 - 項目: 'clinical_npo_timing', 正確時間: '{precomputed_data.npo_start_time}', 原始回應: {response.strip()}, 分數: {score}")
        return score

    async def _check_med_usage_method(self, conversation_context: str, criterion: dict) -> int:
        """
        NEW: 檢查學員是否正確說明清腸劑的泡製方法。此檢查通用於第一包和第二包。
        """
        logger.info("Running custom check for: RAG_med_usage_method")
        
        prompt = f"""
        你是一個資深護理師，請判斷學員是否正確說明了清腸劑的泡製與服用方式。
        
        [正確方法參考]: 將一包藥粉倒入150c.c.的常溫水中，攪拌至完全溶解後立即喝完。
        
        [學員的對話]:
        {conversation_context}
        
        學員是否有提到類似上述的泡製和服用方法（水量、攪拌、喝完）？只要有清楚說明過一次即可，無論是針對第一包還是第二包。
        如果說明正確，只輸出 "1"。如果說明不完整或不正確，只輸出 "0"。
        """
        response = await generate_llm_response(prompt, SCORING_MODEL_NAME)
        logger.info(f"藥物泡製方法評分 - 項目: '{criterion['id']}', 原始回應: {response.strip()}")
        return 1 if "1" in response.strip() else 0

    async def _check_special_meds(self, session_id: str, conversation_context: str, db: Session) -> int:
        """
        NEW: 根據病人的特殊用藥，判斷學員的衛教是否正確。
        """
        logger.info("Running custom check for: RAG_special_meds")
        
        # 1. 透過 session_id 找到 agent_code
        session_map = db.query(SessionUserMap).filter(SessionUserMap.session_id == session_id).first()
        if not session_map:
            logger.warning(f"[{session_id}] Cannot find session map for _check_special_meds.")
            return 0
            
        # 2. 透過 agent_code 找到 agent 的設定
        agent_settings = db.query(AgentSettings).filter(AgentSettings.agent_code == session_map.agent_code).first()
        if not agent_settings or not agent_settings.med_code:
            logger.info(f"[{session_id}] Agent has no special meds to check. Skipping.")
            return 0 # 如果病人沒有特殊用藥，此項不適用
            
        med_code = agent_settings.med_code.strip()
        correct_instruction = MED_INSTRUCTIONS.get(med_code)
        
        if not correct_instruction:
            logger.warning(f"[{session_id}] Unknown med_code '{med_code}' in MED_INSTRUCTIONS map.")
            return 0

        # 3. 建立提示，要求 LLM 進行判斷
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
        response = await generate_llm_response(prompt, SCORING_MODEL_NAME)
        logger.info(f"特殊藥物評分 - 藥物: '{med_code}', 正確指令: '{correct_instruction}', 原始回應: {response.strip()}")
        return 1 if "1" in response.strip() else 0


    async def _check_diet_logic(self, conversation_context: str, precomputed_data: PrecomputedSessionAnswer) -> int:
        # (此函式保留不變)
        logger.info("Running custom check for: RAG_diet_logic")
        if not precomputed_data:
            logger.warning("Precomputed data is not available for _check_diet_logic.")
            return 0
        prompt = f"""
        你是一個資深護理師，請判斷學員的衛教內容是否正確。
        [情境] 檢查日期是 {precomputed_data.exam_day}。
        - 檢查前三天 ({precomputed_data.prev_3d}) 和前兩天 ({precomputed_data.prev_2d}) 應該要低渣飲食。
        - 檢查前一天 ({precomputed_data.prev_1d}) 應該要無渣流質飲食。
        [學員回答]: {conversation_context}
        
        學員是否給出了正確的飲食衛教？如果正確，只輸出 "1"。如果不正確，只輸出 "0"。
        """
        response = await generate_llm_response(prompt, SCORING_MODEL_NAME)
        return 1 if "1" in response.strip() else 0

# 在 main.py 中實例化
# scoring_service = ScoringService()