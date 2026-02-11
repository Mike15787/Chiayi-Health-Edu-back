# scenarios/colonoscopy_bowklean/scoring_logic.py
import os
import json
import logging
import numpy as np
import faiss
import wave  # [新增] 用於讀取 wav 資訊
import math  # [新增] 用於無條件進位
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

logging.basicConfig(
    level=logging.INFO,
    # 關鍵是這一行：只保留 %(message)s，代表只顯示內容
    format='%(message)s', 
    
    # 注意：如果您的程式在其他地方已經設定過 logging，
    # 加上 force=True 可以強制覆蓋舊設定 (Python 3.8+ 適用)
    force=True 
)

CRITERIA_KEYWORDS = {
    # 禁水相關：一定要提到 水、喝、小時、禁
    "npo_mention": ["水", "禁", "喝", "小時", "不能", "渴"],
    "clinical_npo_timing": ["水", "禁", "喝", "點", "時"],
    # 飲食相關：一定要提到 吃、渣、纖維、菜、奶、肉、飲食
    "diet_basic": ["吃", "渣", "菜", "奶", "肉", "飲食", "纖維", "水果", "湯"],
    # 泡藥相關：一定要提到 泡、攪、溶解、cc、CC、杯、毫升
    "bowklean_mix_method": ["泡","攪","溶","cc","CC","ml","杯","水","混","喝"],
    # 藥丸相關：一定要提到 藥丸、錠、顆、樂可舒、Dulcolax、粉紅
    "dulcolax_method_and_time": ["藥丸","錠","顆","樂可舒","Dulcolax","粉紅","吃",],
    # 確認本人：一定要提到 本人、自己、名字、您
    "confirm_self_use": ["本人", "自己", "名字", "您", "誰"],
    # 檢查型態：一定要提到 無痛、一般、麻醉、睡、費用、錢、4500、800
    "proper_guidance_s2": ["無痛", "一般", "麻醉", "睡", "費", "錢"],
    "proper_guidance_s3": ["無痛", "一般", "麻醉", "睡", "費", "錢", "4500", "800"],
    # 檢查時間：一定要提到 幾點、什麼時候、日期、號、時間
    "proper_guidance_s4": ["點", "時", "號", "日", "哪天"],
    # 水分補充：一定要提到 水、cc、杯、喝
    "hydration_and_goal.s1": ["水", "cc", "CC", "杯", "喝", "2000", "兩千"],
    "hydration_and_goal.s2": ["水", "cc", "CC", "杯", "喝", "1000", "一千"],
    # 理想狀態：一定要提到 便、大號、廁所、顏色、黃、清、水
    "ideal_intestinal_condition": ["便", "廁所", "拉", "色", "黃", "清", "水", "渣"],
    # 防止單純講飲食日期時誤觸發
    "clinical_med_timing_1": ["藥", "包", "喝", "吃", "服用", "第一"],
    "clinical_med_timing_2": ["藥","包","喝","吃","服用","第二","早上","凌晨","上午"],
}

AUDIO_DIR = "audio"

logger = logging.getLogger(__name__)


class ColonoscopyBowkleanScoringLogic:
    def __init__(self):
        logger.info(f"初始化 {MODULE_ID}  的scoring_criteria")
        self.criteria = self._load_criteria(SCORING_CRITERIA_FILE)
        self.model = SentenceTransformer("BAAI/bge-m3")
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

        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        logger.info(f"Vector index for module {MODULE_ID} built successfully.")
        return index, criteria_id_list

    def find_relevant_criteria_ids(self, user_input: str, top_k: int = 5):
        """根據使用者輸入，找到最相關的評分項目ID"""
        # 1. 編碼 Query
        input_embedding = self.model.encode([user_input], convert_to_tensor=False)
        # 2. Query 也要正規化
        faiss.normalize_L2(input_embedding)
        # 3. 搜尋 (現在算出的是相似度分數，範圍 -1 到 1，越接近 1 越相似)
        distances, I = self.index.search(input_embedding, top_k)
        # 如果相似度低於 0.45，代表這句話跟評分標準沒什麼關係，就不要硬抓進來
        # 這能大幅減少 LLM 的誤判
        SIMILARITY_THRESHOLD = 0.45
        
        relevant_ids = set()

        # [新增] 用來收集詳細資訊的列表
        debug_details = []

        for j, idx in enumerate(I[0]):
            score = float(distances[0][j]) # 轉成 float 方便顯示
            if idx != -1 and score > SIMILARITY_THRESHOLD:
                relevant_ids.add(self.criteria_id_list[idx])
                # 記錄詳細資訊
                debug_details.append(f"[{score:.4f}] {self.criteria_id_list[idx]}")

        # --- [新增] 將搜尋結果印在 Terminal ---
        logger.info(f"\n🔍 [向量搜尋詳情] 輸入片段: {user_input[:30]}...")
        if debug_details:
            for detail in debug_details:
                logger.info(f"   {detail}")
        else:
            logger.info("   (無相關結果)")
        logger.info("-" * 40)

        # --- [修改 1] 強制觸發邏輯 ---
        # 如果 "說明保可淨使用方式" (s1) 被觸發，則強制加入 "臨床判斷-服藥時間" (1 & 2)
        # 因為這通常是在說明泡法的時候會一起講到時間，但向量可能只抓到泡法
        if "bowklean_mix_method" in relevant_ids:
            logger.info(
                f"[{MODULE_ID}] 搜尋到 'bowklean_mix_method', 強制檢查clinical_med_timing"
            )
            relevant_ids.add("clinical_med_timing_1")
            relevant_ids.add("clinical_med_timing_2")

        logger.info(
            f"找到可能的評分項目 {MODULE_ID}: {list(relevant_ids)}"
        )
        return list(relevant_ids)

    async def _call_llm_and_log(
        self,
        session_id: str,
        item_id: str,
        prompt: str,
        model_name: str,
        db: Session,
        chat_log_id: int = None,
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
                chat_log_id=chat_log_id,
            )
            db.add(log_entry)

            logger.info(
                f"評分項目 '{item_id}' 模組: {MODULE_ID} 分數: {score} 回應: '{raw_response[:50]}...'"
            )
            return score
        except Exception as e:
            logger.error(
                f"[{session_id}] Error during LLM call or logging for item '{item_id}' (module {MODULE_ID}): {e}"
            )
            return 0

    async def process_user_inputs_for_scoring(
        self,
        session_id: str,
        chat_snippet: List[Dict],
        db: Session,
        chat_log_id: int = None,
    ) -> List[str]:
        """處理多個使用者輸入並進行評分"""
        if not chat_snippet:
            return []

        # 定義格式化函式
        def format_snippet(snippet: List[Dict]) -> str:
            formatted_lines = []
            for item in snippet:
                role = "學員" if item["role"] == "user" else "病患"
                formatted_lines.append(f"{role}: {item['message']}")
            return "\n".join(formatted_lines)

        # 1. 準備給 LLM 評分的完整上下文 (保持完整，讓 LLM 看得懂前因後果)
        formatted_conversation_for_llm = format_snippet(chat_snippet)

        user_messages = [item['message'] for item in chat_snippet[-3:] if item['role']=='user']

        if not user_messages:
            logger.debug(f"[{session_id}] Skip search: No user messages in snippet.")
            return []

        # 將學員的訊息串接起來，不包含 "學員:" 標籤，純文字搜尋最準
        formatted_conversation_for_search = " ".join(user_messages)

        logger.info(
            f"[{session_id}] 向量搜尋 (最新三句): {formatted_conversation_for_search.replace(chr(10), ' | ')}"
        )

        # 3. 執行搜尋 (使用縮減後的文字搜尋，保持 top_k=5)
        relevant_ids = self.find_relevant_criteria_ids(
            formatted_conversation_for_search, top_k=5
        )

        if not relevant_ids:
            logger.info(
                f"[{session_id}] No relevant scoring criteria found for module {MODULE_ID}."
            )
            return []

        # ======================================================
        # [新增] 只有在測試環境下，才把搜尋結果寫回 ChatLog
        # ======================================================
        if chat_log_id and os.environ.get("APP_ENV") in ["auto", "human"]:
            try:
                # 撈出該句 chat_log
                chat_entry = db.query(ChatLog).filter(ChatLog.id == chat_log_id).first()
                if chat_entry:
                    # 將 list 轉成 JSON string 存入
                    debug_data = {"vector_found": relevant_ids}
                    chat_entry.debug_info = json.dumps(debug_data, ensure_ascii=False)
                    # 注意：這裡不急著 commit，後面 scoring 完會一起 commit，
                    # 或者你可以這裡先 db.add(chat_entry); db.commit()
                    db.add(chat_entry)
                    db.commit()
            except Exception as e:
                logger.error(f"寫入 Debug Info 失敗: {e}")

        # 4. 準備資料庫查詢已通過項目
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

        # 5. 逐一評分
        for item_id in relevant_ids:
            if item_id in passed_ids:
                # 這裡稍微降低 log 等級或略過 log，避免洗版
                continue

            criterion = next(
                (item for item in self.criteria if item["id"] == item_id), None
            )
            if criterion:
                # 注意：這裡傳給 LLM 的是「完整的上下文 (formatted_conversation_for_llm)」
                score = await self.score_item(
                    session_id,
                    formatted_conversation_for_llm,
                    criterion,
                    db,
                    precomputed_data,
                    chat_log_id=chat_log_id,
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
        chat_log_id: int = None,
    ):
        """對單個評分項目進行評分並儲存結果"""
        item_id = criterion["id"]
        item_type = criterion.get("type", "RAG")

        score = 0
        try:
            if item_type == "RAG":
                score = await self._check_rag_basic(
                    session_id, conversation_context, criterion, db, chat_log_id
                )
            elif item_type == "RAG_confirm_med_time":
                score = await self._check_confirm_med_time(
                    session_id, conversation_context, criterion, precomputed_data, db, chat_log_id
                )
            elif item_type == "RAG_confirm_npo":
                score = await self._check_npo_time(
                    session_id, conversation_context, precomputed_data, db, chat_log_id
                )
            elif item_type == "RAG_diet_logic":
                score = await self._check_diet_logic(
                    session_id, conversation_context, precomputed_data, db, chat_log_id
                )
            elif item_type == "RAG_med_usage_method":
                score = await self._check_med_usage_method(
                    session_id, conversation_context, criterion, db, chat_log_id
                )
            elif item_type == "RAG_special_meds":
                score = await self._check_special_meds(
                    session_id, conversation_context, db, chat_log_id
                )
            elif item_type == "non_term":
                score = await self._check_non_term(
                    session_id, conversation_context, criterion, db, chat_log_id
                )
            elif item_type == "RAG_emo_response":
                score = await self._check_emo_response(
                    session_id, conversation_context, criterion, db, chat_log_id
                )
            elif item_type == "RAG_review_med_history":
                score = await self._check_review_med_history(
                    session_id, conversation_context, criterion, db, chat_log_id
                )
            elif item_type == "first_med_mix_method":
                # 對應: 說明保可淨使用方式 (泡製)
                score = await self._check_med_mix_method(
                    session_id, conversation_context, criterion, db, chat_log_id
                )
            elif item_type == "dulcolax_method_and_time":
                # 對應: 說明樂可舒使用方式 (錠劑時間)
                score = await self._check_dulcolax_method_and_time(
                    session_id, conversation_context, criterion, db, chat_log_id
                )
            elif item_type == "hydration_method":
                # [修正] 區分 s1 (2000cc) 與 s2 (1000cc)
                score = await self._check_hydration_method(
                    session_id, conversation_context, criterion, db, chat_log_id
                )
            elif item_type == "ideal_intestinal":
                score = await self._check_ideal_intestinal(
                    session_id, conversation_context, criterion, db, chat_log_id
                )
            elif item_type == "RAG_explain_npo":
                # 這是單純檢查「有無提到禁水」，不同於 RAG_confirm_npo (檢查時間點準確性)
                score = await self._check_explain_npo_mention(
                    session_id, conversation_context, criterion, db, chat_log_id
                )
            elif item_type == "RAG_guidance3_check_type":
                score = await self._check_proper_guidance_s3(
                    session_id, conversation_context, criterion, db, chat_log_id
                )
            elif item_type == "bowklean_onset_time":
                score = await self._check_bowklean_onset_time(
                    session_id, conversation_context, criterion, db, chat_log_id
                )
            else:  # 對於未定義的類型，使用基礎 RAG 邏輯
                score = await self._check_rag_basic(
                    session_id, conversation_context, criterion, db, chat_log_id
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
        self,
        session_id: str,
        conversation_context: str,
        criterion: dict,
        db: Session,
        chat_log_id: int = None,
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
            session_id,
            criterion["id"],
            prompt,
            SCORING_MODEL_NAME,
            db,
            chat_log_id=chat_log_id,
        )


    # 12/04檢查 OK
    async def _check_confirm_med_time(
        self,
        session_id: str,
        conversation_context: str,
        criterion: dict,
        precomputed_data: PrecomputedSessionAnswer,
        db: Session,
        chat_log_id: int = None
    ) -> int:
        """檢查學員是否正確說明了第一包和第二包清腸劑的服用時間。"""
        if not precomputed_data:
            logger.warning(
                f"[{session_id}] Precomputed data is not available for _check_confirm_med_time in module {MODULE_ID}."
            )
            return 0

        # --- 加入 Debug Log ---
        logger.info(
            f"[{session_id}] Checking Timing: {criterion['id']}. Truth: Prev1d={precomputed_data.prev_1d}, ExamDay={precomputed_data.exam_day}, 2ndDose={precomputed_data.second_dose_time}"
        )
        prompt = ""
        if criterion["id"] == "clinical_med_timing_1":
            prompt = f"""
            請根據[要求]中的條件判斷[對話紀錄中]學員的衛教內容是否正確。
            [要求]: 學員必須提到 病人應該在檢查前一天 ({precomputed_data.prev_1d}) 的下午5點服用第一包(藥/保可淨/清腸藥)。
            [對話紀錄]: {conversation_context}
            
            忽略無關藥物(清腸劑/保可淨)的日期說明。這裡只檢查「藥物(清腸劑/保可淨)」的服用時間
            學員是否有清楚且正確地告知病人第一包藥的服用日期和時間？如果正確，只輸出 "1"。如果不正確，只輸出 "0"。
            """
        elif criterion["id"] == "clinical_med_timing_2":
            prompt = f"""
            請根據[要求]中的條件判斷[對話紀錄中]學員的衛教內容是否正確
            [要求] 學員必須提到 病人應該在檢查當天 ({precomputed_data.exam_day}) 的 {precomputed_data.second_dose_time} 服用第二包藥。
            [對話紀錄]: {conversation_context}
            
            忽略無關藥物(清腸劑/保可淨)的日期說明。這裡只檢查「藥物(清腸劑/保可淨)」的服用時間
            學員是否有正確告知病人第二包藥的服用日期和時間？如果正確，只輸出 "1"。如果不正確，只輸出 "0"。
            """
        else:
            return 0

        return await self._call_llm_and_log(
            session_id,
            criterion["id"],
            prompt,
            SCORING_MODEL_NAME,
            db,
            chat_log_id=chat_log_id,
        )

    async def _check_npo_time(
        self,
        session_id: str,
        conversation_context: str,
        precomputed_data: PrecomputedSessionAnswer,
        db: Session,
        chat_log_id: int = None
    ) -> int:
        """檢查學員是否清楚說明禁水時間點，並連帶給予 npo_mention 分數。"""
        if not precomputed_data:
            logger.warning(
                f"[{session_id}] Precomputed data is not available for _check_npo_time in module {MODULE_ID}."
            )
            return 0

        # 根據實際檢查類型判斷禁水前應停止多久
        npo_hours_before = 3 if precomputed_data.actual_check_type == "無痛" else 2

        # 為了避免 LLM 被 "上午" vs "早上" 或 "7:00" vs "7點" 搞混，我們強調語意
        prompt = f"""
        你是一個資深護理師，請判斷學員的衛教內容是否正確。
        
        [標準答案]: 
        病人正確的禁水開始時間是 **{precomputed_data.npo_start_time}**。
        (這是根據檢查時間往前推算 {npo_hours_before} 小時)。

        [學員回答]: 
        {conversation_context}
        
        [評分任務]:
        學員是否有告知病人從 **{precomputed_data.npo_start_time}** (或意思相同的時間點) 開始禁水？
        
        [注意]:
        1. 格式不同是可以的 (例如 "早上7點" 等於 "上午07:00")。
        2. 如果學員說 "檢查前X小時"，且換算出來的時間正確，也算得分。
        3. 重點是「時間點」要講對，且要提到「不能喝水/禁水」。

        如果正確，只輸出 "1"。否則輸出 "0"。
        """
        score = await self._call_llm_and_log(
            session_id,
            "clinical_npo_timing",
            prompt,
            SCORING_MODEL_NAME,
            db,
            chat_log_id=chat_log_id,
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
        self,
        session_id: str,
        conversation_context: str,
        criterion: dict,
        db: Session,
        chat_log_id: int = None,
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
            session_id,
            criterion["id"],
            prompt,
            SCORING_MODEL_NAME,
            db,
            chat_log_id=chat_log_id,
        )

    async def _check_special_meds(
        self,
        session_id: str,
        conversation_context: str,
        db: Session,
        chat_log_id: int = None,
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
                        db, chat_log_id=chat_log_id
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
        db: Session, chat_log_id: int = None
    ) -> int:
        """檢查學員是否正確說明了飲食衛教。"""
        if not precomputed_data:
            logger.warning(
                f"[{session_id}] Precomputed data is not available for _check_diet_logic in module {MODULE_ID}."
            )
            return 0
        prompt = f"""
        你是一個資深護理師，請判斷學員的「飲食衛教」內容是否正確。

        [正確的飲食時程與內容標準]:
        1. **低渣飲食**：
           - 時間：檢查前三天 ({precomputed_data.prev_3d}) 及 前兩天 ({precomputed_data.prev_2d})。
           - 內容：減少糞便體積。
           - 禁止：**蔬菜、水果**、高纖維食物、**奶類/乳製品**(牛奶、起司)、堅果。
           - 可食：白飯、白麵條、去皮魚肉、豆腐、蛋。

        2. **無渣流質飲食 (清流質)**：
           - 時間：檢查前一天 ({precomputed_data.prev_1d})。
           - 內容：完全無固體，可透光的液體。
           - 禁止：**牛奶/豆漿** (不透明)、濃湯、固體食物。
           - 可食：運動飲料 (舒跑/寶礦力)、無渣菜湯/魚湯/肉湯、蜂蜜水、冬瓜茶、開水。

        [學員回答]: 
        {conversation_context}
        
        [評分標準]:
        請判斷學員是否正確說明了上述兩個階段的「時間點」以及「飲食原則」。
        
        - 必須區分「前三天/前兩天是低渣」與「前一天是無渣流質」。
        - 若學員提到「不能吃青菜水果」、「不能喝牛奶」，這是正確的低渣/無渣觀念，請給予肯定。
        - 只要邏輯正確（時間階段對，且飲食內容大致符合標準），即可給分。

        如果飲食觀念與時程正確，只輸出 "1"。如果不正確或未提及，只輸出 "0"。
        """
        return await self._call_llm_and_log(
            session_id,
            "diet_basic",
            prompt,
            SCORING_MODEL_NAME,
            db,
            chat_log_id=chat_log_id,
        )

    async def _check_med_mix_method(
        self,
        session_id: str,
        conversation_context: str,
        criterion: dict,
        db: Session,
        chat_log_id: int = None,
    ) -> int:
        """檢查學員是否正確說明了特定一包清腸劑的泡製方法。"""

        # [修正] 不再單純依賴 ID 判斷是第幾包，因為 s1 代表通用的泡法說明
        # 我們改為讓 LLM 判斷「是否有說明泡法（無論第一包或第二包）」

        prompt = f"""
        你是一位嚴謹的臨床藥師，請判斷學員是否正確地說明了清腸劑(保可淨)的泡製與服用方式。

        [正確的泡製方法參考]:
        將一包「保可淨」倒入裝有150c.c.常溫水的杯中，攪拌至完全溶解後立即喝完。

        [學員與病人的對話紀錄]:
        ---
        {conversation_context}
        ---

        [你的判斷任務]:
        請判斷學員在對話中，是否有針對藥劑(第一包或第二包皆可)，清楚說明了類似上述的泡製方法？
        你需要檢查的關鍵點包含：「150c.c.的水量」、「攪拌/溶解」、「立即喝完」。
        只要語意正確即可，不需逐字比對。

        - 如果學員有說明上述關鍵泡製步驟，請只輸出 "1"。
        - 如果說明不完整、不正確或完全沒有提到，請只輸出 "0"。

        [你的判斷 (只輸出 1 或 0)]:
        """
        return await self._call_llm_and_log(
            session_id,
            criterion["id"],
            prompt,
            SCORING_MODEL_NAME,
            db,
            chat_log_id=chat_log_id,
        )

    async def _check_dulcolax_method_and_time(
        self,
        session_id: str,
        conversation_context: str,
        criterion: dict,
        db: Session,
        chat_log_id: int = None,
    ) -> int:
        prompt = f"""
        你是一位藥師。請檢查學員是否有針對「口服瀉藥錠劑」(例如：樂可舒/Dulcolax/粉紅色藥丸) 進行衛教。
        [對話紀錄]: {conversation_context}
        學員是否有提到除了喝的保可淨之外，還有「藥丸/錠劑」要吃？
        並且是否有說明大概什麼時候吃 (例如：檢查前一天、中午、或搭配第一包)？
        有提到錠劑及其服用時機輸出 "1"，否則 "0"。
        """
        return await self._call_llm_and_log(
            session_id,
            criterion["id"],
            prompt,
            SCORING_MODEL_NAME,
            db,
            chat_log_id=chat_log_id,
        )

    # 12/3 從non_term開始修正 好的 從non_term跳過XD 我這邊沒有那個專業用語集
    async def _check_non_term(
        self,
        session_id: str,
        conversation_context: str,
        criterion: dict,
        db: Session,
        chat_log_id: int = None,
    ) -> int:
        """
        檢查學員是否避免使用專業術語 (non_term)。
        邏輯：如果學員使用了 exclude_term 中的詞彙，則為 0 分；若使用通俗語言解釋，則為 1 分。
        """
        exclude_terms = criterion.get("exclude_term", [])
        terms_str = "、".join(exclude_terms)

        prompt = f"""
        你是一個嚴格的醫學溝通評分員。你的任務是檢查學員(藥師/護理師)在對話中是否使用了病人聽不懂的「艱澀專業術語」。

        [對話紀錄]:
        ---
        {conversation_context}
        ---
        
        [評分標準 - 請仔細閱讀]:
        1. ✅ **允許使用**：常見的疾病名稱是允許的，例如：「高血壓」、「糖尿病」、「中風」、「胃潰瘍」、「抗凝血劑」。這些在台灣是普及的用語，**不**算專業術語。
        2. ❌ **禁止使用**：艱澀的生理學或藥理學名詞，例如：「腸道蠕動」、「滲透壓」、「電解質不平衡」、「代謝性酸中毒」、「交互作用機制」。
        
        [你的判斷]:
        - 如果學員使用的詞彙都是病人聽得懂的（包含高血壓、糖尿病等），請輸出 "1"。
        - 如果學員使用了艱澀名詞且**沒有**立刻用白話文解釋（例如說了「增加滲透壓」卻沒解釋這是什麼），請輸出 "0"。

        [請只輸出 1 或 0]:
        """
        return await self._call_llm_and_log(
            session_id,
            criterion["id"],
            prompt,
            SCORING_MODEL_NAME,
            db,
            chat_log_id=chat_log_id,
        )

    # 1207 修正
    async def _check_emo_response(
        self,
        session_id: str,
        conversation_context: str,
        criterion: dict,
        db: Session,
        chat_log_id: int = None,
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
            session_id,
            criterion["id"],
            prompt,
            SCORING_MODEL_NAME,
            db,
            chat_log_id=chat_log_id,
        )

    # 1207 修正
    async def _check_review_med_history(
        self,
        session_id: str,
        conversation_context: str,
        criterion: dict,
        db: Session,
        chat_log_id: int = None,
    ) -> int:
        """檢查學員是否詢問過往用藥經驗。"""
        prompt = f"""
        請判斷學員是否詢問了病人關於「過去使用清腸藥物」的經驗。
        
        [對話紀錄]:
        {conversation_context}
        
        [評分標準]:
        學員是否有問類似以下的問題：
        - "您以前有做過大腸鏡嗎？"
        - "請問您之前有使用過清腸劑(藥)/保可淨嗎"
        - "您之前有用過這個藥嗎?"
        
        如果有詢問經驗，請輸出 "1"，否則輸出 "0"。
        """
        return await self._call_llm_and_log(
            session_id,
            criterion["id"],
            prompt,
            SCORING_MODEL_NAME,
            db,
            chat_log_id=chat_log_id,
        )

    # 1207 修正中
    async def _check_hydration_method(
        self,
        session_id: str,
        conversation_context: str,
        criterion: dict,
        db: Session,
        chat_log_id: int = None,
    ) -> int:
        item_id = criterion["id"]

        if item_id == "hydration_and_goal.s1":
            correct_volume = "2000c.c.以上 (或 8 杯 250cc)"
            correct_timing_concept = "2-3小時內陸續分次喝完"
        elif item_id == "hydration_and_goal.s2":
            correct_volume = "1000c.c. (或 4 杯 250cc)"
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
        
        注意事項
        1.學員是否提到了「{correct_volume}」以及「分次喝/慢慢喝」這兩個關鍵概念？
        2.學員在教泡藥(保可淨)時，會提到「用 150c.c. 的水泡開藥粉」。
          *請注意：這 150c.c. 絕對不算在補充水分內！*
          如果學員只說了「用150cc水泡藥」，但沒有提到後續要喝 {correct_volume}的水，請務輸出 "0"
        符合輸出 "1"，不符合或未提及輸出 "0" 不要輸出除了數字以外的文字
        """
        return await self._call_llm_and_log(
            session_id, item_id, prompt, SCORING_MODEL_NAME, db, chat_log_id=chat_log_id
        )

    async def _check_ideal_intestinal(
        self,
        session_id: str,
        conversation_context: str,
        criterion: dict,
        db: Session,
        chat_log_id: int = None,
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
            session_id,
            criterion["id"],
            prompt,
            SCORING_MODEL_NAME,
            db,
            chat_log_id=chat_log_id,
        )

    async def _check_proper_guidance_s3(
        self,
        session_id: str,
        conversation_context: str,
        criterion: dict,
        db: Session,
        chat_log_id: int = None,
    ) -> int:
        """
        [Type: RAG_guidance3_check_type]
        檢查是否進行引導對話。
        只要學員有試圖詢問或確認檢查類型 (直接問、問費用、問麻醉)，都算得分。
        """
        prompt = f"""
        你是一個資深護理師。評分項目是：「當病人不清楚或需要確認時，引導病人判斷是大腸鏡檢查是『一般』還是『無痛』」。

        [學員與病人的對話]:
        ---
        {conversation_context}
        ---

        [判斷標準]:
        請檢查學員是否說了類似以下任何一句話：
        1. "請問您是做一般的還是無痛的？" (直接詢問)
        2. "請問您繳費大約多少錢？是800還是4500？" (透過費用引導)
        3. "需要麻醉嗎？" (透過麻醉引導)

        只要學員有**試圖詢問或確認檢查類型**，不管是用問的、用猜的、還是用費用判斷，都請判定為合格。
        
        符合輸出 "1"，不符合輸出 "0"。
        """
        return await self._call_llm_and_log(
            session_id,
            criterion["id"],
            prompt,
            SCORING_MODEL_NAME,
            db,
            chat_log_id=chat_log_id,
        )

    async def _check_explain_npo_mention(
        self,
        session_id: str,
        conversation_context: str,
        criterion: dict,
        db: Session,
        chat_log_id: int = None,
    ) -> int:
        """
        單純檢查學員是否有提到「禁水」這個動作 (RAG_explain_npo)。
        [雙向綁定]: 如果這裡通過，clinical_npo_timing 也會自動給分。
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
        score = await self._call_llm_and_log(
            session_id,
            criterion["id"],
            prompt,
            SCORING_MODEL_NAME,
            db,
            chat_log_id=chat_log_id,
        )

        # [雙向綁定邏輯 B]：如果有提到禁水，連同「臨床判斷-禁水時間」一起給分
        if score == 1:
            logger.info(
                f"[{session_id}] npo_mention passed, auto-scoring clinical_npo_timing."
            )
            try:
                stmt = sqlite_insert(AnswerLog).values(
                    session_id=session_id,
                    module_id=MODULE_ID,
                    scoring_item_id="clinical_npo_timing",  # 連動項目
                    score=1,
                    created_at=datetime.now(),
                )
                on_conflict_stmt = stmt.on_conflict_do_update(
                    index_elements=["session_id", "scoring_item_id"], set_=dict(score=1)
                )
                db.execute(on_conflict_stmt)
            except Exception as e:
                logger.error(f"Failed to auto-score 'clinical_npo_timing': {e}")

        return score

    async def _check_satisfy_info_global(
        self,
        session_id: str,
        chat_logs: List[ChatLog],
        db: Session,
        chat_log_id: int = None,
    ) -> int:
        """
        [補考機制]
        在對話結束後，掃描學員整場的所有發言，檢查是否有詢問「有沒有問題」或「是否清楚」。
        """
        # 1. 撈出所有學員(User)講過的話
        user_messages = [log.text for log in chat_logs if log.role == "user"]

        if not user_messages:
            return 0

        # 2. 關鍵字快篩 (節省 LLM 資源)
        # 如果整場對話連這些詞都沒出現過，直接判 0 分
        keywords = ["問題", "清楚", "了解", "懂", "疑問", "ok", "OK", "可以嗎"]
        full_text_combined = " ".join(user_messages)

        if not any(k in full_text_combined for k in keywords):
            logger.info(
                f"[{session_id}] Global check for satisfy_info: No keywords found. Score 0."
            )
            return 0

        # 3. 組合 Prompt
        # 我們把學員的所有發言列出來，請 LLM 判斷
        formatted_messages = "\n".join([f"- {msg}" for msg in user_messages])

        prompt = f"""
        你是一個嚴格的護理溝通評分員。以下是學員在整場衛教過程中說過的所有話。
        請檢查學員是否曾經**主動詢問**病人「有沒有問題」、「是否聽得懂」或「哪裡不清楚」。

        [學員的所有發言]:
        {formatted_messages}

        [判斷標準]:
        只要學員有說出類似以下的句子 (語意相符即可)，請輸出 "1"：
        - "這樣有清楚嗎？"
        - "有沒有什麼問題想問？"
        - "剛剛講的可以嗎？"
        - "哪邊不清楚嗎？"

        如果完全沒有這類詢問，輸出 "0"。
        
        [你的判斷 (1 或 0)]:
        """

        # 呼叫 LLM (使用較強的模型以確保準確度)
        score = await self._call_llm_and_log(
            session_id,
            "satisfy_patient_infomation_global",
            prompt,
            SCORING_MODEL_NAME,
            db, chat_log_id=chat_log_id
        )

        if score == 1:
            logger.info(
                f"[{session_id}] Global check for satisfy_info PASSED! Updating record."
            )

            # 重要：補寫入資料庫，這樣 /details API 才會顯示通過
            try:
                stmt = sqlite_insert(AnswerLog).values(
                    session_id=session_id,
                    module_id=MODULE_ID,
                    scoring_item_id="satisfy_patient_infomation",  # 使用原始 ID
                    score=1,
                    created_at=datetime.now(),
                )
                on_conflict_stmt = stmt.on_conflict_do_update(
                    index_elements=["session_id", "scoring_item_id"], set_=dict(score=1)
                )
                db.execute(on_conflict_stmt)
                db.commit()
            except Exception as e:
                logger.error(f"Failed to update AnswerLog in global check: {e}")

        return score

    async def _check_bowklean_onset_time(
        self,
        session_id: str,
        conversation_context: str,
        criterion: dict,
        db: Session,
        chat_log_id: int = None,
    ) -> int:
        """
        [Type: bowklean_onset_time]
        檢查藥物作用時間。
        重點：
        1. 正確時間約為服藥後 2-3 小時 (或至少1小時以上)。
        2. 嚴格禁止說「馬上」、「立刻」、「吃完就拉」，這屬於錯誤醫療常識，必須給 0 分。
        """
        prompt = f"""
        你是一位資深臨床藥師。請判斷學員對於「保可淨(Bowklean)藥物作用時間」的衛教是否正確且安全。

        [正確的醫療知識]:
        保可淨藥物作用較溫和，服用後通常需要等待 **2到3小時** 才會開始出現腹瀉反應 (最快也要1小時)。

        [常見的錯誤觀念 (必須判為 0 分)]:
        ❌ 絕對不能說「吃完馬上」、「立刻」、「幾分鐘內」就會開始拉肚子。這是錯誤的資訊。

        [學員與病人的對話紀錄]:
        ---
        {conversation_context}
        ---

        [你的判斷任務]:
        1. 如果學員有提到 **「2-3小時」**、**「1-2小時」** 或 **「稍後/晚一點」** 才會開始作用 -> 請輸出 "1"。
        2. 如果學員說 **「馬上」**、**「立刻」**、**「吃下去就會拉」** -> 請務必輸出 "0" (因為這是錯誤衛教)。
        3. 如果完全沒提到時間 -> 請輸出 "0"。

        請只輸出 "1" 或 "0"。
        """
        return await self._call_llm_and_log(
            session_id,
            criterion["id"],
            prompt,
            SCORING_MODEL_NAME,
            db,
            chat_log_id=chat_log_id,
        )

    ''' 舊的 透過時間戳記來評分組織效率 不太穩定
    def _check_organization_sequence_by_time(
        self, session_id: str, has_dulcolax: bool, has_special_meds: bool, db: Session
    ) -> float:
        """
        [組織效率] 透過資料庫紀錄的時間戳記，檢查衛教順序。

        標準順序 (類別代號)：
        1. 飲食衛教 (diet_basic)
        2. 口服瀉藥錠劑 (dulcolax_method_and_time) -> 僅在 has_dulcolax=True 時檢查
        3. 清腸粉劑 (bowklean_mix_method)
        4. 禁水時間 (npo_mention)
        5. 其他用藥 (specify_special_meds) -> 僅在 has_special_meds=True 時檢查

        評分邏輯：
        - 4.0 分：所有「應說明項目」皆存在，且時間順序符合 1 -> 2 -> 3 -> 4 -> 5
        - 1.0 分：有缺漏項目，或順序顛倒
        """

        # 1. 定義要檢查的項目 ID
        target_items = ["diet_basic", "bowklean_mix_method", "npo_mention"]

        if has_dulcolax:
            target_items.append("dulcolax_method_and_time")

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
        t_powder = passed_times["bowklean_mix_method"]
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
            t_tablet = passed_times["dulcolax_method_and_time"]
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
        return 4.0'''

    async def _check_organization_sequence_by_llm(
        self,
        session_id: str,
        has_dulcolax: bool,
        has_special_meds: bool,
        db: Session,
        chat_log_id: int = None,
    ) -> float:
        """
        [組織效率] 改用 LLM 閱讀整場對話，判斷衛教順序是否符合邏輯。
        解決時間戳記在自動測試中過於接近導致誤判的問題。
        """

        # 1. 撈取整場對話紀錄 (按照時間排序)
        chat_logs = (
            db.query(ChatLog)
            .filter(ChatLog.session_id == session_id)
            .order_by(ChatLog.time.asc())
            .all()
        )

        if not chat_logs:
            return 0.0

        # 轉成純文字對話
        # 只取 User 講的話來做快篩即可，避免被 AI 的話干擾
        user_texts = [log.text for log in chat_logs if log.role == "user"]
        full_dialogue = "\n".join([f"{log.role}: {log.text}" for log in chat_logs])
        combined_user_text = " ".join(user_texts)

        # ====================================================
        # 第一道防線：Python 關鍵字快篩 (Hard Filter)
        # ====================================================
        # 定義此衛教模組的「必要關鍵字」，只要出現任一個，就視為「相關對話」
        # 如果使用者整場只講「聖誕快樂」、「戴安全帽」，一定不會命中這些詞
        relevant_keywords = [
            "檢查",
            "大腸鏡",
            "無痛",
            "一般",
            "喝水",
            "水分",
            "飲食",
            "低渣",
            "無渣",
            "吃藥",
            "服用",
            "藥粉",
            "保可淨",
            "樂可舒",
            "藥丸",
            "便秘",
            "拉肚子",
            "排便",
            "顏色",
            "清澈",
            "黃色",
            "禁水",
            "不能喝",
            "停藥",
            "高血壓",
            "糖尿病",
            "抗凝血",
            "阿斯匹靈",
            "除了",
            "問題",
            "清楚",
            "了解",
        ]

        # 檢查是否包含任一關鍵字
        is_relevant = any(k in combined_user_text for k in relevant_keywords)

        if not is_relevant:
            logger.info(
                f"[{session_id}] 組織效率(快篩): 未偵測到衛教關鍵字 -> 判定為無效對話 (0分)"
            )
            return 0.0

        # 2. 動態建構標準 SOP 順序
        steps = ["1. 飲食衛教 (低渣/無渣)"]

        if has_dulcolax:
            steps.append("2. 口服瀉藥錠劑 (樂可舒) [通常在檢查前一天中午]")
            steps.append("3. 清腸粉劑 (保可淨) [通常在檢查前一天傍晚/當天早上]")
        else:
            steps.append("2. 清腸粉劑 (保可淨) [通常在檢查前一天傍晚/當天早上]")

        steps.append("最後. 禁水時間 (NPO)")

        if has_special_meds:
            steps.append(
                "補充. 特殊用藥停藥/續用 (通常在講完禁水後補充，或穿插在藥物段落)"
            )

        sop_text = "\n".join(steps)

        # 3. 撰寫 Prompt
        prompt = f"""
        你是一個嚴格的衛教評分員。請閱讀下方對話，將學員的表現分類到下列三個類別之一。

        [類別定義 - 請依序判斷]:
        【類別 0】：無效或無關對話。
             - 對話內容完全沒有提到「大腸鏡、飲食、吃藥、禁水」等衛教主題。
             - 例如只說了：「你好」、「天氣不錯」、「記得戴安全帽」、「聖誕快樂」。
             - 只要內容與醫療衛教無關，一律選此項。

        【類別 1】：順序混亂或有缺漏。
             - 有提到衛教內容，但順序錯誤（例如：先講禁水才講飲食）。
             - 或者遺漏了重要步驟（例如：只講了吃藥，沒講飲食）。
        
        【類別 4】：順序正確且完整。
             - 依照 SOP 順序進行：飲食 -> 吃藥 -> 禁水。
             - 邏輯通順。

        [標準 SOP]:
        {sop_text}

        [對話紀錄]:
        {full_dialogue}

        [你的判斷]:
        請只輸出一個數字代號 ("0", "1", 或 "4")。
        """

        # 4. 呼叫 LLM
        # 使用 SCORING_MODEL_NAME (建議 gemma3:12b)
        response = await generate_llm_response(prompt, SCORING_MODEL_NAME)

        # 解析結果
        if "4" in response:
            logger.info(f"[{session_id}] 組織效率(LLM): 順序正確 (4分)")
            return 4.0
        elif "0" in response:
            logger.info(
                f"[{session_id}] 組織效率(LLM): 判定為無關對話 (0分). Response: {response}"
            )
            return 0.0
        else:
            # 預設落點為 1 分 (代表有講到一點相關的，但不好)
            logger.info(
                f"[{session_id}] 組織效率(LLM): 順序混亂 (1分). Response: {response}"
            )
            return 1.0

    # [新增] 輔助函式：計算單一音檔秒數
    def _get_audio_duration(self, filename: str) -> float:
        """讀取 wav 檔案並返回秒數，若讀取失敗則回傳 0"""
        if not filename:
            return 0.0

        file_path = os.path.join(AUDIO_DIR, filename)
        if not os.path.exists(file_path):
            logger.warning(f"Audio file not found: {file_path}")
            return 0.0

        try:
            with wave.open(file_path, "r") as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                return duration
        except Exception as e:
            logger.warning(f"Error reading audio duration for {filename}: {e}")
            return 0.0

    # [新增] 輔助函式：計算 Session 總衛教時間 (新公式)
    def _calculate_total_session_time(self, chat_logs: List[ChatLog]) -> float:
        """
        計算邏輯：
        1. 累加 user 和 patient(AI) 的音檔總長度。
        2. 加上 (音檔數量 / 1.25) 秒作為緩衝 (模擬思考與換氣時間)。
        3. 回傳單位為「分鐘」。
        """
        total_audio_seconds = 0.0
        audio_count = 0

        for log in chat_logs:
            if log.audio_filename:
                duration = self._get_audio_duration(log.audio_filename)
                total_audio_seconds += duration
                audio_count += 1

        # 公式：總音檔時間 + (總數量 / 1.25) 取整數
        buffer_seconds = int(audio_count / 1.25)
        final_total_seconds = total_audio_seconds + buffer_seconds

        # 轉為分鐘
        duration_minutes = final_total_seconds / 60.0

        logger.info(
            f"Time Calc: AudioSecs={total_audio_seconds:.2f}, Count={audio_count}, Buffer={buffer_seconds}, TotalMin={duration_minutes:.2f}"
        )
        return duration_minutes

    # --- 新增：最終分數計算邏輯 ---
    # scenarios/colonoscopy_bowklean/scoring_logic.py

    async def calculate_final_scores(
        self, session_id: str, db: Session
    ) -> Dict[str, str]:
        """
        計算該 Session 的最終分數。
        包含標準項目的累加以及特殊複合規則（如 S1-S4 藥物衛教）的計算。
        (已加入詳細 Debug Log 報表功能)
        """
        logger.info(
            f"[{session_id}] Calculating final scores using module logic: {MODULE_ID}"
        )

        # 用來儲存詳細評分報告的容器
        score_report = []
        score_report.append(f"\n{'='*20} [{session_id}] 評分詳細報表 {'='*20}")

        # 輔助函式：用來記錄每一項的得分狀況
        def record_item(category, item_name, passed_bool, score_got, description=""):
            status = "✅ PASS" if passed_bool else "❌ FAIL"
            score_report.append(
                f"[{category}] {item_name:<20} | {status} | 得分: {score_got:<4} | {description}"
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
        score_report.append(f"設定: Agent={agent_code} (Is A2? {is_case_A2})")

        # 判斷藥物組合 (E: 保可淨 Only, F: 保可淨 + 樂可舒)
        has_dulcolax = False
        if agent_settings:
            if agent_settings.drug_combination == "組合二":
                has_dulcolax = True
        score_report.append(
            f"設定: 藥物組合={'組合二(有樂可舒)' if has_dulcolax else '組合一(無樂可舒)'}"
        )

        # 判斷是否有特殊用藥
        has_special_meds = False
        if agent_settings and agent_settings.med_code:
            if (
                agent_settings.med_code.strip()
                and agent_settings.med_code.strip() != "無"
            ):
                has_special_meds = True
        score_report.append(
            f"設定: 特殊用藥={'有' if has_special_meds else '無'} ({agent_settings.med_code if agent_settings else ''})"
        )
        score_report.append("-" * 60)

        # 1. 獲取所有已得分項目 ID
        passed_items_query = (
            db.query(AnswerLog.scoring_item_id)
            .filter(AnswerLog.session_id == session_id, AnswerLog.score == 1)
            .all()
        )
        passed_item_ids = {item.scoring_item_id for item in passed_items_query}

        # 補考機制
        if "satisfy_patient_infomation" not in passed_item_ids:
            # 這裡為了不影響 log 邏輯，假設你已經在外部呼叫過補考，或是這裡再次呼叫
            # 為了簡化顯示，這裡只紀錄是否已通過
            pass

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

        # ==========================================
        # 1. 檢閱藥歷 (Review Med History) - 總分 9
        # ==========================================
        interaction_log = (
            db.query(SessionInteractionLog)
            .filter(SessionInteractionLog.session_id == session_id)
            .first()
        )

        ui_review_score = 0.0
        if interaction_log:
            if interaction_log.viewed_alltimes_ci:
                ui_review_score += 2.0
                record_item("1.檢閱藥歷", "歷次清腸", True, 2.0, "UI")
            else:
                record_item("1.檢閱藥歷", "歷次清腸", False, 0.0, "UI")

            if interaction_log.viewed_chiachi_med:
                ui_review_score += 3.0
                record_item("1.檢閱藥歷", "本院用藥", True, 3.0, "UI")
            else:
                record_item("1.檢閱藥歷", "本院用藥", False, 0.0, "UI")

            if interaction_log.viewed_med_allergy:
                ui_review_score += 1.0
                record_item("1.檢閱藥歷", "過敏史", True, 1.0, "UI")
            else:
                record_item("1.檢閱藥歷", "過敏史", False, 0.0, "UI")

            if interaction_log.viewed_disease_diag:
                ui_review_score += 1.0
                record_item("1.檢閱藥歷", "疾病診斷", True, 1.0, "UI")
            else:
                record_item("1.檢閱藥歷", "疾病診斷", False, 0.0, "UI")

            if interaction_log.viewed_cloud_med:
                ui_review_score += 2.0
                record_item("1.檢閱藥歷", "雲端藥歷", True, 2.0, "UI")
            else:
                record_item("1.檢閱藥歷", "雲端藥歷", False, 0.0, "UI")
        else:
            score_report.append("[1.檢閱藥歷] 無 UI 互動紀錄 (0分)")

        scores["review_med_history_score"] = ui_review_score
        score_report.append(f"   >>> [1.檢閱藥歷] 小計: {ui_review_score}")

        # ==========================================
        # 2. 醫療面談 (Medical Interview) - 總分 9
        # ==========================================

        # 2-1. 問好
        score_hello = 0.5 if is_case_A2 else 1.0
        p_hello = is_passed("greeting_hello")
        val_hello = p_hello * score_hello
        record_item("2.醫療面談", "問好", p_hello, val_hello)

        # 2-2. 請坐
        score_sit = 0.5 if is_case_A2 else 1.0
        p_sit = is_passed("invite_to_sit")
        val_sit = p_sit * score_sit
        record_item("2.醫療面談", "請坐", p_sit, val_sit)

        # 2-3. 適切發問
        pg_s1 = is_passed("proper_guidance_s1")
        pg_s2 = is_passed("proper_guidance_s2")
        pg_s3 = is_passed("proper_guidance_s3")
        pg_s4 = is_passed("proper_guidance_s4")
        pg_s5 = is_passed("proper_guidance_s5")

        pg_s2_s3_score = 1 if (pg_s2 + pg_s3) > 0 else 0
        proper_guidance_total = pg_s1 + pg_s4 + pg_s5 + pg_s2_s3_score

        record_item("2.醫療面談", "引導進入衛教", pg_s1, pg_s1 * 1.0)
        record_item(
            "2.醫療面談",
            "確認檢查類型",
            (pg_s2 + pg_s3) > 0,
            pg_s2_s3_score * 1.0,
            f"s2:{pg_s2}, s3:{pg_s3}",
        )
        record_item("2.醫療面談", "確認檢查時間", pg_s4, pg_s4 * 1.0)
        record_item("2.醫療面談", "確認額外用藥", pg_s5, pg_s5 * 1.0)

        # 2-4. 確認本人
        p_confirm = is_passed("confirm_self_use")
        val_confirm_self = p_confirm * 1.0
        record_item("2.醫療面談", "確認本人", p_confirm, val_confirm_self)

        # 5. 詢問經驗
        p_ask_exp = is_passed("review_med_history_1")
        val_ask_exp = p_ask_exp * 1.0
        record_item("2.醫療面談", "詢問過往經驗", p_ask_exp, val_ask_exp)

        # 7. 無專業術語
        p_no_term = is_passed("no_use_term")
        val_no_term = p_no_term * 1.0
        record_item("2.醫療面談", "無專業術語", p_no_term, val_no_term)

        # 8. 情緒回應 (A2才算分)
        p_emo = is_passed("emo_response")
        val_emo = (p_emo * 1.0) if is_case_A2 else 0.0
        record_item(
            "2.醫療面談",
            "情緒回應",
            p_emo,
            val_emo,
            "僅A2計分" if is_case_A2 else "非A2不計分",
        )

        scores["medical_interview_score"] = (
            val_hello
            + val_sit
            + proper_guidance_total
            + val_confirm_self
            + val_ask_exp
            + val_no_term
            + val_emo
        )
        score_report.append(
            f"   >>> [2.醫療面談] 小計: {scores['medical_interview_score']}"
        )

        # ==========================================
        # 3. 諮商衛教 (Counseling) - 總分 9
        # ==========================================

        # 3-1. 開立目的
        p_purp_s1 = is_passed("explain_med_purpose.s1")
        p_purp_s2 = is_passed("explain_med_purpose.s2")
        val_purpose = (p_purp_s1 or p_purp_s2) * 0.5
        record_item("3.諮商衛教", "開立目的", (p_purp_s1 or p_purp_s2), val_purpose)

        # 3-2. 註記時間
        pass_note_phrase = is_passed("note_have_med_time")
        pass_time1 = is_passed("clinical_med_timing_1")
        pass_time2 = is_passed("clinical_med_timing_2")
        pass_s2 = is_passed("dulcolax_method_and_time")

        logic_met = pass_time1 and pass_time2
        if has_dulcolax:
            logic_met = logic_met and pass_s2

        final_note_passed = pass_note_phrase or logic_met
        val_note_time = 0.5 if final_note_passed else 0.0

        detail_note = f"RAG:{pass_note_phrase} 或 Logic:{logic_met} (T1:{pass_time1}, T2:{pass_time2})"
        record_item(
            "3.諮商衛教", "協助註記時間", final_note_passed, val_note_time, detail_note
        )

        # 3-3. 藥物使用時機及方式
        val_med_method = 0.0
        if has_dulcolax:
            p_m_s1 = is_passed("bowklean_mix_method")
            p_m_s2 = is_passed("dulcolax_method_and_time")
            val_med_method = (p_m_s1 * 1.0) + (p_m_s2 * 1.0)
            record_item(
                "3.諮商衛教", "藥物方法(保可淨)", p_m_s1, p_m_s1 * 0.5, "組合F: 佔0.5分"
            )
            record_item(
                "3.諮商衛教", "藥物方法(樂可舒)", p_m_s2, p_m_s2 * 0.5, "組合F: 佔0.5分"
            )
        else:
            p_m_s1 = is_passed("bowklean_mix_method")
            val_med_method = p_m_s1
            record_item(
                "3.諮商衛教", "藥物方法(保可淨)", p_m_s1, val_med_method, "組合E: 佔2分"
            )

        pass_time1 = is_passed("clinical_med_timing_1")  # 第一包時間
        pass_time2 = is_passed("clinical_med_timing_2")  # 第二包時間
        val_med_method = val_med_method + pass_time1*0.5 + pass_time2*0.5

        # 3-4. 水分補充
        p_hydro = is_passed("hydration_and_goal.s1") or is_passed(
            "hydration_and_goal.s2"
        )
        val_hydro = p_hydro * 1.0
        record_item("3.諮商衛教", "水分補充", p_hydro, val_hydro)

        # 3-5. 清腸理想狀態
        p_ideal = is_passed("ideal_intestinal_condition")
        val_ideal = p_ideal * 1.0
        record_item("3.諮商衛教", "理想糞便狀態", p_ideal, val_ideal)

        # 3-6. 作用時間
        p_onset = is_passed("med_onset_duration")
        val_onset = p_onset * 0.5
        record_item("3.諮商衛教", "作用時間", p_onset, val_onset)

        # 3-7. 無痛確認 (共用變數)
        val_pain_check = pg_s2_s3_score
        record_item(
            "3.諮商衛教",
            "無痛/檢查確認",
            val_pain_check > 0,
            val_pain_check * 1.0,
            "同醫療面談S2/S3",
        )

        # 3-8. 禁水時間
        p_npo = is_passed("npo_mention")
        val_npo_explain = p_npo * 1.0
        record_item("3.諮商衛教", "禁水時間說明", p_npo, val_npo_explain)

        # 3-9. 特殊用藥
        p_special = is_passed("specify_special_meds")
        val_special_med = p_special * 1.0
        record_item("3.諮商衛教", "特殊用藥說明", p_special, val_special_med)

        # 3-10. 簡易飲食
        p_diet = is_passed("diet_basic")
        val_diet = p_diet * 0.5
        record_item("3.諮商衛教", "低渣飲食衛教", p_diet, val_diet)

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
        score_report.append(
            f"   >>> [3.諮商衛教] 小計: {scores['counseling_edu_score']}"
        )

        # ==========================================
        # 4. 人道專業 (Humanitarian) - 總分 9
        # ==========================================

        # 4-1. 表現尊重
        score_respect = val_hello + val_sit + pg_s1
        record_item(
            "4.人道專業",
            "表現尊重",
            score_respect > 0,
            score_respect,
            "邏輯: 問好+請坐+引導",
        )

        # 4-2. 需求滿足
        score_satisfy_weight = 4.0 if is_case_A2 else 3.0
        p_satisfy = is_passed("satisfy_patient_infomation")
        val_satisfy = p_satisfy * score_satisfy_weight
        record_item("4.人道專業", "需求滿足(確認理解)", p_satisfy, val_satisfy)

        # 4-3. 同理心
        score_empathy = val_no_term + val_ask_exp
        record_item(
            "4.人道專業",
            "同理心",
            score_empathy > 0,
            score_empathy,
            "邏輯: 無術語+問經驗",
        )

        # 4-4. 信賴感
        p_trust = is_passed("great_relationship_trust")
        val_trust = p_trust * 1.0
        record_item("4.人道專業", "信賴感(結尾)", p_trust, val_trust)

        # 4-5. 舒適守密
        val_comfort = (p_emo * 1.0) if is_case_A2 else 0.0
        record_item("4.人道專業", "舒適守密", p_emo, val_comfort, "同情緒回應 (僅A2)")

        scores["humanitarian_score"] = (
            score_respect + val_satisfy + score_empathy + val_trust + val_comfort
        )
        score_report.append(f"   >>> [4.人道專業] 小計: {scores['humanitarian_score']}")

        # ==========================================
        # 5. 組織效率 (Organization Efficiency) - 總分 9
        # ==========================================

        # 5-1. 優先順序
        val_sequence = await self._check_organization_sequence_by_llm(
            session_id, has_dulcolax, has_special_meds, db
        )
        record_item(
            "5.組織效率",
            "優先順序(LLM)",
            val_sequence == 4.0,
            val_sequence,
            "LLM判斷邏輯順序",
        )

        # 5-2. 及時且適時
        val_time = 0.0
        duration_minutes = 0.0

        if chat_logs:
            # 改用音檔計算邏輯
            duration_minutes = self._calculate_total_session_time(chat_logs)

            val_time = 0.5  # 只要有開口，至少給 0.5 分
            # 設定標準：5 到 9 分鐘為滿分 (此標準可依需求調整)
            if 5.0 <= duration_minutes <= 9.0:
                val_time = 1.5

        record_item(
            "5.組織效率",
            "時間控制",
            val_time == 1.5,
            val_time,
            f"耗時: {duration_minutes:.2f}分",
        )

        # 5-3. 歷練而簡潔
        mi_3_sub = val_confirm_self + proper_guidance_total
        rmh_3_sub = 0.0
        if interaction_log:
            if interaction_log.viewed_alltimes_ci:
                rmh_3_sub += 2.0
            if interaction_log.viewed_chiachi_med:
                rmh_3_sub += 3.0
            if interaction_log.viewed_cloud_med:
                rmh_3_sub += 2.0

        concise_final = (
            (scores["counseling_edu_score"] / 6.0)
            + (mi_3_sub / 5.0)
            + (rmh_3_sub / 7.0)
        )

        record_item(
            "5.組織效率", "歷練而簡潔", True, round(concise_final, 2), "公式計算"
        )

        scores["organization_efficiency_score"] = (
            val_sequence + val_time + concise_final
        )
        score_report.append(
            f"   >>> [5.組織效率] 小計: {scores['organization_efficiency_score']}"
        )

        # ==========================================
        # 6. 臨床判斷 (Clinical Judgment) - 總分 9
        # ==========================================

        p_id_spec = is_passed("identify_special_meds")
        val_has_special = p_id_spec * 2.0
        record_item("6.臨床判斷", "辨識特殊藥物", p_id_spec, val_has_special)

        val_judge_time1 = pass_time1 * 1.0
        record_item("6.臨床判斷", "判斷服藥時間1", pass_time1, val_judge_time1)

        val_judge_time2 = pass_time2 * 1.0
        record_item("6.臨床判斷", "判斷服藥時間2", pass_time2, val_judge_time2)

        p_clin_npo = is_passed("clinical_npo_timing")
        val_judge_npo = p_clin_npo * 1.0
        record_item("6.臨床判斷", "判斷禁水時間", p_clin_npo, val_judge_npo)

        val_judge_stop = p_special * 2.0
        record_item(
            "6.臨床判斷", "判斷停藥邏輯", p_special, val_judge_stop, "同諮商衛教-特殊藥"
        )

        # 判斷理解程度
        if score_satisfy_weight > 0:
            val_judge_understand = val_satisfy / score_satisfy_weight * 1.0
        else:
            val_judge_understand = 0.0
        record_item(
            "6.臨床判斷", "判斷理解程度", val_judge_understand > 0, val_judge_understand
        )

        # 判斷開立合理
        val_judge_reasonable = val_ask_exp
        record_item(
            "6.臨床判斷", "判斷開立合理", val_judge_reasonable > 0, val_judge_reasonable
        )

        scores["clinical_judgment_score"] = (
            val_has_special
            + val_judge_time1
            + val_judge_time2
            + val_judge_npo
            + val_judge_stop
            + val_judge_understand
            + val_judge_reasonable
        )
        score_report.append(
            f"   >>> [6.臨床判斷] 小計: {scores['clinical_judgment_score']}"
        )

        # ==========================================
        # 7. 整體臨床技能 (Overall) - 總分 9
        # ==========================================

        # 1. 態度
        val_attitude = (
            (
                (scores["medical_interview_score"] / 6.0)
                + (scores["humanitarian_score"] / 6.0)
            )
        )
        record_item("7.整體臨床", "態度(愛心同理)", True, round(val_attitude, 2))

        # 2. 整合能力
        mi_4_sub = val_confirm_self + proper_guidance_total + val_no_term
        val_integration = (
            (
                (scores["review_med_history_score"] / 6.0)
                + (mi_4_sub / 6.0)
                + (scores["clinical_judgment_score"] / 6.0)
            )
        )
        record_item("7.整體臨床", "整合能力", True, round(val_integration, 2))

        # 3. 整體有效性
        sum_ratios = (
            scores["review_med_history_score"] / 9.0
            + scores["medical_interview_score"] / 9.0
            + scores["counseling_edu_score"] / 9.0
            + scores["humanitarian_score"] / 9.0
            + scores["organization_efficiency_score"] / 9.0
            + scores["clinical_judgment_score"] / 9.0
        )
        val_effectiveness = sum_ratios / 3.0  
        # 滿分是 2 分
        record_item("7.整體臨床", "整體有效性", True, round(val_effectiveness, 2))

        scores["overall_clinical_skills_score"] = (
            val_attitude + val_integration + val_effectiveness
        )
        score_report.append(
            f"   >>> [7.整體臨床] 小計: {scores['overall_clinical_skills_score']}"
        )

        # --- 10. 計算總分並格式化 ---
        real_total = sum(scores.values())

        # 將所有數值轉為字串格式 (保留兩位小數)
        result = {key: str(round(value, 2)) for key, value in scores.items()}
        result["total_score"] = str(round(real_total, 2))

        score_report.append("-" * 60)
        score_report.append(f"🏆 總分: {result['total_score']}")
        score_report.append("=" * 60)

        # 一次性輸出完整報表
        logger.info("\n".join(score_report))

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

        # [新增] 需要手動處理顯示的項目，先在迴圈中跳過，稍後手動加入
        manual_display_items = [
            "bowklean_mix_method",  # 保可淨泡法 (合併用)
            "clinical_med_timing_1",  # 第一包時間 (合併用)
            "clinical_med_timing_2",  # 第二包時間 (合併用)
        ]

        for criterion in self.criteria:
            item_id = criterion["id"]
            category = criterion.get("category", "其他")

            # --- [修改重點 1] 過濾掉要合併的「適切發問」子項目 ---
            if item_id in proper_guidance_ids:
                continue

            if item_id in manual_display_items:
                continue  # 跳過手動處理項
            # --- [修改重點 1] 過濾「說明藥物開立目的」 ---
            # 邏輯：組合一 (has_dulcolax=False) -> 顯示 s1
            #       組合二 (has_dulcolax=True)  -> 顯示 s2
            if item_id == "explain_med_purpose.s1" and has_dulcolax:
                continue  # 如果是組合二，跳過 s1
            if item_id == "explain_med_purpose.s2" and not has_dulcolax:
                continue  # 如果是組合一，跳過 s2

            # --- [修改重點 2] 過濾「說明樂可舒使用方式」 ---
            # 邏輯：組合一沒有樂可舒，直接隱藏 s2
            if item_id == "dulcolax_method_and_time" and not has_dulcolax:
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

            if item_id == "bowklean_mix_method":
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
            elif item_id == "bowklean_mix_method":
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

        # --- 1. 組合「說明藥物使用時機及方式」 (取代原本的保可淨泡法) ---
        p_m_s1 = is_passed("bowklean_mix_method")  # 保可淨泡法 (1分)
        p_t1 = is_passed("clinical_med_timing_1")  # 第一包時間 (0.5分)
        p_t2 = is_passed("clinical_med_timing_2")  # 第二包時間 (0.5分)

        # 計算總分 (滿分2分)
        med_usage_total_score = (p_m_s1 * 1.0) + (p_t1 * 0.5) + (p_t2 * 0.5)

        # 組合描述文字
        med_usage_desc = (
            f"• 保可淨泡製方式(1分): {'✅' if p_m_s1 else '❌'}\n"
            f"• 第一包服用時間(0.5分): {'✅' if p_t1 else '❌'}\n"
            f"• 第二包服用時間(0.5分): {'✅' if p_t2 else '❌'}"
        )

        # 組合相關對話
        med_usage_dialogues = (
            dialogue_map.get("bowklean_mix_method", [])
            + dialogue_map.get("clinical_med_timing_1", [])
            + dialogue_map.get("clinical_med_timing_2", [])
        )

        grouped_details["諮商衛教"]["items"].append(
            {
                "item_id": "med_usage_combined",
                "item_name": "說明藥物使用時機及方式",  # 這是新標題
                "description": med_usage_desc,
                "weight": 2.0,
                "user_score": float(med_usage_total_score),
                "scoring_type": "Composite Logic",
                "relevant_dialogues": list(set(med_usage_dialogues)),
            }
        )

        # --- 2. 新增「確認是否進行無痛麻醉」 (補回缺失項目) ---
        # 邏輯同醫療面談的 s2/s3 (s23_score)，但在此處佔1分
        pain_check_dialogues = dialogue_map.get(
            "proper_guidance_s2", []
        ) + dialogue_map.get("proper_guidance_s3", [])
        grouped_details["諮商衛教"]["items"].append(
            {
                "item_id": "logic_pain_check",
                "item_name": "確認是否進行無痛麻醉",
                "description": "依據醫療面談中是否確認檢查型態(s2)或引導判斷(s3)",
                "weight": 1.0,
                "user_score": float(s23_score),  # 這是之前算好的變數
                "scoring_type": "Derived Logic",
                "relevant_dialogues": list(set(pain_check_dialogues)),
            }
        )

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
        w_hello = 0.5 if is_case_A2 else 1.0
        w_sit = 0.5 if is_case_A2 else 1.0
        w_pg1 = 1.0

        # 表現尊重
        s_hello = w_hello if is_passed("greeting_hello") else 0.0
        s_sit = w_sit if is_passed("invite_to_sit") else 0.0
        s_pg1 = w_pg1 if is_passed("proper_guidance_s1") else 0.0
        final_respect_score = s_hello + s_sit + s_pg1
        grouped_details["人道專業"]["items"].append(
            {
                "item_id": "logic_respect",
                "item_name": "表現尊重",
                "description": "同時達成：問好、請坐、引導衛教",
                "weight": w_hello + w_sit + w_pg1,
                "user_score": final_respect_score,
                "scoring_type": "Logic Formula",
                "relevant_dialogues": [],
            }
        )

        # 同理心
        s_no_term = 1.0 if is_passed("no_use_term") else 0.0
        s_ask_exp = 1.0 if is_passed("review_med_history_1") else 0.0
        final_empathy_score = s_no_term + s_ask_exp
        grouped_details["人道專業"]["items"].append(
            {
                "item_id": "logic_empathy",
                "item_name": "同理心(感同身受)",
                "description": "同時達成：無專業術語、詢問過往經驗",
                "weight": 2.0,
                "user_score": final_empathy_score,
                "scoring_type": "Logic Formula",
                "relevant_dialogues": [],
            }
        )

        # --- 3. 組織效率 ---
        # 優先順序
        val_sequence = await self._check_organization_sequence_by_llm(
            session_id, has_dulcolax, has_special_meds, db
        )
        grouped_details["組織效率"]["items"].append(
            {
                "item_id": "logic_sequence",
                "item_name": "按優先順序處置",
                "description": "飲食 -> 口服藥(若有) -> 清腸粉劑 -> 禁水 -> 其他用藥",
                "weight": 3.0,
                "user_score": val_sequence,
                "scoring_type": "LLM Logic",
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
        duration_minutes = 0.0
        if chat_logs_q:
            # 改用音檔計算邏輯
            duration_minutes = self._calculate_total_session_time(chat_logs_q)

        val_time = 1.5 if (5.0 <= duration_minutes <= 9.0) else 0.5

        grouped_details["組織效率"]["items"].append(
            {
                "item_id": "logic_time",
                "item_name": "及時且適時",
                "description": f"總衛教時間：{duration_minutes:.1f} 分鐘 (目標 5-9 分鐘)",
                "weight": 1.5,
                "user_score": val_time,
                "scoring_type": "Time Logic (Audio Based)",
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

        # [補回: 臨床判斷 - 判斷服藥時間 T1]
        grouped_details["臨床判斷"]["items"].append(
            {
                "item_id": "clinical_med_timing_1_judge",
                "item_name": "能依病人狀況判斷服藥時間(第一包)",
                "description": "臨床判斷維度計分",
                "weight": 1.0,
                "user_score": p_t1 * 1.0,  # 使用前面算好的 p_t1
                "scoring_type": "Standard",
                "relevant_dialogues": dialogue_map.get("clinical_med_timing_1", []),
            }
        )

        # [補回: 臨床判斷 - 判斷服藥時間 T2]
        grouped_details["臨床判斷"]["items"].append(
            {
                "item_id": "clinical_med_timing_2_judge",
                "item_name": "能依檢查時間判斷早上服藥時間點(第二包)",
                "description": "臨床判斷維度計分",
                "weight": 1.0,
                "user_score": p_t2 * 1.0,  # 使用前面算好的 p_t2
                "scoring_type": "Standard",
                "relevant_dialogues": dialogue_map.get("clinical_med_timing_2", []),
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
