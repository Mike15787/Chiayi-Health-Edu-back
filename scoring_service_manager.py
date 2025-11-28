# scoring_service_manager.py
import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session

from databases import AnswerLog, PrecomputedSessionAnswer, ScoringAttributionLog
from module_manager import ModuleManager # 引入新的 ModuleManager

logger = logging.getLogger(__name__)

#ScoringServiceManager是一個class 但他聚集了 
# get_scoring_logic 取得指定模組的評分instance 
# process_user_inputs_for_scoring 將評分任務委派給特定模組的評分邏輯。

class ScoringServiceManager:
    """
    管理所有教育模組的評分邏輯。
    它會根據 session_id 找到對應的模組，並將評分任務委派給該模組的 ScoringLogic。
    """
    def __init__(self):
        logger.info("Initializing Scoring Service Manager...")
        self.module_manager = ModuleManager()
        self.active_scoring_logics: Dict[str, Any] = {} # 存放已實例化的模組評分邏輯
        logger.info("Scoring Service Manager initialized.")

    def _get_scoring_logic(self, module_id: str):
        """獲取指定模組的評分邏輯實例，如果未實例化則創建"""
        if module_id not in self.active_scoring_logics:
            self.active_scoring_logics[module_id] = self.module_manager.get_scoring_logic_instance(module_id)
        return self.active_scoring_logics[module_id]

    async def process_user_inputs_for_scoring(self, session_id: str, module_id: str, chat_snippet: List[Dict], db: Session) -> List[str]:
        """
        將評分任務委派給特定模組的評分邏輯。
        """
        scoring_logic = self._get_scoring_logic(module_id)
        
        # 這將呼叫模組專屬的 process_user_inputs_for_scoring
        newly_passed_item_ids = await scoring_logic.process_user_inputs_for_scoring(
            session_id, chat_snippet, db
        )
        return newly_passed_item_ids
    
    async def calculate_final_scores(self, session_id: str, module_id: str, db: Session) -> Dict[str, str]:
        """
        將「最終分數計算任務」委派給特定模組。
        """
        scoring_logic = self._get_scoring_logic(module_id)

        # 檢查該模組是否有實作 calculate_final_scores
        if hasattr(scoring_logic, 'calculate_final_scores'):
            return await scoring_logic.calculate_final_scores(session_id, db)
        else:
            logger.error(f"Module {module_id} does not implement calculate_final_scores.")
            # 回傳預設零分，避免崩潰
            return {
                "total_score": "0.00",
                "review_med_history_score": "0.00",
                "medical_interview_score": "0.00",
                "counseling_edu_score": "0.00",
                "organization_efficiency_score": "0.00",
                "clinical_judgment_score": "0.00",
                "humanitarian_score": "0.00",
                "overall_clinical_skills_score": "0.00"
            }
    
    def get_scoring_criteria_map(self, module_id: str) -> Dict[str, Dict]:
        """獲取指定模組的評分標準映射"""
        return {item['id']: item for item in self._get_scoring_logic(module_id).criteria}

    def get_module_config(self, module_id: str) -> Any:
        """獲取指定模組的配置對象"""
        return self.module_manager.get_module_config(module_id)

    def get_summary_generator(self, module_id: str) -> Any:
        """獲取指定模組的總結生成器函數"""
        return self.module_manager.get_summary_generator(module_id)
    
    def get_org_efficiency_scorer(self, module_id: str) -> Any:
        """獲取指定模組的組織效率評分器函數"""
        return self.module_manager.get_org_efficiency_scorer(module_id)