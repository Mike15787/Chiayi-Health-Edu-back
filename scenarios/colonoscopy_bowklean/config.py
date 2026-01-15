# HEALTHCARE_BACKEND/scenerios/colonoscopy_bowklean/config.py

MODULE_ID = "colonoscopy_bowklean"
SCORING_CRITERIA_FILE = (
    "scenarios/colonoscopy_bowklean/scoring_criteria_v2.json"  # 相對於專案根目錄
)
PATIENT_AGENT_MODEL_NAME = "gemma3:12b"  # 病患 AI 使用的模型
SCORING_MODEL_NAME = "gemma3:12b"  # 評分 LLM 使用的模型
STRONGER_SCORING_MODEL_NAME = "gemma3:12b"  # 強評分 LLM 使用的模型

# 特殊藥物衛教指令對照表
MED_INSTRUCTIONS = {
    "S1": "抗凝血劑請依醫師指示停藥；其他藥品，應於服用清腸劑前2小時或清腸後6小時服用。",  # 這裡 S1 代表抗凝血劑
    "S2": "抗血小板藥物請依醫師指示停藥；其他藥品，應於服用清腸劑前2小時或清腸後6小時服用。",  # S2 抗血小板
    "S3": "降血糖藥物請依醫師指示停藥；其他藥品，應於服用清腸劑前2小時或清腸後6小時服用。",  # S3 降血糖
    "S4": "緩解便秘藥物請勿停藥，應繼續服用；其他藥品應於服用清腸劑前2小時或清腸後6小時服用。",  # S4 緩解便秘
    "X1": "高血壓藥物請勿停藥，應繼續服用；其他藥品應於服用清腸劑前2小時或清腸後6小時服用。",  # X1 高血壓藥
    "X2": "抗癲癇藥物請勿停藥，應繼續服用；其他藥品應於服用清腸劑前2小時或清腸後6小時服用。",  # X2 抗癲癇藥
    "N": "無特殊用藥，所有藥品應於服用清腸劑前2小時或清腸後6小時服用。",  # N 無特殊用藥
}

# 將 JSON 中的 category 映射到 ScoresModel 的欄位 (用於總分計算)
CATEGORY_TO_FIELD_MAP = {
    "檢閱藥歷": "review_med_history_score",
    "醫療面談": "medical_interview_score",
    "諮商衛教": "counseling_edu_score",
    "人道專業": "humanitarian_score",
    "組織效率": "organization_efficiency_score",  # 新增
    "臨床判斷": "clinical_judgment_score",
    "整體臨床技能": "overall_clinical_skills_score",  # 新增
}

# 定義所有複合規則的子項目ID (用於總分計算時排除重複計分)
COMPOSITE_SUB_ITEM_IDS = {
    "proper_guidance_s1",
    "proper_guidance_s2",
    "proper_guidance_s3",
    "proper_guidance_s4",
    "proper_guidance_s5",
    "med_usage_timing_method.s1",
    "med_usage_timing_method.s2",
    "hydration_and_goal.s1",
    "hydration_and_goal.s2",
}
