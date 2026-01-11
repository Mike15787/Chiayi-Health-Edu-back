# tests/auto_tester.py
# 這份測試程式內 有主要三種測試模式
# run_replay
# run_all_replays
# run_simulation
import sys
import os
import asyncio
import argparse
import logging
from datetime import datetime
import uuid
import re

# 設定環境變數為 test
os.environ["APP_ENV"] = "auto"

# 路徑設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 導入專案模組
from databases import (
    init_database,
    SessionLocal,
    ChatLog,
    AnswerLog,
    AgentSettings,
    SessionUserMap,
    PrecomputedSessionAnswer,
    Scores,
    Summary,
    SessionInteractionLog,
)
from module_manager import ModuleManager
from scoring_service_manager import ScoringServiceManager
from tests.simulated_user import SimulatedUserAgent
from tests.standard_script_generator import StandardScriptGenerator
from utils import generate_llm_response


# 設定 Log
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AutoTester")

# 初始化 Managers
module_manager = ModuleManager()
scoring_service_manager = ScoringServiceManager()


def get_next_replay_session_id(db, original_session_id: str) -> str:
    """
    產生新的 Session ID。
    邏輯：搜尋 db 中是否有 original_session_id_1, original_session_id_2...
    找出最大的數字並 +1。
    """
    # 1. 如果傳入的 ID 已經帶有 _數字，先還原成原始 ID
    # 例如：傳入 uuid_1，我們應該視為 base 是 uuid，然後找 uuid_2
    base_id = original_session_id
    match = re.match(r"^(.*)_(\d+)$", original_session_id)
    if match:
        base_id = match.group(1)

    # 2. 搜尋所有以此 base_id 開頭的 session
    pattern = f"{base_id}%"
    similar_sessions = (
        db.query(SessionUserMap.session_id)
        .filter(SessionUserMap.session_id.like(pattern))
        .all()
    )

    existing_ids = [s[0] for s in similar_sessions]

    max_suffix = 0
    for sid in existing_ids:
        if sid == base_id:
            continue
        # 解析後綴
        suffix_match = re.match(rf"^{re.escape(base_id)}_(\d+)$", sid)
        if suffix_match:
            num = int(suffix_match.group(1))
            if num > max_suffix:
                max_suffix = num

    new_id = f"{base_id}_{max_suffix + 1}"
    return new_id


def clone_session_data(db, source_session_id: str, new_session_id: str):
    """
    複製 Session 的必要資料以進行 Replay。
    包含：SessionUserMap, ChatLog, PrecomputedSessionAnswer, SessionInteractionLog
    """
    logger.info(f"正在複製資料: {source_session_id} -> {new_session_id}")

    # 1. 複製 SessionUserMap
    src_map = (
        db.query(SessionUserMap)
        .filter(SessionUserMap.session_id == source_session_id)
        .first()
    )
    if not src_map:
        raise ValueError("找不到原始 SessionUserMap")

    new_map = SessionUserMap(
        session_id=new_session_id,
        username=src_map.username,
        agent_code=src_map.agent_code,
        module_id=src_map.module_id,
        created_at=datetime.now(),  # Replay 的時間點
        is_completed=False,  # 重置完成狀態
        score=None,  # 重置分數
    )
    db.add(new_map)

    # 2. 複製 ChatLog (關鍵：保留 role, text, agent_code)
    # 我們不複製 id (自增) 和 time (可以用原始時間，也可以用現在時間，
    # 但為了計算組織效率的「時間控制」分數，我們必須保留「相對時間間隔」。
    # 最簡單的方法是直接拷貝原始的 time，雖然日期是舊的，但計算 duration (end - start) 仍會正確。
    src_logs = (
        db.query(ChatLog)
        .filter(ChatLog.session_id == source_session_id)
        .order_by(ChatLog.time)
        .all()
    )
    for log in src_logs:
        new_log = ChatLog(
            session_id=new_session_id,
            user_id=log.user_id,
            agent_code=log.agent_code,
            module_id=log.module_id,
            role=log.role,
            text=log.text,
            audio_filename=log.audio_filename,
            time=log.time,  # 保留原始時間戳記以維持時間評分準確性
        )
        db.add(new_log)

    # 3. 複製 PrecomputedSessionAnswer (關鍵：不能重新計算，必須用當時的標準答案)
    src_pre = (
        db.query(PrecomputedSessionAnswer)
        .filter(PrecomputedSessionAnswer.session_id == source_session_id)
        .first()
    )
    if src_pre:
        new_pre = PrecomputedSessionAnswer(
            session_id=new_session_id,
            module_id=src_pre.module_id,
            exam_day=src_pre.exam_day,
            prev_1d=src_pre.prev_1d,
            prev_2d=src_pre.prev_2d,
            prev_3d=src_pre.prev_3d,
            second_dose_time=src_pre.second_dose_time,
            npo_start_time=src_pre.npo_start_time,
            actual_check_type=src_pre.actual_check_type,
        )
        db.add(new_pre)

    # 4. 複製 SessionInteractionLog (UI 互動紀錄，這會影響檢閱藥歷分數)
    src_interact = (
        db.query(SessionInteractionLog)
        .filter(SessionInteractionLog.session_id == source_session_id)
        .first()
    )
    if src_interact:
        new_interact = SessionInteractionLog(
            session_id=new_session_id,
            module_id=src_interact.module_id,
            viewed_alltimes_ci=src_interact.viewed_alltimes_ci,
            viewed_chiachi_med=src_interact.viewed_chiachi_med,
            viewed_med_allergy=src_interact.viewed_med_allergy,
            viewed_disease_diag=src_interact.viewed_disease_diag,
            viewed_cloud_med=src_interact.viewed_cloud_med,
        )
        db.add(new_interact)

    db.commit()


async def run_replay(source_session_id: str):
    """
    回放模式：基於舊的對話，建立一個帶有後綴的新 Session，並重新執行評分。
    """
    logger.info(f"啟動回放測試，來源 Session ID: {source_session_id}")

    db = SessionLocal()
    try:
        # 1. 檢查來源是否存在
        src_map = (
            db.query(SessionUserMap)
            .filter(SessionUserMap.session_id == source_session_id)
            .first()
        )
        if not src_map:
            logger.error("找不到來源 Session ID")
            return

        # 2. 產生並建立新的 Session ID (複製資料)
        new_session_id = get_next_replay_session_id(db, source_session_id)
        clone_session_data(db, source_session_id, new_session_id)

        logger.info(
            f"已建立回放 Session: {new_session_id} (複製自 {source_session_id})"
        )
        module_id = src_map.module_id

        # 3. 讀取新 Session 的對話紀錄進行評分
        # 注意：我們現在是對 new_session_id 進行操作
        chat_logs = (
            db.query(ChatLog)
            .filter(ChatLog.session_id == new_session_id)
            .order_by(ChatLog.time)
            .all()
        )

        # 模擬逐句評分
        current_history = []
        for log in chat_logs:
            current_history.append({"role": log.role, "message": log.text})

            # 當遇到 User 發言時，嘗試觸發評分
            if log.role == "user":
                logger.info(f"正在重新評分 ({new_session_id}): {log.text[:20]}...")
                snippet_to_score = current_history[-5:]

                newly_passed = (
                    await scoring_service_manager.process_user_inputs_for_scoring(
                        new_session_id, module_id, snippet_to_score, db
                    )
                )
                if newly_passed:
                    logger.info(f"  -> 達成項目: {newly_passed}")

        # 4. 計算最終分數
        final_scores = await scoring_service_manager.calculate_final_scores(
            new_session_id, module_id, db
        )

        # 5. 更新 Scores 表 (calculate_final_scores 只是計算回傳 Dict，我們需要寫入 DB)
        # 這裡模擬 main.py 的 /finish 行為，將分數存入 DB
        new_score_record = Scores(
            session_id=new_session_id,
            module_id=module_id,
            **{
                k: v
                for k, v in final_scores.items()
                if k in Scores.__table__.columns.keys()
            },
        )
        db.merge(new_score_record)

        # 更新 Session 完成狀態
        new_session_map = (
            db.query(SessionUserMap)
            .filter(SessionUserMap.session_id == new_session_id)
            .first()
        )
        new_session_map.score = final_scores.get("total_score", "0")
        new_session_map.is_completed = True

        db.commit()

        print("\n" + "=" * 50)
        print(f"回放測試完成！")
        print(f"原始 Session: {source_session_id}")
        print(f"回放 Session: {new_session_id}")
        print("-" * 20)
        print("最終分數報告:")
        for k, v in final_scores.items():
            print(f"{k}: {v}")
        print("=" * 50)

    except Exception as e:
        logger.error(f"回放失敗: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


async def run_all_replays():
    """
    抓取資料庫中所有「原始」的 Session (排除 _1, _2 這種回放產生的)，
    並全部執行一次 replay。
    """
    logger.info("準備執行全量回放測試...")
    db = SessionLocal()
    try:
        # 1. 撈出所有的 Session ID
        all_sessions = db.query(SessionUserMap.session_id).all()
        all_ids = [s[0] for s in all_sessions]

        # 2. 過濾掉已經是 Clone 的 Session (避免無限增生)
        # 規則：如果 ID 結尾是 "_數字"，就視為 Clone，跳過
        original_ids = []
        for sid in all_ids:
            if not re.search(r"_\d+$", sid):
                original_ids.append(sid)

        logger.info(
            f"資料庫總筆數: {len(all_ids)}，原始 Session 筆數: {len(original_ids)}"
        )

        # 3. 逐一執行回放
        for idx, sid in enumerate(original_ids):
            print(f"\n[{idx+1}/{len(original_ids)}] 正在處理: {sid}")
            await run_replay(sid)

        print(f"\n全量測試結束！共執行了 {len(original_ids)} 筆回放。")

    except Exception as e:
        logger.error(f"全量回放發生錯誤: {e}")
    finally:
        db.close()


async def run_simulation(module_id: str, agent_code: str, max_turns: int = 15):
    """
    模擬模式：建立全新的 Session，由 SimulatedUserAgent 扮演藥師，與 Patient Agent 互動。
    用途：全自動化測試整個系統流程。
    """
    session_id = str(uuid.uuid4())
    logger.info(f"啟動模擬測試，新 Session ID: {session_id}, Agent: {agent_code}")

    db = SessionLocal()
    try:
        # 1. 建立 Session 資料 (模擬 /create_session)
        new_session = SessionUserMap(
            session_id=session_id,
            username="AutoTester",
            agent_code=agent_code,
            module_id=module_id,
            created_at=datetime.now(),
        )
        db.add(new_session)
        db.commit()

        # 2. 執行預計算 (Precomputation)
        precomputation_func = module_manager.get_precomputation_performer(module_id)
        if precomputation_func:
            await precomputation_func(session_id, agent_code, db)

        # 3. 初始化 AI 藥師
        agent_settings = (
            db.query(AgentSettings)
            .filter(AgentSettings.agent_code == agent_code)
            .first()
        )
        simulated_user = SimulatedUserAgent(agent_settings, module_id)

        # 取得 Agent 設定 (Dict)
        agent_settings_dict = {
            c.key: getattr(agent_settings, c.key)
            for c in agent_settings.__table__.columns
        }

        # 取得 Precomputed Data
        precomputed_obj = (
            db.query(PrecomputedSessionAnswer)
            .filter(PrecomputedSessionAnswer.session_id == session_id)
            .first()
        )
        precomputed_dict = (
            {
                c.key: getattr(precomputed_obj, c.key)
                for c in precomputed_obj.__table__.columns
            }
            if precomputed_obj
            else None
        )

        # 4. 開始對話迴圈
        history_text = ""  # 純文字歷史，給 Prompt 用
        chat_history_list = []  # List Dict 歷史，給 Scoring 用

        print("=" * 50)
        print(f"開始對話 (Session: {session_id})")
        print("=" * 50)

        for turn in range(max_turns):
            # --- A. 使用者 (AI 藥師) 發言 ---
            user_text = await simulated_user.generate_response(history_text)
            print(f"\n[Turn {turn+1}] 藥師(AI): {user_text}")

            # 存入 DB
            db.add(
                ChatLog(
                    session_id=session_id,
                    role="user",
                    text=user_text,
                    agent_code=agent_code,
                    module_id=module_id,
                )
            )
            db.commit()

            # 更新歷史
            history_text += f"藥師：{user_text}\n"
            chat_history_list.append({"role": "user", "message": user_text})

            # --- B. 系統評分 ---
            # 取最近 5 句給評分邏輯
            snippet = chat_history_list[-5:]
            passed_items = (
                await scoring_service_manager.process_user_inputs_for_scoring(
                    session_id, module_id, snippet, db
                )
            )
            if passed_items:
                print(f"   >>> [評分] 達成: {passed_items}")

            # --- C. 病患 (AI Agent) 回應 ---
            # 獲取 Patient Agent Builder
            patient_builder = module_manager.get_patient_agent_builder(module_id)
            # 構建 Prompt
            patient_prompt = patient_builder(
                user_text, history_text, agent_settings_dict, precomputed_dict
            )

            # 取得 Module Config 中的 Model Name
            mod_config = module_manager.get_module_config(module_id)
            model_name = getattr(mod_config, "PATIENT_AGENT_MODEL_NAME", "gemma3:4b")

            # 生成回應
            patient_response = await generate_llm_response(
                patient_prompt, model_name=model_name
            )

            # 處理回應 (去除 "你：" 等前綴)
            if patient_response.startswith("你：") or patient_response.startswith(
                "病患："
            ):
                patient_response = patient_response.split("：", 1)[1]

            print(f"[Turn {turn+1}] 病患(AI): {patient_response}")

            # 存入 DB
            db.add(
                ChatLog(
                    session_id=session_id,
                    role="patient",
                    text=patient_response,
                    agent_code=agent_code,
                    module_id=module_id,
                )
            )
            db.commit()

            # 更新歷史
            history_text += f"你(病患)：{patient_response}\n"
            chat_history_list.append({"role": "patient", "message": patient_response})

            # 簡單的結束判斷
            if "再見" in user_text or "結束" in user_text:
                print("\n對話結束。")
                break

        # 5. 最終算分
        print("-" * 50)
        final_scores = await scoring_service_manager.calculate_final_scores(
            session_id, module_id, db
        )
        print("模擬測試分數報告:")
        for k, v in final_scores.items():
            print(f"{k}: {v}")

    except Exception as e:
        logger.error(f"模擬測試失敗: {e}", exc_info=True)
    finally:
        db.close()


async def run_golden_path_test(target_agent: str = None, iterations: int = 2):
    """
    黃金路徑測試：
    針對所有 Agent (或指定 Agent)，使用 GoldenScriptGenerator 產生滿分劇本。
    與系統進行對話並評分，驗證系統能否在標準輸入下給出高分。
    """
    logger.info("啟動黃金路徑回歸測試 (Golden Path Test)...")

    db = SessionLocal()
    try:
        # 1. 取得要測試的 Agent 列表
        agents_query = db.query(AgentSettings)
        if target_agent:
            agents_query = agents_query.filter(AgentSettings.agent_code == target_agent)

        agents = agents_query.all()
        if not agents:
            logger.error("找不到任何 Agent 資料，請確認 agentset 是否已匯入。")
            return

        total_agents = len(agents)
        print(f"共需測試 {total_agents} 位 Agent，每位執行 {iterations} 次。")

        results = []

        for idx, agent in enumerate(agents):
            print(
                f"\n[{idx+1}/{total_agents}] 正在測試 Agent: {agent.agent_code} ({agent.med_complexity})"
            )

            for i in range(iterations):
                print(f"  > Run {i+1}...")
                session_id = str(uuid.uuid4())
                module_id = "colonoscopy_bowklean"  # 假設目前只測這個模組

                # A. 建立 Session
                new_session = SessionUserMap(
                    session_id=session_id,
                    username="GoldenTester",
                    agent_code=agent.agent_code,
                    module_id=module_id,
                    created_at=datetime.now(),
                )
                db.add(new_session)
                db.commit()

                # B. 執行預計算 (為了取得正確日期)
                precomputation_func = module_manager.get_precomputation_performer(
                    module_id
                )
                await precomputation_func(session_id, agent.agent_code, db)

                # C. 讀取預計算資料與設定
                precomputed_obj = (
                    db.query(PrecomputedSessionAnswer)
                    .filter(PrecomputedSessionAnswer.session_id == session_id)
                    .first()
                )

                # D. 生成標準範例流程
                generator = StandardScriptGenerator(agent, precomputed_obj)
                script_lines = generator.generate()

                # E. 執行對話迴圈
                history_text = ""
                chat_history_list = []

                # 模擬點擊 UI (檢閱藥歷) 以獲得分數
                ui_interaction = SessionInteractionLog(
                    session_id=session_id,
                    module_id=module_id,
                    viewed_alltimes_ci=True,
                    viewed_chiachi_med=True,
                    viewed_med_allergy=True,
                    viewed_disease_diag=True,
                    viewed_cloud_med=True,
                )
                db.add(ui_interaction)
                db.commit()

                for turn_idx, user_text in enumerate(script_lines):
                    # 1. 儲存使用者(藥師)發言
                    db.add(
                        ChatLog(
                            session_id=session_id,
                            role="user",
                            text=user_text,
                            agent_code=agent.agent_code,
                            module_id=module_id,
                        )
                    )
                    db.commit()

                    chat_history_list.append({"role": "user", "message": user_text})
                    history_text += f"藥師：{user_text}\n"

                    # 2. 觸發評分 (取最近5句)
                    snippet = chat_history_list[-5:]
                    await scoring_service_manager.process_user_inputs_for_scoring(
                        session_id, module_id, snippet, db
                    )

                    # 3. 產生病患回應 (為了維持對話流，但不影響使用者劇本)
                    # 這裡我們使用 Patient Agent，但因為是自動測試，我們不需要語音
                    agent_settings_dict = {
                        c.key: getattr(agent, c.key) for c in agent.__table__.columns
                    }
                    precomputed_dict = {
                        c.key: getattr(precomputed_obj, c.key)
                        for c in precomputed_obj.__table__.columns
                    }

                    patient_builder = module_manager.get_patient_agent_builder(
                        module_id
                    )
                    patient_prompt = patient_builder(
                        user_text, history_text, agent_settings_dict, precomputed_dict
                    )

                    # 為了速度，可以使用較快的模型，或者如果不在意回應內容，甚至可以 Mock 掉
                    # 這裡還是呼叫 LLM 讓 log 看起來真實
                    mod_config = module_manager.get_module_config(module_id)
                    model_name = getattr(
                        mod_config, "PATIENT_AGENT_MODEL_NAME", "gemma3:4b"
                    )

                    patient_response = await generate_llm_response(
                        patient_prompt, model_name=model_name
                    )
                    if patient_response.startswith(
                        "你："
                    ) or patient_response.startswith("病患："):
                        patient_response = patient_response.split("：", 1)[1]

                    db.add(
                        ChatLog(
                            session_id=session_id,
                            role="patient",
                            text=patient_response,
                            agent_code=agent.agent_code,
                            module_id=module_id,
                        )
                    )
                    db.commit()

                    chat_history_list.append(
                        {"role": "patient", "message": patient_response}
                    )
                    history_text += f"你(病患)：{patient_response}\n"

                # F. 結算分數
                final_scores = await scoring_service_manager.calculate_final_scores(
                    session_id, module_id, db
                )

                # G. 儲存與顯示結果
                total_score = float(final_scores.get("total_score", 0))
                # 更新到 Session
                session_update = (
                    db.query(SessionUserMap)
                    .filter(SessionUserMap.session_id == session_id)
                    .first()
                )
                session_update.score = str(total_score)
                session_update.is_completed = True

                # 寫入 Scores 表
                new_score_record = Scores(
                    session_id=session_id,
                    module_id=module_id,
                    **{
                        k: v
                        for k, v in final_scores.items()
                        if k in Scores.__table__.columns.keys()
                    },
                )
                db.merge(new_score_record)
                db.commit()

                print(f"    -> 分數: {total_score} / 63.0 (Pass: {total_score > 55})")
                results.append(
                    {
                        "agent": agent.agent_code,
                        "run": i + 1,
                        "score": total_score,
                        "details": final_scores,
                    }
                )

        # H. 輸出總報表
        print("\n" + "=" * 50)
        print("黃金路徑測試總結報告")
        print("=" * 50)
        print(f"{'Agent':<10} | {'Run':<5} | {'Score':<10} | {'Result'}")
        print("-" * 40)

        pass_count = 0
        for r in results:
            is_pass = "PASS" if r["score"] > 55 else "FAIL"  # 假設 55 分為通過門檻
            if is_pass == "PASS":
                pass_count += 1
            print(f"{r['agent']:<10} | {r['run']:<5} | {r['score']:<10} | {is_pass}")

            if r["score"] < 50:
                # 若分數過低，印出詳細扣分項目以便除錯
                print(f"    [Low Score Debug] {r['details']}")

        print("-" * 40)
        print(f"總執行次數: {len(results)}")
        print(f"通過次數: {pass_count}")
        print(f"通過率: {pass_count/len(results)*100:.1f}%")
        print("=" * 50)

    except Exception as e:
        logger.error(f"黃金路徑測試發生錯誤: {e}", exc_info=True)
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Healthcare System Auto Tester")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- 修改這裡 ---
    # Replay Command
    replay_parser = subparsers.add_parser("replay", help="Replay existing session(s)")
    # 讓 session_id 變成選填 (nargs='?')，但如果沒加 --all 就必須填
    replay_parser.add_argument("session_id", type=str, nargs="?", help="The specific Session ID to replay")
    # 新增 --all 參數
    replay_parser.add_argument("--all", action="store_true", help="Replay ALL original sessions in database")
    # --- [新增] 指定模型參數 ---
    replay_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override Scoring Model (e.g., gemma2:9b, gemini-1.5-flash)",
    )

    # Simulation Command
    sim_parser = subparsers.add_parser("sim", help="Simulate a new conversation")
    sim_parser.add_argument("--agent", type=str, default="A1", help="Agent Code (e.g., A1, B2)")
    sim_parser.add_argument("--module", type=str, default="colonoscopy_bowklean", help="Module ID")
    sim_parser.add_argument("--turns", type=int, default=15, help="Max turns")

    # --- 新增 Golden Command ---
    golden_parser = subparsers.add_parser("golden", help="Run Golden Path Regression Test")
    golden_parser.add_argument("--agent", type=str, help="Specific agent code to test (optional)")
    golden_parser.add_argument("--iter", type=int, default=2, help="Iterations per agent (default: 2)")

    args = parser.parse_args()

    # 確保資料庫已初始化 (Test DB)
    init_database()

    if args.command == "replay":
        if args.all:
            asyncio.run(run_all_replays())
        else:
            if not args.session_id:
                print("Error: session_id is required unless --all is specified.")
            else:
                asyncio.run(run_replay(args.session_id))
    elif args.command == "sim":
        asyncio.run(run_simulation(args.module, args.agent, args.turns))
    elif args.command == "golden":
        asyncio.run(run_golden_path_test(args.agent, args.iter))
    else:
        parser.print_help()
