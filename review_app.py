# review_app.py
import streamlit as st
import sqlite3
import pandas as pd
import os
import json
from datetime import datetime

# --- 設定 ---
# 請確保這裡指向正確的資料庫 (human_test.db 或 chatlog.db)
DB_PATH = "human_test.db"
AUDIO_DIR = "audio"

st.set_page_config(layout="wide", page_title="衛教對話回顧與除錯系統")

# --- CSS 優化 (讓除錯訊息比較好看) ---
st.markdown(
    """
<style>
    .debug-vector {
        font-size: 0.85em; 
        color: #0d6efd; 
        background-color: #f0f7ff; 
        padding: 5px 10px; 
        border-radius: 5px; 
        border: 1px dashed #0d6efd;
        margin-top: 5px;
        margin-bottom: 5px;
    }
    .score-pass { color: green; font-weight: bold; }
    .score-fail { color: red; font-weight: bold; }
    .stTextArea textarea { font-family: monospace; font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)


# --- 資料庫函式 ---
def get_connection():
    # 使用 URI 模式開啟唯讀連接，避免意外鎖死資料庫
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def get_sessions():
    """取得所有 Session 清單"""
    conn = get_connection()
    try:
        query = """
        SELECT 
            s.session_id, s.username, s.agent_code, s.created_at, sc.total_score
        FROM sessionid_user s
        LEFT JOIN sessionid_score sc ON s.session_id = sc.session_id
        ORDER BY s.created_at DESC
        """
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"讀取 Session 列表失敗: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def get_extended_chat_data(session_id):
    """
    取得完整的對話資料，包含：
    1. ChatLog (含 debug_info -> 向量搜尋結果)
    2. ScoringPromptLog (含 Prompt 與 LLM 原始回應)
    """
    conn = get_connection()

    # 1. 撈取對話 (包含 debug_info)
    chat_query = """
    SELECT id, role, text, audio_filename, time, debug_info
    FROM chatlog 
    WHERE session_id = ? 
    ORDER BY time ASC
    """
    chat_df = pd.read_sql(chat_query, conn, params=(session_id,))

    # 2. 撈取 LLM 評分除錯紀錄 (關聯 chat_log_id)
    # 使用 try-except 防止資料庫 schema 還沒更新時報錯
    try:
        prompt_query = """
        SELECT chat_log_id, scoring_item_id, llm_response, final_score, prompt_text
        FROM scoring_prompt_log 
        WHERE session_id = ?
        """
        prompt_df = pd.read_sql(prompt_query, conn, params=(session_id,))
    except Exception as e:
        # 如果資料表不存在或欄位沒加，回傳空 DataFrame 避免報錯
        # st.warning(f"注意: 無法讀取詳細評分紀錄 (可能是資料庫結構舊): {e}")
        prompt_df = pd.DataFrame(
            columns=[
                "chat_log_id",
                "scoring_item_id",
                "llm_response",
                "final_score",
                "prompt_text",
            ]
        )

    conn.close()
    return chat_df, prompt_df


def get_detailed_scores(session_id):
    conn = get_connection()
    try:
        query = "SELECT scoring_item_id, score FROM answer_log WHERE session_id = ?"
        df = pd.read_sql(query, conn, params=(session_id,))
        return df
    finally:
        conn.close()


def get_category_scores(session_id):
    conn = get_connection()
    try:
        query = "SELECT * FROM sessionid_score WHERE session_id = ?"
        df = pd.read_sql(query, conn, params=(session_id,))
        return df.iloc[0].to_dict() if not df.empty else {}
    finally:
        conn.close()


def get_criteria_map():
    """讀取 JSON 設定檔以顯示中文項目名稱"""
    try:
        # 路徑可能需要根據實際執行位置調整
        path = "scenarios/colonoscopy_bowklean/scoring_criteria_v2.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {item["id"]: item for item in data}
    except:
        pass
    return {}


# --- 側邊欄 ---
st.sidebar.title("🗂️ Session 選擇")
sessions_df = get_sessions()

if not sessions_df.empty:
    sessions_df["display"] = sessions_df.apply(
        lambda x: f"{x['created_at'][5:16]} | {x['agent_code']} | 分數: {x['total_score']} ({x['username']})",
        axis=1,
    )

    idx = st.sidebar.selectbox(
        "選擇紀錄",
        range(len(sessions_df)),
        format_func=lambda i: sessions_df.iloc[i]["display"],
    )
    sel_session = sessions_df.iloc[idx]
    current_session_id = sel_session["session_id"]

    st.sidebar.divider()
    st.sidebar.info(f"**ID:** `{current_session_id}`")
    st.sidebar.info(f"**Agent:** `{sel_session['agent_code']}`")
else:
    st.warning("資料庫中無資料，請先執行測試。")
    st.stop()

# --- 主畫面 ---
st.title(f"💬 對話回顧與除錯: {sel_session['agent_code']}")

# 載入資料
chat_df, prompt_df = get_extended_chat_data(current_session_id)
criteria_map = get_criteria_map()

col_chat, col_score = st.columns([0.65, 0.35])

# === 左欄：對話與除錯 ===
with col_chat:
    st.subheader("對話內容")

    for _, row in chat_df.iterrows():
        role = row["role"]
        text = row["text"]
        audio_file = row["audio_filename"]
        cid = row["id"]  # 這是 chat_log.id
        debug_info_str = row["debug_info"]

        # 顯示對話氣泡
        with st.chat_message(role, avatar="🧑‍⚕️" if role == "user" else "👴"):
            st.markdown(f"**{text}**")

            # 1. 播放器 (僅 User 有音檔)
            if role == "user" and audio_file:
                audio_path = os.path.join(AUDIO_DIR, audio_file)
                if os.path.exists(audio_path):
                    st.audio(audio_path, format="audio/wav")
                # else:
                #     st.caption(f"⚠️ 音檔遺失")

            # === 除錯資訊顯示區 (僅 User 發言需要顯示) ===
            if role == "user":

                # 2. 顯示向量搜尋結果 (觸發了哪些評分項目)
                if debug_info_str:
                    try:
                        debug_json = json.loads(debug_info_str)
                        vector_found = debug_json.get("vector_found", [])
                        if vector_found:
                            # 將 ID 轉為中文名稱，方便閱讀
                            items_display = []
                            for vid in vector_found:
                                c_info = criteria_map.get(vid, {})
                                c_name = c_info.get("item", vid)
                                items_display.append(f"{c_name}")

                            items_str = "、".join(items_display)
                            st.markdown(
                                f"""<div class="debug-vector">🕷️ 向量搜尋命中 (觸發評分): <br><b>{items_str}</b></div>""",
                                unsafe_allow_html=True,
                            )
                    except:
                        pass

                # 3. 顯示 LLM 評分詳情 (Prompt 與 Response)
                if not prompt_df.empty:
                    # 篩選出這句話 (cid) 所觸發的評分紀錄
                    my_logs = prompt_df[prompt_df["chat_log_id"] == cid]

                    if not my_logs.empty:
                        with st.expander(
                            f"🤖 LLM 評分細節 ({len(my_logs)} 項)", expanded=False
                        ):
                            for _, log in my_logs.iterrows():
                                item_id = log["scoring_item_id"]
                                score = log["final_score"]
                                raw_resp = log["llm_response"]
                                prompt_text = log["prompt_text"]  # 這是你要的 Context

                                # 取得中文名稱
                                c_info = criteria_map.get(item_id, {})
                                c_name = c_info.get("item", item_id)

                                # 狀態圖示
                                status_html = (
                                    f'<span class="score-pass">✅ PASS (1分)</span>'
                                    if score == 1
                                    else f'<span class="score-fail">❌ FAIL (0分)</span>'
                                )

                                st.markdown(
                                    f"#### 評分項目: {c_name} (`{item_id}`) - {status_html}",
                                    unsafe_allow_html=True,
                                )

                                # A. 顯示送給 LLM 的 Prompt (包含對話 Context)
                                st.markdown("**📤 送給 LLM 的 Prompt (包含 Context):**")
                                st.text_area(
                                    label="Prompt Content",
                                    value=prompt_text,
                                    height=200,
                                    key=f"prompt_{cid}_{item_id}",
                                    label_visibility="collapsed",
                                    disabled=True,
                                )

                                # B. 顯示 LLM 的原始回應
                                st.markdown("**📥 LLM 原始回應:**")
                                st.text_area(
                                    label="LLM Response",
                                    value=raw_resp,
                                    height=60,
                                    key=f"resp_{cid}_{item_id}",
                                    label_visibility="collapsed",
                                    disabled=True,
                                )
                                st.markdown("---")

# === 右欄：總分表 ===
with col_score:
    st.subheader("📊 最終得分")
    scores = get_category_scores(current_session_id)

    if scores:
        # 使用 str() 轉換確保顯示，避免 None
        total = scores.get("total_score", 0)
        st.metric("🏆 總分", total)

        categories_map = {
            "檢閱藥歷": "review_med_history_score",
            "醫療面談": "medical_interview_score",
            "諮商衛教": "counseling_edu_score",
            "人道專業": "humanitarian_score",
            "組織效率": "organization_efficiency_score",
            "臨床判斷": "clinical_judgment_score",
            "整體臨床": "overall_clinical_skills_score",
        }

        with st.expander("類別得分細項", expanded=True):
            for label, key in categories_map.items():
                val = scores.get(key, 0)
                st.write(f"**{label}:** {val}")

    st.markdown("---")
    st.write("**詳細項目清單 (僅列出資料庫紀錄)**")

    detail_df = get_detailed_scores(current_session_id)
    if not detail_df.empty:
        # 整理表格
        rows = []
        for _, row in detail_df.iterrows():
            item_id = row["scoring_item_id"]
            score = row["score"]
            info = criteria_map.get(item_id, {})

            rows.append(
                {
                    "類別": info.get("category", "其他"),
                    "項目": info.get("item", item_id),
                    "結果": "✅" if score > 0 else "❌",
                }
            )

        df_display = pd.DataFrame(rows)
        # 簡單的 DataFrame 顯示
        st.dataframe(
            df_display,
            hide_index=True,
            use_container_width=True,
            column_config={"結果": st.column_config.TextColumn("結果", width="small")},
        )
    else:
        st.info("尚無評分細節資料")
