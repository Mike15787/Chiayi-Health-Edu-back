# review_app.py
import streamlit as st
import sqlite3
import pandas as pd
import os
import json

# --- 設定 ---
DB_PATH = "human_test.db"
AUDIO_DIR = "audio"

st.set_page_config(layout="wide", page_title="衛教對話回顧與除錯系統")

# --- CSS 優化 ---
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
    /* 優化側邊欄顯示 */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- 資料庫函式 (加入快取與優化) ---

def get_connection():
    # 使用 URI 模式開啟唯讀連接
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

@st.cache_data(ttl=60)  # [優化1] 快取 Session 列表 60秒，避免頻繁讀庫
def get_sessions(limit=100):
    """取得 Session 清單 (預設限制最新的 100 筆)"""
    conn = get_connection()
    try:
        # [優化2] 加入 LIMIT，防止載入幾千筆導致側邊欄卡死
        query = f"""
        SELECT 
            s.session_id, s.username, s.agent_code, s.created_at, sc.total_score
        FROM sessionid_user s
        LEFT JOIN sessionid_score sc ON s.session_id = sc.session_id
        ORDER BY s.created_at DESC
        LIMIT {limit}
        """
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"讀取 Session 列表失敗: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_data(show_spinner=False) # [優化1] 對話內容通常不變，可以快取久一點
def get_extended_chat_data(session_id):
    conn = get_connection()

    # 1. 撈取對話
    chat_query = """
    SELECT id, role, text, audio_filename, time, debug_info
    FROM chatlog 
    WHERE session_id = ? 
    ORDER BY time ASC
    """
    chat_df = pd.read_sql(chat_query, conn, params=(session_id,))

    # 2. 撈取 LLM 評分除錯紀錄
    try:
        prompt_query = """
        SELECT chat_log_id, scoring_item_id, llm_response, final_score, prompt_text
        FROM scoring_prompt_log 
        WHERE session_id = ?
        """
        prompt_df = pd.read_sql(prompt_query, conn, params=(session_id,))
    except Exception:
        prompt_df = pd.DataFrame(
            columns=["chat_log_id", "scoring_item_id", "llm_response", "final_score", "prompt_text"]
        )

    conn.close()
    return chat_df, prompt_df

@st.cache_data
def get_detailed_scores(session_id):
    conn = get_connection()
    try:
        query = "SELECT scoring_item_id, score FROM answer_log WHERE session_id = ?"
        df = pd.read_sql(query, conn, params=(session_id,))
        return df
    finally:
        conn.close()

@st.cache_data
def get_category_scores(session_id):
    conn = get_connection()
    try:
        query = "SELECT * FROM sessionid_score WHERE session_id = ?"
        df = pd.read_sql(query, conn, params=(session_id,))
        return df.iloc[0].to_dict() if not df.empty else {}
    finally:
        conn.close()

@st.cache_data # JSON 檔幾乎不變，一定要快取
def get_criteria_map():
    try:
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

# [優化] 增加一個選項讓使用者決定要不要載入更多
load_all = st.sidebar.checkbox("載入全部歷史紀錄 (可能會慢)", value=False)
limit_num = 10000 if load_all else 100

sessions_df = get_sessions(limit=limit_num)

if not sessions_df.empty:
    sessions_df["display"] = sessions_df.apply(
        lambda x: f"{x['created_at'][5:16]} | {x['agent_code']} | 分: {x['total_score']} | {x['session_id'][:8]}...",
        axis=1,
    )

    idx = st.sidebar.selectbox(
        f"選擇紀錄 (顯示最近 {len(sessions_df)} 筆)",
        range(len(sessions_df)),
        format_func=lambda i: sessions_df.iloc[i]["display"],
    )
    sel_session = sessions_df.iloc[idx]
    current_session_id = sel_session["session_id"]

    st.sidebar.divider()
    st.sidebar.info(f"**ID:** `{current_session_id}`")
    st.sidebar.info(f"**Agent:** `{sel_session['agent_code']}`")
else:
    st.warning("資料庫中無資料")
    st.stop()

# --- 主畫面 ---
st.title(f"💬 對話回顧: {sel_session['agent_code']} ({sel_session['username']})")

# 載入資料
chat_df, prompt_df = get_extended_chat_data(current_session_id)
criteria_map = get_criteria_map()

# [優化3] 預先處理 prompt_df，將其轉為以 chat_log_id 為 Key 的字典
# 這樣在迴圈中就不用每次都 filter DataFrame，速度提升巨大
prompt_dict = {}
if not prompt_df.empty:
    # Group by chat_log_id
    grouped = prompt_df.groupby("chat_log_id")
    prompt_dict = {k: v for k, v in grouped}

col_chat, col_score = st.columns([0.65, 0.35])

# === 左欄：對話與除錯 ===
with col_chat:
    st.subheader("對話內容")

    # 使用 iterrows 雖然方便，但如果資料量大建議用 itertuples
    for row in chat_df.itertuples():
        role = row.role
        text = row.text
        audio_file = row.audio_filename
        cid = row.id
        debug_info_str = row.debug_info

        with st.chat_message(role, avatar="🧑‍⚕️" if role == "user" else "👴"):
            st.markdown(f"**{text}**")

            if role == "user" and audio_file:
                audio_path = os.path.join(AUDIO_DIR, audio_file)
                if os.path.exists(audio_path):
                    st.audio(audio_path, format="audio/wav")

            if role == "user":
                # 向量搜尋結果
                if debug_info_str:
                    try:
                        debug_json = json.loads(debug_info_str)
                        vector_found = debug_json.get("vector_found", [])
                        if vector_found:
                            items_display = []
                            for vid in vector_found:
                                c_info = criteria_map.get(vid, {})
                                c_name = c_info.get("item", vid)
                                items_display.append(f"{c_name}")
                            items_str = "、".join(items_display)
                            st.markdown(
                                f"""<div class="debug-vector">🕷️ 向量搜尋命中: <br><b>{items_str}</b></div>""",
                                unsafe_allow_html=True,
                            )
                    except:
                        pass

                # [優化3] 直接從字典查表，O(1) 複雜度
                if cid in prompt_dict:
                    my_logs = prompt_dict[cid]
                    
                    with st.expander(f"🤖 LLM 評分細節 ({len(my_logs)} 項)", expanded=False):
                        for log_row in my_logs.itertuples():
                            item_id = log_row.scoring_item_id
                            score = log_row.final_score
                            raw_resp = log_row.llm_response
                            prompt_text = log_row.prompt_text

                            c_info = criteria_map.get(item_id, {})
                            c_name = c_info.get("item", item_id)

                            status_html = (
                                f'<span class="score-pass">✅ PASS (1分)</span>'
                                if score == 1
                                else f'<span class="score-fail">❌ FAIL (0分)</span>'
                            )

                            st.markdown(
                                f"#### {c_name} (`{item_id}`) - {status_html}",
                                unsafe_allow_html=True,
                            )

                            st.text_area(
                                "Prompt",
                                value=prompt_text,
                                height=150,
                                key=f"p_{cid}_{item_id}",
                                disabled=True,
                            )
                            st.text_area(
                                "Response",
                                value=raw_resp,
                                height=60,
                                key=f"r_{cid}_{item_id}",
                                disabled=True,
                            )
                            st.divider()

# === 右欄：總分表 ===
with col_score:
    st.subheader("📊 最終得分")
    scores = get_category_scores(current_session_id)

    if scores:
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
    st.write("**詳細項目清單**")

    detail_df = get_detailed_scores(current_session_id)
    if not detail_df.empty:
        rows = []
        for row in detail_df.itertuples():
            item_id = row.scoring_item_id
            score = row.score
            info = criteria_map.get(item_id, {})

            rows.append(
                {
                    "類別": info.get("category", "其他"),
                    "項目": info.get("item", item_id),
                    "結果": "✅" if score > 0 else "❌",
                }
            )

        df_display = pd.DataFrame(rows)
        st.dataframe(
            df_display,
            hide_index=True,
            use_container_width=True,
            column_config={"結果": st.column_config.TextColumn("結果", width="small")},
        )
    else:
        st.info("尚無評分細節資料")