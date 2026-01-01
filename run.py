# run.py
import os
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="啟動 AI Voice Chat API")

    # 新增 --test 參數
    parser.add_argument(
        "--test",
        action="store_true",
        help="啟動測試模式 (使用 chatlog_test.db)，保留資料供他人測試用",
    )

    # 新增 --host 和 --port 參數，方便你調整
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    # 設定環境變數
    if args.test:
        os.environ["APP_ENV"] = "test"
        print("==========================================")
        print("🚀 啟動模式: 給其他人測試用 (TEST MODE)")
        print("📂 資料庫: chatlog_test.db")
        print("==========================================")
    else:
        os.environ["APP_ENV"] = "dev"
        print("==========================================")
        print("🛠️  啟動模式: 本地開發用 (DEV MODE)")
        print("🗑️  資料庫: chatlog.db (可隨時刪除重置)")
        print("==========================================")

    # 啟動 Uvicorn
    # 注意：這裡使用 factory 模式或直接傳入字串讓 uvicorn 能夠吃到環境變數
    uvicorn.run(
        "main:app", host=args.host, port=args.port, reload=args.reload, log_level="info"
    )


if __name__ == "__main__":
    main()
