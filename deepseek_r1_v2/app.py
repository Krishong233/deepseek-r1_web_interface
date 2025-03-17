from flask import Flask, request, render_template, Response, stream_with_context
import openai
import os
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

# 環境變數設定
api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("請設置 DEEPSEEK_API_KEY 環境變數")

# 初始化 DeepSeek 客戶端
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1"
)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/chatroom", methods=["GET"])
def chatroom():
    return render_template("chatroom.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "messages" not in data:
            return Response("無效的請求數據", status=400)

        messages = data["messages"]
        if not messages or not any(msg["role"] == "user" for msg in messages):
            return Response("請輸入有效內容", status=400)

        logger.info(f"發送的訊息序列：{messages}")

        def generate():
            try:
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=messages,
                    stream=True
                )
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except Exception as e:
                yield f"錯誤：{str(e)}"

        return Response(stream_with_context(generate()), mimetype="text/plain")

    except Exception as e:
        return Response(f"處理請求時發生錯誤：{str(e)}", status=500)

@app.route("/reset", methods=["POST"])
def reset():
    return render_template("chatroom.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
