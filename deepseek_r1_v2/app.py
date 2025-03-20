from flask import Flask, request, render_template, Response, stream_with_context, session, redirect, url_for, jsonify
import openai
import os
import logging
import json
from functools import wraps
import random
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

# Environment variable setup
api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("請設置 DEEPSEEK_API_KEY 環境變數")

# Initialize DeepSeek client
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1"
)

# Define questions and answers (moved from client-side)
questions = [
    {"q": "我的 classno 是什麼？", "a": "23"},
    {"q": "我的中文全名是什麼？", "a": "洪麒翔"},
    {"q": "我的生日是？（dd/mm）", "a": "07/08"}
]

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/get_question", methods=["GET"])
def get_question():
    if session.get("blocked"):
        return "您的 IP 已被封鎖", 403
    question = random.choice(questions)
    session["current_question"] = question["q"]
    session["correct_answer"] = question["a"]
    session["attempts"] = 0  # Reset attempts
    return question["q"]

@app.route("/validate_answer", methods=["POST"])
def validate_answer():
    ip = request.remote_addr
    attempts_key = f"attempts:{ip}"
    block_key = f"blocked:{ip}"
    if redis_client.get(block_key):
        return jsonify({"success": False, "blocked": True})
    data = request.get_json()
    answer = data.get("answer")
    correct_answer = session.get("correct_answer")
    if answer == correct_answer:
        session.permanent = True
        session["authenticated"] = True
        redis_client.delete(attempts_key)
        return jsonify({"success": True})
    else:
        attempts = int(redis_client.get(attempts_key) or 0) + 1
        redis_client.setex(attempts_key, 3600, attempts)  # 1 小時內有效
        if attempts >= 4:
            redis_client.setex(block_key, 86400, 1)  # 封鎖 24 小時
            return jsonify({"success": False, "blocked": True})
        return jsonify({"success": False, "attempts": attempts})

@app.route("/blocked_ips", methods=["GET"])
def get_blocked_ips():
    blocked_keys = redis_client.keys("blocked:*")  # 找所有 "blocked:IP" 鍵
    blocked_ips = [key.decode().split(":")[1] for key in blocked_keys]
    return jsonify({"blocked_ips": blocked_ips})

@app.route("/unblock_ip/<ip>", methods=["POST"])
def unblock_ip(ip):
    block_key = f"blocked:{ip}"
    attempts_key = f"attempts:{ip}"
    redis_client.delete(block_key)  # 刪除封鎖標記
    redis_client.delete(attempts_key)  # 刪除嘗試次數
    return jsonify({"success": True, "message": f"IP {ip} 已解鎖"})

@app.route("/unblock_all", methods=["POST"])
def unblock_all():
    blocked_keys = redis_client.keys("blocked:*")
    for key in blocked_keys:
        redis_client.delete(key)
    return jsonify({"success": True, "message": "所有封鎖的 IP 已解除"})


@app.route("/chatroom", methods=["GET"])
@login_required
def chatroom():
    return render_template("chatroom.html")

@app.route("/chat", methods=["POST"])
@login_required
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
@login_required
def reset():
    return render_template("chatroom.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
