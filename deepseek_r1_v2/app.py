from flask import Flask, request, render_template, Response, stream_with_context, session, redirect, url_for, jsonify
import openai
import os
import logging
import json
from functools import wraps
import random
import redis
import re
import time
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

# --------------------- New SQL Quiz Generator Endpoints ---------------------

def count_subqueries(sql):
    # Count occurrences of subquery patterns: a '(' followed by SELECT (case-insensitive)
    return len(re.findall(r'\(\s*SELECT', sql, re.IGNORECASE))

# Define fallback questions
sql_questions = [
    {"q": "Write a SQL query to select all employees from the employees table.", "a": "SELECT * FROM employees;"},
    {"q": "Write a SQL query to find the average salary of employees in the employees table.", "a": "SELECT AVG(salary) FROM employees;"},
    {"q": "Write a SQL query to count the number of customers registered after January 1, 2020.", "a": "SELECT COUNT(*) FROM customers WHERE registration_date > '2020-01-01';"}
]

# Serve the SQL quiz page
@app.route("/sql-test", methods=["GET"])
def sql_test():
    return render_template("sql-test.html")

# Generate SQL question
@app.route("/get_sql_question", methods=["GET"])
def get_sql_question():
    # 動態生成 prompt，加入隨機性
    random_seed = random.randint(1, 1000)  # 隨機數字
    timestamp = int(time.time())  # 當前時間戳
    prompt = f"""
Generate a SQL quiz question and its correct SQL answer based on the following table schemas.
The quiz question and answer must be in English.
Restrictions:
1. Only use the following SQL keywords, operators, and functions: FALSE, TRUE, AND, NOT, OR, ABS, AVG, INT, MAX, MIN, SUM, COUNT, ASC, AT, CHAR, CHR, CHAR_LENGTH, LEN, LOWER, TRIM, SPACE, SUBSTRING, SUBSTR, MID, UPPER, VALUE, VAL, DATE, DAY, MONTH, YEAR, ADD, ALL, ALTER, ANY, AS, BETWEEN, BY, CREATE, DELETE, DESC, DISTINCT, DROP, EXISTS, FROM, GROUP, HAVING, IN, INDEX, INNER, JOIN, INSERT, INTEGER, INTERSECT, INTO, LEFT, LIKE, MINUS, NULL, RIGHT, FULL, ON, ORDER, SELECT, SET, TABLE, TO, UNION, UNIQUE, UPDATE, VALUES, VIEW, WHERE, +, -, *, /, >, <, =, >=, <=, <>, %, _, '.
2. The SQL statement (correct answer) may contain at most one sub-query (no more than one in whole answer), 1/3 difficulty level.
3. Output your result as a JSON object with exactly two keys: "question" and "answer". Do not include any additional text.
4. To ensure variety, generate a unique question each time, using this random seed: {random_seed} and timestamp: {timestamp}.
Provided Table Schemas:
-- employees table
employee_id   INTEGER    -- Primary Key
name          CHAR(100)
salary        INTEGER
department_id INTEGER    -- Foreign Key

-- customers table
customer_id        INTEGER  -- Primary Key
name               CHAR(100)
registration_date  DATE

-- products table
product_id    INTEGER   -- Primary Key
product_name  CHAR(100)
price         INTEGER

-- orders table
order_id      INTEGER   -- Primary Key
customer_id   INTEGER   -- Foreign Key
total_amount  INTEGER

-- departments table
department_id    INTEGER   -- Primary Key
department_name  CHAR(100)
"""
    messages = [
        {"role": "system", "content": "You are a SQL quiz generator."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            presence_penalty=1,

            messages=messages,
            max_tokens=1000,
            temperature=2,  # 增加隨機性（範圍 0-2，默認可能是 0 或低值）
        )
        generated_text = response.choices[0].message.content.strip()
        
        logger.info(f"Raw API response: {generated_text}")
        
        json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
        else:
            raise ValueError("No JSON found in API response")
        # Try to parse the generated text as JSON
        result = json.loads(json_text)
        question_text = result.get("question", "")
        answer_text = result.get("answer", "")
        if not question_text or not answer_text:
            raise ValueError("Missing question or answer in API response")
    except Exception as e:
        logger.error(f"Error generating SQL quiz question: {e}")
        # Fallback to a predefined question
        chosen = random.choice(sql_questions)
        question_text = chosen["q"]
        answer_text = chosen["a"]

    # Save the expected answer (normalized) for validation.
    session["sql_correct_answer"] = answer_text.strip().lower()
    # Append the provided table schemas to the question for display.
    table_schema = """
    
Provided Table Schemas:
-- employees table
employee_id   INTEGER    -- Primary Key
name          CHAR(100)
salary        INTEGER
department_id INTEGER    -- Foreign Key

-- customers table
customer_id        INTEGER  -- Primary Key
name               CHAR(100)
registration_date  DATE

-- products table
product_id    INTEGER   -- Primary Key
product_name  CHAR(100)
price         INTEGER

-- orders table
order_id      INTEGER   -- Primary Key
customer_id   INTEGER   -- Foreign Key
total_amount  INTEGER

-- departments table
department_id    INTEGER   -- Primary Key
department_name  CHAR(100)
"""
    full_question = question_text
    response = jsonify({"question": full_question})
    response.headers["Cache-Control"] = "no-store"
    return response

@app.route("/validate_sql_answer", methods=["POST"])
def validate_sql_answer():
    data = request.get_json()
    answer = data.get("answer", "")
    
    
        # Validate answer against the expected answer (normalized)
    expected = session.get("sql_correct_answer", "").strip().lower()
    if answer.strip().lower() == expected:
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": f"\n{expected}"})

# --------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
