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
import pandas as pd
import sqlite3
import datetime
import requests
from logging.handlers import RotatingFileHandler
import json as _json
from datetime import datetime as _datetime
# Initialize Redis for rate limiting and IP blocking
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

# 日誌檔案設定（可透過環境變數覆蓋）
LOG_FILE_PATH = os.environ.get("LOG_FILE_PATH", "system_app.log")
LOG_MAX_BYTES = int(os.environ.get("LOG_MAX_BYTES", 5 * 1024 * 1024))  # 預設 5MB
LOG_BACKUP_COUNT = int(os.environ.get("LOG_BACKUP_COUNT", 5))  # 保留 5 個備份

# SQLite 日誌表的最大行數（超過則刪最舊）
LOG_DB_MAX_ROWS = int(os.environ.get("LOG_DB_MAX_ROWS", 5000))

# Redis 最近日誌快取（可選），會把最新 N 筆存在 redis list
REDIS_LOG_LIST = os.environ.get("REDIS_LOG_LIST", "recent_logs")
REDIS_LOG_MAXLEN = int(os.environ.get("REDIS_LOG_MAXLEN", 200))

# 是否允許 log 完整 user payload（預設關閉以免記錄敏感資料）
ALLOW_FULL_PAYLOAD_LOG = os.environ.get("ALLOW_FULL_PAYLOAD_LOG", "false").lower() in ("1","true","yes")

# ===== logging 設定 =====
logger = logging.getLogger("system_logger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    rotating_handler = RotatingFileHandler(
        LOG_FILE_PATH, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
    )
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    rotating_handler.setFormatter(formatter)
    logger.addHandler(rotating_handler)
    
# Database setup for spreadsheet quiz
DATABASE = 'quiz_progress.db'

def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            user_answer TEXT,
            correct_answer TEXT,
            correct_flag INTEGER,
            timestamp TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level TEXT,
            source TEXT,
            message TEXT,
            metadata TEXT,
            timestamp TEXT
        )
    ''')
    # index: 便於按時間或 source 查詢
    c.execute('CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_system_logs_source ON system_logs(source)')

    conn.commit()
    conn.close()

init_db()

# Define questions and answers for authentication
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

# --------------------- Authentication Routes ---------------------
@app.route("/get_question", methods=["GET"])
def get_question():
    if session.get("blocked"):
        log_event("warning", "auth", "blocked client attempted get_question", {"ip": get_client_ip()})
        return "您的 IP 已被封鎖", 403
    question = random.choice(questions)
    session["current_question"] = question["q"]
    session["correct_answer"] = question["a"]
    session["attempts"] = 0  # Reset attempts
    log_event("info", "auth", "issued auth question", {"question": question["q"], "ip": get_client_ip()})
    return question["q"]

def get_client_ip():
    # 如果你的部署在反向代理後面，確保 proxy 正確設置 X-Forwarded-For
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        # 可能是 "client, proxy1, proxy2"
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.remote_addr or "unknown"
    return ip

@app.route("/validate_answer", methods=["POST"])
def validate_answer():
    ip = get_client_ip()
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
        log_event("info", "auth", "authentication success", {"ip": ip, "question": session.get("current_question")})
        return jsonify({"success": True})
    else:
        attempts = int(redis_client.get(attempts_key) or 0) + 1
        redis_client.setex(attempts_key, 3600, attempts)  # 1 小時內有效
        log_event("warning", "auth", "authentication failed", {"ip": ip, "attempts": attempts, "answer_provided": redact_text(answer)})
        if attempts >= 4:
            redis_client.setex(block_key, 86400, 1)  # 封鎖 24 小時
            log_event("warning", "auth", "ip blocked", {"ip": ip})
            return jsonify({"success": False, "blocked": True})
        return jsonify({"success": False, "attempts": attempts})

@app.route("/blocked_ips", methods=["GET"])
def get_blocked_ips():
    blocked_keys = redis_client.keys("blocked:*")
    blocked_ips = [key.decode().split(":")[1] for key in blocked_keys]
    return jsonify({"blocked_ips": blocked_ips})

@app.route("/unblock_ip/<ip>", methods=["POST"])
def unblock_ip(ip):
    block_key = f"blocked:{ip}"
    attempts_key = f"attempts:{ip}"
    redis_client.delete(block_key)
    redis_client.delete(attempts_key)
    return jsonify({"success": True, "message": f"IP {ip} 已解鎖"})

@app.route("/unblock_all", methods=["POST"])
def unblock_all():
    blocked_keys = redis_client.keys("blocked:*")
    for key in blocked_keys:
        redis_client.delete(key)
    return jsonify({"success": True, "message": "所有封鎖的 IP 已解除"})

# --------------------- Chatroom Routes ---------------------
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
            full_response = ""
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    stream=True
                )
                for chunk in response:
                    try:
                        content = None
                        choice = chunk.choices[0]
                        delta = getattr(choice, "delta", {}) if hasattr(choice, "delta") else choice.get("delta", {})
                        content = (delta.get("content") if isinstance(delta, dict) else getattr(delta, "content", None))
                        if content:
                            full_response += content
                            yield content
                    except Exception as e:
                        yield f"[stream error chunk parse: {str(e)}]"
            except Exception as e:
                err_msg = f"錯誤：{str(e)}"
                log_event("error", "chat", "streaming error", {"exception": str(e)})
                yield err_msg
            finally:
                # 將完整 response 存 log（safe metadata）
                try:
                    log_event("info", "chat", "chat completed", {
                        "user_summary": redact_text(next((m['content'] for m in messages if m.get("role")=="user"), ""), max_len=400),
                        "response_summary": redact_text(full_response, max_len=1000)
                    })
                except Exception as e:
                    logger.error(f"log final chat failed: {e}")

        return Response(stream_with_context(generate()), mimetype="text/plain")

    except Exception as e:
        return Response(f"處理請求時發生錯誤：{str(e)}", status=500)

@app.route("/reset", methods=["POST"])
@login_required
def reset():
    return render_template("chatroom.html")

# --------------------- SQL Quiz Routes ---------------------
def count_subqueries(sql):
    return len(re.findall(r'\(\s*SELECT', sql, re.IGNORECASE))

# Define fallback questions
sql_questions = [
    {"q": "Write a SQL query to select all employees from the employees table.", "a": "SELECT * FROM employees;"},
    {"q": "Write a SQL query to find the average salary of employees in the employees table.", "a": "SELECT AVG(salary) FROM employees;"},
    {"q": "Write a SQL query to count the number of customers registered after January 1, 2020.", "a": "SELECT COUNT(*) FROM customers WHERE registration_date > '2020-01-01';"}
]

@app.route("/sql-test", methods=["GET"])
def sql_test():
    return render_template("sql-test.html")

@app.route("/get_sql_question", methods=["GET"])
def get_sql_question():
    random_seed = random.randint(1, 1000)
    random_kind = random.randint(0,3)
    difficulty  = random.uniform(0,0.5)
    kinds="ALTER"
    if random_kind==0:
        kinds="SELECT"
    elif random_kind==1:
        kinds="INSERT"
    elif random_kind==2:
        kinds="UPDATE"
    elif random_kind==3:
        kinds="ALTER"
    timestamp = int(time.time())
    prompt = f"""
Generate a SQL quiz question and its correct SQL answer based on the following table schemas.
The quiz question and answer must be in English.
Restrictions:
1. Only use the following SQL keywords, operators, and functions:AVG, MAX, MIN, SUM, COUNT ,ABSOLUTE (ABS), INT, INTEGER, DATE, DAY, MONTH, YEAR,ASC, CHAR (CHR), VALUE (VAL), LOWER, UPPER, TRIM, CHAR_LENGTH (LEN), SPACE, AT, SUBSTRING (SUBSTR/MID),
CREATE, ALTER, INSERT, DELETE, DROP, UPDATE, VALUES, INTO, SET, TABLE, TO, SELECT, DISTINCT, UNIQUE, AS, FROM,
INNER JOIN, LEFT [OUTER] JOIN, FULL [OUTER] JOIN, RIGHT [OUTER] JOIN, ON ,WHERE, BETWEEN, LIKE, NULL ,GROUP, HAVING,
ORDER, BY, ASC, DESC ,EXISTS, IN, ALL, ANY, INTERSECT, MINUS, UNION, ADD ,INDEX, VIEW,TRUE, FALSE,+,-,*, /,>, <,=, >=,<=,<>, %,_, ', AND, NOT, OR.
2. It must be a {kinds} sql command.
3. The SQL statement (correct answer) may contain at one or no sub-query (no more than one in whole answer, can be no),with difficulty:{difficulty} level.
4. Output your result as a JSON object with exactly two keys: "question" and "answer". Do not include any additional text.
5. To ensure variety, generate a unique question each time, use random seed: {random_seed} and timestamp: {timestamp} to generate structure of answer.
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
            presence_penalty=2,
            messages=messages,
            max_tokens=1000,
            temperature=2,
        )
        generated_text = response.choices[0].message.content.strip()
        
        logger.info(f"Raw API response: {generated_text}")
        
        json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
        else:
            raise ValueError("No JSON found in API response")
        
        result = json.loads(json_text)
        question_text = result.get("question", "")
        answer_text = result.get("answer", "")
        if not question_text or not answer_text:
            raise ValueError("Missing question or answer in API response")
    except Exception as e:
        logger.error(f"Error generating SQL quiz question: {e}")
        chosen = random.choice(sql_questions)
        question_text = chosen["q"]
        answer_text = chosen["a"]

    session["sql_correct_answer"] = answer_text.strip().lower()
    
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
    
    expected = session.get("sql_correct_answer", "").strip().lower()
    if answer.strip().lower() == expected:
        log_event("info", "sql_test", "sql answer correct", {"user_answer": redact_text(answer, 400)})
        return jsonify({"success": True})
    else:
        log_event("warning", "sql_test", "sql answer incorrect", {"user_answer": redact_text(answer, 400), "expected": redact_text(expected, 400)})
        return jsonify({"success": False, "message": f"\n{expected}"})

# --------------------- Spreadsheet Quiz Routes ---------------------
@app.route("/spreadsheet-test", methods=['GET', 'POST'])
def spreadsheet_test():
    if request.method == 'POST':
        user_answer = request.form.get('answer').strip()
        if 'current_question' not in session:
            return redirect(url_for('spreadsheet_test'))
        
        correct_answer = session['current_question']['correct_answer']
        question_text = session['current_question']['question']

        is_correct = (user_answer.lower().replace(" ", "") == 
                     correct_answer.lower().replace(" ", ""))

        # Save to database
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute('''INSERT INTO quiz_results 
                    (question, user_answer, correct_answer, correct_flag, timestamp) 
                    VALUES (?, ?, ?, ?, ?)''',
                 (question_text, user_answer, correct_answer, 
                  int(is_correct), datetime.datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        result = "Correct!" if is_correct else f"Incorrect. The right answer was: {correct_answer}"
        # 在儲存 quiz_results 前或後呼叫 log_event
        log_event("info", "spreadsheet_test", "user submitted spreadsheet answer", {
            "question": session.get('current_question', {}).get('question'),
            "user_answer": redact_text(user_answer),
            "is_correct": is_correct
        })
        q, table_html = generate_question()
        return render_template("spreadsheet-test.html", question=q, table_html=table_html, result=result)
    else:
        q, table_html = generate_question()
        return render_template("spreadsheet-test.html", question=q, table_html=table_html, result="")

@app.route('/spreadsheet-progress')
def spreadsheet_progress():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''SELECT question, user_answer, correct_answer, correct_flag, timestamp 
                FROM quiz_results ORDER BY id DESC LIMIT 50''')
    records = c.fetchall()
    conn.close()
    return render_template("progress.html", records=records)

def generate_question():
    try:
        question_data = generate_question_with_deepseek()
        
        if not isinstance(question_data.get("table_data"), dict):
            raise ValueError("Invalid table data format")
            
        columns = [str(col) for col in question_data["table_data"]["columns"]]
        rows = [
            [str(cell) for cell in row] 
            for row in question_data["table_data"]["rows"]
        ]
        
        df = pd.DataFrame(rows, columns=columns)
        
        question_lower = question_data["question"].lower()
        for col_letter in ["d", "e", "f"]:
            if f"column {col_letter}" in question_lower and chr(64 + ord(col_letter)) not in df.columns:
                df[chr(64 + ord(col_letter))] = ""
        
        table_html = build_excel_style_table(df)
        
        session['current_question'] = {
            'question': question_data["question"],
            'table_html': table_html,
            'correct_answer': question_data["formula"],
            'data_columns': list(df.columns)
        }
        
        return question_data["question"], table_html
        
    except Exception as e:
        logger.error(f"Question Generation Error: {str(e)}")
        return (
            "Calculate total price in Column D (Quantity B * Unit Price C)",
            build_excel_style_table(pd.DataFrame({
                "A": ["Product", "Widget", "Gadget"],
                "B": ["Quantity", 5, 3],
                "C": ["Unit Price", 12.99, 24.50],
                "D": ["Total", "", ""]
            }))
        )

def build_excel_style_table(df):
    def escape_html(cell):
        return str(cell).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    col_headers = "".join(
        f"<th class='header-cell'>{chr(65 + i)}</th>"
        for i in range(len(df.columns))
    )
    
    rows = []
    for i in range(len(df)):
        row_cells = "".join(
            f"<td>{escape_html(df.iloc[i][col])}</td>"
            for col in df.columns
        )
        rows.append(f"<tr><td class='header-cell'>{i + 1}</td>{row_cells}</tr>")
    
    return f"""
    <style>
        .excel-table {{ border-collapse: collapse; margin: 20px 0; width: 100%; }}
        .excel-table th, .excel-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
            min-width: 80px;
        }}
        .header-cell {{
            background-color: #f2f2f2;
            font-weight: bold;
            color: #333;
        }}
    </style>
    <table class='excel-table'>
        <tr><th class='header-cell'></th>{col_headers}</tr>
        {"".join(rows)}
    </table>
    """

def generate_question_with_deepseek():
    prompt = """Generate a challenging spreadsheet practice question using ONLY the following functions:
- Logical: TRUE, FALSE, AND, NOT, OR, IF, ISBLANK
- Math: ABS, INT, RAND, SQRT, ROUND
- Aggregation: AVERAGE, MAX, MIN, SUM, SUMIF
- Counting: COUNT, COUNTA, COUNTBLANK, COUNTIF
- Text: LOWER, UPPER, PROPER, TRIM, LEN, LEFT, RIGHT, MID, CONCATENATE (&), TEXT, VALUE, CHAR, FIND
- Lookup: VLOOKUP, RANK

STRICT REQUIREMENTS:
1. Question must REQUIRE using at least one of these functions (but don't mention function names)
2. Make the question challenging with multiple conditions or steps
3. Provide rich table data with 5-8 columns and 6-10 rows
4. Question should specify exactly which columns and rows to use(can be multiples)
5. Include the correct formula answer
6. Return ONLY JSON in this exact format:
{
    "question": "Clear English question text specifying columns",
    "table_data": {
        "columns": ["Column1", "Column2", ...],
        "rows": [
            ["Value1", "Value2", ...],
            ...
        ]
    },
    "formula": "=CORRECT_FORMULA",
    "function_type": "FUNCTION_CATEGORY"
}"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=3000
        )
        
        message_content = response.choices[0].message.content
        
        if '```json' in message_content:
            message_content = message_content.split('```json')[1].split('```')[0]
        
        api_data = json.loads(message_content.strip())
        
        required_keys = {"question", "table_data", "formula"}
        if not all(key in api_data for key in required_keys):
            raise ValueError("Incomplete API response")
            
        if len(api_data["table_data"]["rows"]) < 5:
            raise ValueError("Insufficient table rows for complex question")
            
        return api_data
        
    except Exception as e:
        logger.error(f"DeepSeek API Error: {str(e)}")
        log_event("error", "deepseek", "DeepSeek API Error", {"exception": str(e)})
        return generate_local_question()

def generate_local_question():
    examples = [
        {
            "question": "Calculate bonus eligibility (column D): Return 'Bonus' if Salary (column C) > 5000 and Department (column B) is 'Sales', otherwise 'Standard'",
            "table_data": {
                "columns": ["Employee", "Department", "Salary"],
                "rows": [
                    ["E101", "Sales", 5500],
                    ["E102", "HR", 4800],
                    ["E103", "IT", 5200],
                    ["E104", "Sales", 6100],
                    ["E105", "Sales", 4900]
                ]
            },
            "formula": "=IF(AND(C2>5000,B2=\"Sales\"),\"Bonus\",\"Standard\")",
            "function_type": "IF"
        },
        {
            "question": "Calculate total price (column D) as Quantity (column B) multiplied by Unit Price (column C)",
            "table_data": {
                "columns": ["Product", "Quantity", "Unit Price"],
                "rows": [
                    ["Widget", 5, 12.99],
                    ["Gadget", 3, 24.50],
                    ["Tool", 10, 8.75],
                    ["Part", 2, 15.00],
                    ["Supply", 7, 3.25]
                ]
            },
            "formula": "=B2*C2",
            "function_type": "MULTIPLICATION"
        }
    ]
    return random.choice(examples)


####logging####

# ===== 日誌儲存與裁剪工具函式 =====
def _prune_system_logs_db(max_rows=LOG_DB_MAX_ROWS):
    # 刪除最舊的超過上限紀錄
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM system_logs")
        count = c.fetchone()[0]
        if count > max_rows:
            to_delete = count - max_rows
            # 刪除最舊的 to_delete 筆
            c.execute("""
                DELETE FROM system_logs WHERE id IN (
                    SELECT id FROM system_logs ORDER BY id ASC LIMIT ?
                )
            """, (to_delete,))
            conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"prune_system_logs_db error: {e}")

def _save_log_to_db(level, source, message, metadata):
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute('''
            INSERT INTO system_logs (level, source, message, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (level, source, message, _json.dumps(metadata or {}), _datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
        # 裁剪（非同步也可以，但簡單方式：同步呼叫）
        _prune_system_logs_db()
    except Exception as e:
        logger.error(f"_save_log_to_db error: {e}")

def _push_log_to_redis(level, source, message, metadata):
    try:
        entry = {
            "ts": _datetime.utcnow().isoformat(),
            "level": level,
            "source": source,
            "message": message,
            "metadata": metadata or {}
        }
        # LPUSH 最新在前，然後 LTRIM 保持長度
        redis_client.lpush(REDIS_LOG_LIST, _json.dumps(entry, ensure_ascii=False))
        redis_client.ltrim(REDIS_LOG_LIST, 0, REDIS_LOG_MAXLEN - 1)
    except Exception as e:
        logger.error(f"_push_log_to_redis error: {e}")

def redact_text(text, max_len=1000):
    # 簡單遮罩/截斷：如果 payload 很大或包含看似 API KEY（含 "key=" 或 "api"），就遮罩
    if not text:
        return ""
    s = str(text)
    lowered = s.lower()
    if "api_key" in lowered or "api-key" in lowered or "authorization" in lowered or "bearer" in lowered:
        return "[REDACTED_SENSITIVE]"
    if len(s) > max_len:
        return s[:max_len] + "...[truncated]"
    return s

def log_event(level="info", source="app", message="", metadata=None, save_db=True, save_redis=True):
    """
    通用日誌函式：會寫入 logger（檔案）、並選擇性寫到 sqlite / redis
    level: "info" / "warning" / "error" / "debug"
    source: 事件來源（如 "auth", "chat", "deepseek", "sql_test"）
    message: 簡短文字（會被 redact）
    metadata: dict，會被 json 化並存入 DB/Redis（會視 ALLOW_FULL_PAYLOAD_LOG 標誌來決定是否完整儲存）
    """
    try:
        msg = redact_text(message)
        if level == "info":
            logger.info(f"[{source}] {msg}")
        elif level == "warning":
            logger.warning(f"[{source}] {msg}")
        elif level == "error":
            logger.error(f"[{source}] {msg}")
        else:
            logger.debug(f"[{source}] {msg}")
    except Exception:
        pass

    # 根據是否允許保存完整 payload 決定 metadata 內容
    safe_metadata = {}
    if metadata:
        if ALLOW_FULL_PAYLOAD_LOG:
            safe_metadata = metadata
        else:
            # 只保留非敏感的 metadata keys（示例策略）
            for k, v in (metadata.items() if isinstance(metadata, dict) else []):
                if k and ("key" in k.lower() or "token" in k.lower() or "password" in k.lower()):
                    safe_metadata[k] = "[REDACTED]"
                else:
                    try:
                        # 轉成字串並做截斷
                        safe_metadata[k] = redact_text(v, max_len=500)
                    except Exception:
                        safe_metadata[k] = "[UNSERIALIZABLE]"

    # 寫入 sqlite 與 redis
    if save_db:
        try:
            _save_log_to_db(level, source, msg, safe_metadata)
        except Exception as e:
            logger.error(f"log_event save_db failed: {e}")

    if save_redis:
        try:
            _push_log_to_redis(level, source, msg, safe_metadata)
        except Exception as e:
            logger.error(f"log_event push_redis failed: {e}")
            
if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)


