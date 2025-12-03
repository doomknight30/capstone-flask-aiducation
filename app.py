from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import random
import os
import requests
import json
import  google.generativeai as genai
from flask_dance.contrib.google import make_google_blueprint, google
from flask import Flask
from datetime import datetime, timedelta
from dotenv import load_dotenv
from functools import wraps
import re

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback_secret")
group_rooms = {}

# Google Generative AI configuration
genai.configure(api_key=os.environ.get("GOOGLE_GENAI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return decorated_function

def gemini_complete(messages, model_name="gemini-2.5-flash"):
    """Call Google Gemini API and return the assistant message text."""
    try:
        # Convert OpenAI-style messages to Gemini format
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += f"System: {message['content']}\n"
            elif message["role"] == "user":
                prompt += f"User: {message['content']}\n"
            elif message["role"] == "assistant":
                prompt += f"Assistant: {message['content']}\n"

        # Generate content using Gemini
        response = model.generate_content(prompt)
        result = response.text.strip()
        print(f"Gemini API response: {result[:200]}...")  # Log first 200 chars for debugging
        return result

    except Exception as e:
        print(f"Gemini API error: {e}")
        # Fallback to simple response
        if messages and len(messages) > 0:
            last_user_message = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")
            return f"I'm having trouble connecting to the AI service. Please try again. (Request: {last_user_message[:50]}...)"
        return "I'm having trouble connecting to the AI service. Please try again."

# File upload config
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

def fetch_book_text(read_link):
    """
    Fetch and extract the main text from a Project Gutenberg HTML or plain text link.
    Returns a string with the book's content, or raises Exception.
    """
    try:
        # Prefer HTML, fallback to plain text
        if read_link.endswith('.htm') or read_link.endswith('.html'):
            resp = requests.get(read_link, timeout=10)
            resp.raise_for_status()
            html = resp.text
            # Try to extract the main content between <body> tags
            body_match = re.search(r'<body.*?>(.*?)</body>', html, re.DOTALL | re.IGNORECASE)
            if body_match:
                text = re.sub('<[^<]+?>', '', body_match.group(1))  # Remove HTML tags
            else:
                text = re.sub('<[^<]+?>', '', html)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        else:
            # Assume plain text
            resp = requests.get(read_link, timeout=10)
            resp.raise_for_status()
            text = resp.text
            # Try to remove Gutenberg header/footer if present
            start = text.find("*** START OF")
            end = text.find("*** END OF")
            if start != -1 and end != -1:
                text = text[start:end]
            return text.strip()
    except Exception as e:
        print(f"Error fetching book text: {e}")
        raise Exception("Could not fetch or process the book content.")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_uploaded_file(file_storage):
    """Just search and extract a random but logically complete plain text from an uploaded file (PDF, DOCX, TXT). Returns text or raises Exception."""
    filename = secure_filename(file_storage.filename)
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""

    if ext == "txt":
        # Read as plain text
        raw = file_storage.read()
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return raw.decode("latin-1", errors="ignore")

    if ext == "pdf":
        try:
            import PyPDF2
        except ImportError:
            raise Exception("PDF support requires PyPDF2. Install with: pip install PyPDF2")
        try:
            reader = PyPDF2.PdfReader(file_storage)
            pages_text = []
            for page in reader.pages:
                pages_text.append(page.extract_text() or "")
            return "\n".join(pages_text)
        except Exception as e:
            raise Exception(f"Failed to read PDF: {str(e)}")

    if ext == "docx":
        try:
            import docx
        except ImportError:
            raise Exception("DOCX support requires python-docx. Install with: pip install python-docx")
        try:
            document = docx.Document(file_storage)
            return "\n".join(p.text for p in document.paragraphs)
        except Exception as e:
            raise Exception(f"Failed to read DOCX: {str(e)}")

    raise Exception("Unsupported file type")

# Database connection
def connect_db():
    return mysql.connector.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        user=os.environ.get("DB_USER", "root"),
        password=os.environ.get("DB_PASSWORD", ""),
        database=os.environ.get("DB_NAME", "capstone")
    )

def admin_required(f):
    @wraps(f)  # This preserves the function attributes
    def wrapper(*args, **kwargs):
        if "username" not in session or session.get("role") != "admin":
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return wrapper

@app.route("/", methods=["GET", "POST"])
def home():
    error = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        mode = request.form.get("mode")  # "registered" or "guest"

        if mode == "registered":
            conn = connect_db()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            conn.close()

            if user and check_password_hash(user['password'], password):
                session["username"] = username
                session["guest"] = False
                session["score"] = 0
                session["question_count"] = 0
                session["role"] = user.get("role")
                session["user_id"] = user["user_id"]
                return redirect(url_for("homepage"))
            else:
                error = "Invalid username or password"

        elif mode == "guest":
            session["username"] = username
            session["guest"] = True
            session["score"] = 0
            session["question_count"] = 0
            return redirect(url_for("homepage"))

    return render_template("login.html", error=error)

@app.route("/quiz_customization", methods=["GET"])
@login_required
def quiz_customization():
    if "username" not in session:
        return redirect(url_for("home"))
    return render_template("quiz-customization.html")

def fetch_book_text(read_link):
    """
    Fetch and extract the main text from a Project Gutenberg plain text link.
    Returns a string with the book's content, or raises Exception.
    """
    try:
        # Download only the first 200KB to avoid incomplete reads and huge files
        resp = requests.get(read_link, timeout=10, stream=True)
        resp.raise_for_status()
        content = b""
        max_bytes = 200_000  # 200 KB
        for chunk in resp.iter_content(8192):
            content += chunk
            if len(content) > max_bytes:
                break
        text = content.decode("utf-8", errors="ignore")
        # Remove Gutenberg header/footer
        start_match = re.search(r"\*\*\* START OF(.*?)\*\*\*", text, re.DOTALL)
        end_match = re.search(r"\*\*\* END OF(.*?)\*\*\*", text, re.DOTALL)
        start = start_match.end() if start_match else 0
        end = end_match.start() if end_match else len(text)
        main_text = text[start:end].strip()
        # Optionally, further clean up whitespace
        main_text = re.sub(r'\s+', ' ', main_text)
        return main_text
    except Exception as e:
        print(f"Error fetching book text: {e}")
        return ""

def get_book_text_from_db(book_id, user_id):
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT read_link FROM selected_books WHERE id = %s AND user_id = %s", (book_id, user_id))
    book = cursor.fetchone()
    cursor.close()
    conn.close()
    if not book:
        return ""
    return fetch_book_text(book["read_link"])

@app.route("/start_quiz_custom", methods=["POST"])
@login_required
def start_quiz_custom():
    if "username" not in session:
        return redirect(url_for("home"))

    quiz_format = request.form.get("format")
    num_items = int(request.form.get("num_items", 5))
    timed = request.form.get("timed")
    time_limit = request.form.get("time_limit")
    if time_limit is not None and time_limit != "":
        time_limit = int(time_limit)
    else:
        time_limit = 0

    session["quiz_format"] = quiz_format
    session["num_items"] = num_items
    session["timed"] = timed
    session["time_limit"] = time_limit

    # Get prompt, file, or book
    quiz_source_content = ""
    if quiz_format == "prompt":
        session["quiz_prompt"] = request.form.get("prompt_text")
        session["quiz_source_type"] = "prompt"
        quiz_source_content = session["quiz_prompt"]
    elif quiz_format == "upload":
        file = request.files.get("upload_file")
        if file and allowed_file(file.filename):
            try:
                full_text = extract_text_from_uploaded_file(file)
                session["quiz_uploaded_text"] = full_text[:2000]
                session["quiz_source_type"] = "upload"
                quiz_source_content = session["quiz_uploaded_text"]
            except Exception as e:
                return render_template("quiz-customization.html", error=str(e))
        else:
            return render_template("quiz-customization.html", error="Invalid or missing file.")
    elif quiz_format == "library":
        book_id = request.form.get("library_book_id")
        if not book_id:
            return render_template("quiz-customization.html", error="No book selected.")
        conn = connect_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT title, authors, read_link FROM selected_books WHERE id = %s AND user_id = %s", (book_id, session.get("user_id")))
        book = cursor.fetchone()
        cursor.close()
        conn.close()
        if not book:
            return render_template("quiz-customization.html", error="Book not found.")
        session["quiz_book_id"] = book_id
        session["quiz_book_title"] = book["title"]
        session["quiz_book_authors"] = book["authors"]
        session["quiz_book_read_link"] = book["read_link"]
        session["quiz_source_type"] = "library"
        quiz_source_content = f"{book['title']} by {book['authors']}"
    else:
        return render_template("quiz-customization.html", error="Invalid quiz format.")

    # Reset quiz session counters
    session["score"] = 0
    session["question_count"] = 0
    session["current_question_index"] = 0

    # Generate all questions up front and save to DB
    questions = []
    for i in range(num_items):
        if quiz_format == "prompt":
            passage = generate_varied_reading_comprehension_text(session["quiz_prompt"], i+1)
        elif quiz_format == "upload":
            passage = select_relevant_excerpt_with_variety(session["quiz_uploaded_text"], [q["passage"] for q in questions])
        elif quiz_format == "library":
            book_id = session.get("quiz_book_id")
            user_id = session.get("user_id")
            book_text = get_book_text_from_db(book_id, user_id)
            if not book_text:
                return render_template("quiz-customization.html", error="Could not fetch book content.")
            passage = select_relevant_excerpt_with_variety(book_text, [q["passage"] for q in questions])
        else:
            passage = "No valid source"
        mcq = generate_multiple_choice_question_with_context(passage, i+1)
        questions.append({
            "passage": passage,
            "question": mcq["question"],
            "choices": mcq["choices"],
            "correct_answer": mcq["correct_answer"]
        })

    db = connect_db()
    cursor = db.cursor()

    # Insert quiz metadata
    cursor.execute("""
        INSERT INTO custom_quizzes (creator_username, format, num_items, timed, time_limit, source_type, source_content)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        session["username"],
        quiz_format,
        num_items,
        timed,
        time_limit,
        session.get("quiz_source_type"),
        quiz_source_content
    ))
    quiz_id = cursor.lastrowid

    # Insert each question
    for idx, q in enumerate(questions, start=1):
        cursor.execute("""
            INSERT INTO custom_quiz_questions (quiz_id, question_number, passage, question, choices, correct_answer)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            quiz_id,
            idx,
            q["passage"],
            q["question"],
            json.dumps(q["choices"]),
            q["correct_answer"]
        ))

    db.commit()
    cursor.close()
    db.close()

    # Store only quiz_id in session
    session["quiz_id"] = quiz_id
    session["current_question_index"] = 0

    return redirect(url_for("quiz"))

@app.route("/quiz", methods=["GET", "POST"])
@login_required
def quiz():
    if "username" not in session:
        return redirect(url_for("home"))

    # Initialize score and question_count if not present
    if "score" not in session:
        session["score"] = 0
    if "question_count" not in session:
        session["question_count"] = 0
    if "quiz_id" not in session:
        return redirect(url_for("quiz_customization"))

    quiz_id = session.get("quiz_id")
    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM custom_quiz_questions WHERE quiz_id = %s ORDER BY question_number", (quiz_id,))
    questions = cursor.fetchall()
    cursor.close()
    db.close()

    idx = session.get("current_question_index", 0)
    num_items = session.get("num_items", 5)

    if request.method == "POST":
        # Save remaining time from form
        if "remaining_time" in request.form:
            try:
                session["remaining_time"] = int(request.form.get("remaining_time", session.get("remaining_time", session.get("time_limit", 0) * 60)))
            except Exception:
                session["remaining_time"] = session.get("time_limit", 0) * 60
        else:
            if "remaining_time" not in session or not isinstance(session["remaining_time"], int):
                session["remaining_time"] = int(session.get("time_limit", 0)) * 60

        if request.form.get("end_exam") == "1":
            session["question_count"] = num_items
            return redirect(url_for("result"))

        selected_answer = request.form.get("answer")
        correct_answer = questions[idx]["correct_answer"]

        if selected_answer == correct_answer:
            session["score"] += 1

        # Save the user's answer and time spent to quiz_attempts table
        time_spent = request.form.get("time_spent", type=int)
        quiz_id = session.get("quiz_id")
        question_id = questions[idx]["question_id"]  # Use the actual question_id from DB
        user_id = session["user_id"]

        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO quiz_attempts (user_id, quiz_id, question_id, answer, time_spent) VALUES (%s, %s, %s, %s, %s)",
            (user_id, quiz_id, question_id, selected_answer, time_spent)
        )
        conn.commit()
        cursor.close()
        conn.close()

        session["question_count"] += 1
        session["current_question_index"] = idx + 1

        if session["question_count"] >= num_items:
            return redirect(url_for("result"))

        idx += 1

        
    else:
        if "remaining_time" not in session:
            session["remaining_time"] = session.get("time_limit", 0) * 60

    if idx < len(questions):
        q = questions[idx]
        return render_template("quiz.html",
            question={"question_text": q["question"], "options": json.loads(q["choices"]), "correct_answer": q["correct_answer"]},
            username=session["username"],
            score=session["score"],
            generated_text=q["passage"],
            quiz_started=True,
            timed=session.get("timed"),
            time_limit=int(session.get("time_limit", 0)),
            remaining_time=int(session.get("remaining_time", int(session.get("time_limit", 0)) * 60)),
        )
    else:
        return redirect(url_for("result"))

@app.route('/library')
@login_required
def library():
    return render_template('library.html')

@app.route('/user_manual')
@login_required
def user_manual():
    return render_template('user_manual.html')

@app.route('/admin_manual')
@login_required
def admin_manual():
    return render_template('admin_user_manual.html')

@app.route('/save_book', methods=['POST'])
@login_required
def save_book():
    data = request.json
    title = data.get('title')
    authors = data.get('authors')
    read_link = data.get('read_link')
    user_id = session.get('user_id')  # Or however you track the user

    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO selected_books (user_id, title, authors, read_link, selected_at)
        VALUES (%s, %s, %s, %s, NOW())
    """, (user_id, title, authors, read_link))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"status": "success"})

@app.route('/get_selected_books')
@login_required
def get_selected_books():
    user_id = session.get('user_id')
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, title, authors FROM selected_books WHERE user_id = %s", (user_id,))
    books = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(books)

@app.route('/account', methods=['GET', 'POST'])
@login_required
def account_settings():
    if 'username' not in session:
        return redirect(url_for('home'))

    username = session['username']

    db = connect_db()
    cursor = db.cursor(dictionary=True)

    # Fetch current user details including role
    cursor.execute("SELECT user_id, username, email, role FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()

    if request.method == 'POST':
        new_username = request.form['username']
        new_email = request.form['email']
        new_password = request.form['password']

        # Check if new username or email already exists
        cursor.execute("SELECT * FROM users WHERE (username = %s OR email = %s) AND user_id != %s", (new_username, new_email, user['user_id']))
        existing_user = cursor.fetchone()

        if existing_user:
            return render_template('account.html', user=user, message="Username or Email already exists!")

        # Update details in database
        if new_password:
            hashed_password = generate_password_hash(new_password)
            cursor.execute("UPDATE users SET username = %s, email = %s, password = %s WHERE user_id = %s",
                           (new_username, new_email, hashed_password, user['user_id']))
        else:
            cursor.execute("UPDATE users SET username = %s, email = %s WHERE user_id = %s",
                           (new_username, new_email, user['user_id']))

        db.commit()
        session['username'] = new_username  # Update session with new username

        return render_template('account.html', user={'username': new_username, 'email': new_email, 'role': user['role']}, message="Account updated successfully!")

    cursor.close()
    db.close()
    return render_template('account.html', user=user)

@app.route('/create_child_account', methods=['POST'])
@login_required
def create_child_account():
    if 'username' not in session:
        return redirect(url_for('home'))

    username = session['username']

    # Check if user has parent role
    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT user_id, role FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()

    if not user or user['role'] != 'parent':
        cursor.close()
        db.close()
        return redirect(url_for('home'))

    if request.method == 'POST':
        child_username = request.form['child_username']
        child_password = request.form['child_password']
        child_first_name = request.form['child_first_name']
        child_last_name = request.form['child_last_name']
        child_birthdate = request.form['child_birthdate']
        child_year_level = request.form['child_year_level']
        child_school = request.form.get("child_school")

        # Check if child username already exists
        cursor.execute("SELECT * FROM users WHERE username = %s", (child_username,))
        existing_child = cursor.fetchone()

        if existing_child:
            cursor.close()
            db.close()
            return render_template('account.html',
                                user={'username': username, 'role': 'parent'},
                                message="Child username already taken!")

        try:
            # Create kid user account
            hashed_password = generate_password_hash(child_password)
            cursor.execute("""
                INSERT INTO users (username, password, role, first_name, last_name, birthdate)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (child_username, hashed_password, 's.child', child_first_name, child_last_name, child_birthdate))

            child_user_id = cursor.lastrowid

            # Create child info record
            cursor.execute("""
                INSERT INTO child_info (user_id, first_name, last_name, birthdate, year_level, school)
                VALUES (%s, %s, %s, %s, %s, %s)
                """, (child_user_id, child_first_name, child_last_name, child_birthdate, child_year_level, child_school))

            db.commit()

            cursor.close()
            db.close()

            return render_template('account.html',
                                user={'username': username, 'role': 'parent'},
                                message="Child account created successfully!")

        except Exception as e:
            db.rollback()
            cursor.close()
            db.close()
            return render_template('account.html',
                                user={'username': username, 'role': 'parent'},
                                message=f"Error creating child account: {str(e)}")

    cursor.close()
    db.close()
    return redirect(url_for('account_settings'))

@app.route("/result")
@login_required
def result():
    if "username" not in session:
        return redirect(url_for("home"))

    username = session["username"]
    total_questions = session.get("question_count", 0)
    correct_answers = session.get("score", 0)
    current_difficulty = "medium"  # Adjust this dynamically if needed

    # Establish database connection
    db = connect_db()
    cursor = db.cursor()

    # Get user_id based on username
    user_id = get_user_id(username, cursor)

    if user_id:
        quiz_id = session.get("quiz_id")
        topic = session.get("quiz_topic") or session.get("quiz_book_title") or session.get("quiz_prompt")
        cursor.execute(
            """
            INSERT INTO user_progress (user_id, quiz_id, topic, total_questions, correct_answers, current_difficulty, last_updated)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """,
            (user_id, quiz_id, topic, total_questions, correct_answers, current_difficulty)
        )
        db.commit()

    cursor.close()
    db.close()

    session.pop("questions", None)
    session.pop("current_question_index", None)
    session.pop("score", None)
    session.pop("question_count", None)
    session.pop("quiz_format", None)
    session.pop("timed", None)
    session.pop("time_limit", None)
    session.pop("quiz_prompt", None)
    session.pop("quiz_uploaded_text", None)
    session.pop("quiz_source_type", None)

    return render_template("result.html", total_questions=total_questions, correct_answers=correct_answers, username = username)

@app.route("/homepage")
@login_required
def homepage():
    username = session.get("username")
    role = session.get("role")

    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT first_name, created_at FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    show_welcome_panel = False
    first_name = None
    if user:
        first_name = user.get("first_name")
        created_at = user.get("created_at")
        if created_at:
            if isinstance(created_at, str):
                from datetime import datetime
                created_at = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
            from datetime import datetime, timedelta
            show_welcome_panel = (datetime.now() - created_at) < timedelta(days=1)

    group_exams = []
    quiz_exams = []
    if not show_welcome_panel:
        # Fetch group exams
        cursor.execute("""
            SELECT g.group_id, g.created_at, s.score
            FROM group_exam_scores s
            JOIN group_exams g ON s.group_id = g.group_id
            WHERE s.username = %s
            ORDER BY g.created_at DESC
        """, (username,))
        group_exams = cursor.fetchall()

        # Fetch individual quizzes from user_progress
        cursor.execute("""
            SELECT total_questions, correct_answers, last_updated
            FROM user_progress
            WHERE user_id = (
                SELECT user_id FROM users WHERE username = %s
            )
            ORDER BY last_updated DESC
        """, (username,))
        quiz_exams = cursor.fetchall()

    cursor.close()
    db.close()
    return render_template(
        "homepage.html",
        first_name=first_name,
        role=role,
        show_welcome_panel=show_welcome_panel,
        group_exams=group_exams,
        quiz_exams=quiz_exams
    )

def get_user_id(username, cursor):
    cursor.execute("SELECT user_id FROM users WHERE username = %s", (username,))
    result = cursor.fetchone()
    return result[0] if result else None

@app.route("/logout")
def logout():
    if "username" in session:
        session.clear()
    return redirect(url_for("home"))

# Function to generate reading comprehension quiz text using DeepSeek via OpenRouter
def generate_reading_comprehension_text(topic):
    messages = [
        {"role": "system", "content": "You write short, simple passages for reading comprehension. Respond with only the passage text."},
        {"role": "user", "content": f"Write a short, concise and simple 1-3 sentence passage about {topic}."},
    ]
    return gemini_complete(messages)

def generate_multiple_choice_question(text):
    messages = [
        {"role": "system", "content": "Create one MCQ from a passage. Respond with ONLY valid JSON: {\"question\": string, \"choices\": [string,string,string,string], \"correct_answer\": string}. No explanations or extra fields."},
        {"role": "user", "content": f"Based on the following text, create the MCQ.\n\n{text}"},
    ]
    generated_text = gemini_complete(messages)

    print(f"Raw API response for MCQ: {generated_text[:200]}...")  # Debug logging

    try:
        # First try to parse directly as JSON
        mcq = json.loads(generated_text)
    except json.JSONDecodeError:
        try:
            # If direct parsing fails, try to extract JSON from the response
            # Look for JSON patterns in the text
            import re
            # Try to find JSON object in the response
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                mcq = json.loads(json_str)
                print(f"Extracted JSON from response: {json_str[:100]}...")
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            print(f"Failed to parse JSON from Gemini response: {e}")
            print(f"Full response was: {generated_text}")
            # Create a fallback question based on the text
            mcq = {
                "question": f"What is the main topic of this text?",
                "choices": [
                    "A general topic",
                    "The specific subject mentioned",
                    "Something unrelated",
                    "Not enough information"
                ],
                "correct_answer": "The specific subject mentioned"
            }

    # Validate the MCQ structure
    if not mcq.get("question") or not mcq.get("choices") or not mcq.get("correct_answer"):
        print(f"Invalid MCQ structure: {mcq}")
        mcq = {
            "question": f"Based on the text: {text[:50]}...",
            "choices": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": "Option A"
        }

    return mcq

@app.route('/generate_quiz_text', methods=['POST'])
@login_required
def generate_quiz_text():
    data = request.json
    topic = data.get('topic')

    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    try:
        generated_text = generate_reading_comprehension_text(topic)
        mcq = generate_multiple_choice_question(generated_text)
        return jsonify({
            "generated_text": generated_text,
            "topic": topic,
            "question": mcq.get("question"),
            "choices": mcq.get("choices"),
            "correct_answer": mcq.get("correct_answer")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def select_relevant_excerpt(full_text):
    """
    Use AI to select a short, relevant excerpt from the document text.
    """
    messages = [
        {"role": "system", "content": "From the provided text, return only a short self-contained excerpt (1-4 sentences). No explanations."},
        {"role": "user", "content": f"Select an excerpt from this text:\n\n{full_text}"},
    ]
    return gemini_complete(messages)

@app.route('/upload_document', methods=['POST'])
def upload_document():
    if 'document' not in request.files:
        return jsonify({"error": "No document uploaded"}), 400

    file = request.files['document']

    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Allowed: pdf, docx, txt"}), 400

    try:
        # Extract full text
        full_text = extract_text_from_uploaded_file(file)

        # Store the full document text in session for future question generation
        session["full_document_text"] = full_text

        # Let AI select an excerpt
        excerpt = select_relevant_excerpt(full_text)

        # Generate MCQ from excerpt
        mcq = generate_multiple_choice_question(excerpt)

        return jsonify({
            "generated_text": excerpt,
            "question": mcq.get("question"),
            "choices": mcq.get("choices"),
            "correct_answer": mcq.get("correct_answer")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/admin")
@admin_required
def admin_dashboard():
    return render_template("admin.html")

@app.route('/admin_analytics')
@admin_required
def admin_analytics():
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    # Total users
    cursor.execute("SELECT COUNT(*) AS total_users FROM users")
    total_users = cursor.fetchone()["total_users"]

    # Total quizzes
    cursor.execute("SELECT COUNT(*) AS total_quizzes FROM custom_quizzes")
    total_quizzes = cursor.fetchone()["total_quizzes"]

    # Average score (from user_progress)
    cursor.execute("SELECT SUM(correct_answers) AS total_correct, SUM(total_questions) AS total_questions FROM user_progress")
    row = cursor.fetchone()
    if row["total_questions"]:
        average_score = round((row["total_correct"] / row["total_questions"]) * 100, 2)
    else:
        average_score = 0

    # Recent activity: last 5 users
    cursor.execute("SELECT username, created_at FROM users ORDER BY created_at DESC LIMIT 5")
    recent_users = cursor.fetchall()

    cursor.close()
    conn.close()
    return {
        "total_users": total_users,
        "total_quizzes": total_quizzes,
        "average_score": average_score,
        "recent_users": recent_users
    }

# Add announcement (admin only)
@app.route('/add_announcement', methods=['POST'])
@admin_required
def add_announcement():
    if session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    data = request.json
    title = data.get("title")
    content = data.get("content")
    posted_by = session.get("username")
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO announcements (title, content, posted_by) VALUES (%s, %s, %s)",
        (title, content, posted_by)
    )
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": "Announcement posted!"})

# Get all announcements (for all users)
@app.route('/get_announcements')
def get_announcements():
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM announcements ORDER BY posted_at DESC")
    announcements = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(announcements)

@app.route("/group_exam")
@login_required
def group_exam():
    if 'username' not in session:
        return redirect(url_for('home'))
    return render_template("group_exam.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    success = None

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        user_type = request.form.get("user_type")
        first_name = request.form.get("first_name")
        middle_name = request.form.get("middle_name")
        last_name = request.form.get("last_name")
        email = request.form.get("email")
        birthdate = request.form.get("birthdate")
        contact_no = request.form.get("contact_no")


        if password != confirm_password:
            error = "Passwords do not match."
        else:
            conn = connect_db()
            cursor = conn.cursor(dictionary=True)

            # Check if username already exists
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            existing_user = cursor.fetchone()

            if existing_user:
                error = "Username already taken."
            else:
                hashed_password = generate_password_hash(password)
                cursor.execute("""
                    INSERT INTO users (username, password, role, first_name, middle_name, last_name, email, birthdate, contact_no)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (username, hashed_password, user_type, first_name, middle_name, last_name, email, birthdate, contact_no))
                user_id = cursor.lastrowid


                conn.commit()
                success = "Registration successful! You can now log in."

            conn.close()

    return render_template("register.html", error=error, success=success)

@app.route('/get_users', methods=['GET'])
def get_users():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, username, password, role, email, birthdate, created_at FROM users")
    users = cursor.fetchall()
    conn.close()
    user_list = [
        {
            "user_id": row[0],
            "username": row[1],
            "password": row[2],
            "role": row[3],
            "email":row[4],
            "birthdate": row[5],
            "created_at": row[6]
        }
        for row in users
    ]

    return jsonify(user_list)

from werkzeug.security import generate_password_hash, check_password_hash

@app.route('/add_user', methods=['POST'])
def add_user():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    role = data.get("role")
    email = data.get("email")
    birthdate = data.get("birthdate")

    if not username or not password or not role or not email or not birthdate:
        return jsonify({"error": "All fields are required"}), 400

    # Hash the password before saving
    hashed_password = generate_password_hash(password)

    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO users (username, password, role, email, birthdate)
        VALUES (%s, %s, %s, %s, %s)
    """, (username, hashed_password, role, email, birthdate))
    conn.commit()
    conn.close()

    return jsonify({"message": "User added successfully"}), 201


@app.route('/edit_user/<int:user_id>', methods=['PUT'])
def edit_user(user_id):
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400

        username = data.get("username")
        role = data.get("role")
        email = data.get("email")
        birthdate = data.get("birthdate")
        password = data.get("password")  # optional, only if updating

        if not username or not role or not email or not birthdate:
            return jsonify({"error": "All fields are required"}), 400

        conn = connect_db()
        cursor = conn.cursor()

        if password:
            hashed_password = generate_password_hash(password)
            cursor.execute("""
                UPDATE users
                SET username = %s, role = %s, email = %s, birthdate = %s, password = %s
                WHERE user_id = %s
            """, (username, role, email, birthdate, hashed_password, user_id))
        else:
            cursor.execute("""
                UPDATE users
                SET username = %s, role = %s, email = %s, birthdate = %s
                WHERE user_id = %s
            """, (username, role, email, birthdate, user_id))

        conn.commit()
        conn.close()

        return jsonify({"message": "User updated successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/delete_user/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
    conn.commit()
    conn.close()

    return jsonify({"message": "User deleted successfully"})

@app.route('/get_questions', methods=['GET'])
def get_questions():
    """
    Retrieve all questions from custom_quiz_questions and group_exam_questions,
    including the creator (username) for each question.
    Returns a JSON list with a 'source' field and 'creator' field.
    """
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    # Fetch from custom_quiz_questions and join with custom_quizzes to get creator
    cursor.execute("""
        SELECT
            q.question_id, q.quiz_id, q.question_number, q.passage, q.question, q.choices, q.correct_answer,
            'custom_quiz' AS source,
            cq.creator_username AS creator
        FROM custom_quiz_questions q
        JOIN custom_quizzes cq ON q.quiz_id = cq.quiz_id
    """)
    custom_questions = cursor.fetchall()

    # Fetch from group_exam_questions and join with group_exams to get host (creator)
    cursor.execute("""
        SELECT
            q.id AS question_id, q.group_id, q.question_number, q.passage, q.question, q.choices, q.correct_answer,
            'group_exam' AS source,
            ge.host_username AS creator
        FROM group_exam_questions q
        JOIN group_exams ge ON q.group_id = ge.group_id
    """)
    group_questions = cursor.fetchall()

    cursor.close()
    conn.close()

    # Combine and return as JSON
    all_questions = custom_questions + group_questions
    return jsonify(all_questions)

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session.get('user_id')
    # Fetch user info
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT first_name, birthdate, role FROM users WHERE user_id = %s", (user_id,))
    user = cursor.fetchone()
    # Fetch accepted lesson plans
    cursor.execute("""
        SELECT lp.plan_id, lp.plan_title, lp.topic1, lp.topic2, lp.file1_path, lp.file2_path, u.username AS teacher_name
        FROM learning_plan_applications lpa
        JOIN learning_plans lp ON lpa.plan_id = lp.plan_id
        JOIN users u ON lp.teacher_id = u.user_id
        WHERE lpa.student_id = %s AND lpa.status = 'accepted'
        ORDER BY lpa.applied_at DESC
    """, (user_id,))
    accepted_plans = cursor.fetchall()
    # Fetch quizzes taken by the user
    cursor.execute("""
    SELECT up.quiz_id, cq.source_content, up.topic, up.total_questions, up.correct_answers, up.last_updated
        FROM user_progress up
        LEFT JOIN custom_quizzes cq ON up.quiz_id = cq.quiz_id
        WHERE up.user_id = %s
        ORDER BY up.last_updated DESC
    """, (user_id,))
    quizzes = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template("dashboard.html", user=user, accepted_plans=accepted_plans, quizzes=quizzes)

@app.route('/get_quizzes_by_topics')
@login_required
def get_quizzes_by_topics():
    user_id = session.get('user_id')
    topics_param = request.args.get('topics', '')
    if not topics_param:
        return jsonify([])

    # Normalize topics: split, trim, lowercase
    topics = [t.strip().lower() for t in topics_param.split(',') if t.strip()]
    if not topics:
        return jsonify([])

    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    # Fetch quizzes for this user where topic matches any of the topics (case-insensitive)
    # We'll use LOWER() in SQL for case-insensitive match
    format_strings = ','.join(['%s'] * len(topics))
    query = f"""
        SELECT up.quiz_id, cq.source_content, up.topic, up.total_questions, up.correct_answers, up.last_updated
        FROM user_progress up
        LEFT JOIN custom_quizzes cq ON up.quiz_id = cq.quiz_id
        WHERE up.user_id = %s AND LOWER(up.topic) IN ({format_strings})
        ORDER BY up.last_updated DESC
    """
    params = [user_id] + topics
    cursor.execute(query, params)
    quizzes = cursor.fetchall()

    # Prepare output
    result = []
    for q in quizzes:
        result.append({
            'quiz_id': q['quiz_id'],
            'topic': (q['topic'] or q['source_content'] or '').strip(),
            'correct_answers': q['correct_answers'] or 0,
            'total_questions': q['total_questions'] or 0,
            'last_updated': q['last_updated'].strftime('%Y-%m-%d %H:%M') if q['last_updated'] else 'N/A'
        })

    cursor.close()
    conn.close()
    return jsonify(result)

@app.route('/get_quiz_questions/<int:quiz_id>')
def get_quiz_questions(quiz_id):
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT question, passage, choices, correct_answer
        FROM custom_quiz_questions
        WHERE quiz_id = %s
        ORDER BY question_number
    """, (quiz_id,))
    questions = cursor.fetchall()
    for q in questions:
        # Parse choices from JSON string if needed
        if isinstance(q["choices"], str):
            try:
                q["choices"] = json.loads(q["choices"])
            except Exception:
                q["choices"] = []
    cursor.close()
    conn.close()
    return jsonify(questions)

@app.route('/get_quiz_details/<int:quiz_id>')
@login_required
def get_quiz_details(quiz_id):
    user_id = session.get('user_id')
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    # Get all user's answers for this quiz (one row per question)
    cursor.execute("""
        SELECT question_id, answer
        FROM quiz_attempts
        WHERE user_id = %s AND quiz_id = %s
    """, (user_id, quiz_id))
    user_attempts = cursor.fetchall()
    user_answers = {}
    for row in user_attempts:
        user_answers[row['question_id']] = row['answer']

    # Get quiz questions
    cursor.execute("""
        SELECT question_id, question_number, passage, question, choices, correct_answer
        FROM custom_quiz_questions
        WHERE quiz_id = %s
        ORDER BY question_number ASC
    """, (quiz_id,))
    questions = cursor.fetchall()
    for q in questions:
        q['choices'] = json.loads(q['choices']) if q['choices'] else []
        q['user_answer'] = user_answers.get(q['question_id'], None)
    cursor.close()
    conn.close()
    return jsonify(questions)

@app.route('/ai_analyze_quiz/<int:quiz_id>')
@login_required
def ai_analyze_quiz(quiz_id):
    user_id = session.get('user_id')
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    # Fetch all questions for the quiz
    cursor.execute("""
        SELECT question_id, question_number, passage, question, choices, correct_answer
        FROM custom_quiz_questions
        WHERE quiz_id = %s
        ORDER BY question_number ASC
    """, (quiz_id,))
    questions = cursor.fetchall()

    # Fetch user's answers and time spent for each question
    cursor.execute("""
        SELECT question_id, answer as user_answer, time_spent
        FROM quiz_attempts
        WHERE user_id = %s AND quiz_id = %s
    """, (user_id, quiz_id))
    attempts = {row['question_id']: row for row in cursor.fetchall()}

    # Merge data for AI
    quiz_data = []
    for q in questions:
        qid = q['question_id']
        quiz_data.append({
            "question_number": q['question_number'],
            "question": q['question'],
            "choices": json.loads(q['choices']) if q['choices'] else [],
            "correct_answer": q['correct_answer'],
            "user_answer": attempts.get(qid, {}).get('user_answer'),
            "time_spent": attempts.get(qid, {}).get('time_spent')
        })

    cursor.close()
    conn.close()

    # Build prompt for AI
    prompt = (
        "You are an educational AI assistant. Analyze the following quiz attempt. "
        "For each question, state if the user's answer is correct, comment on the time spent, "
        "and provide specific feedback. After all questions, give an overall summary of the user's strengths, weaknesses, and suggestions for improvement.\n\n"
        "Quiz Questions and Answers:\n"
    )
    for q in quiz_data:
        prompt += (
            f"Question {q['question_number']}:\n"
            f"  Q: {q['question']}\n"
            f"  Choices: {q['choices']}\n"
            f"  Correct Answer: {q['correct_answer']}\n"
            f"  User Answer: {q['user_answer']}\n"
            f"  Time Spent: {q['time_spent']} seconds\n"
        )
    prompt += "\nPlease provide your analysis in this format:\n" \
              "1. For each question: Correct/Incorrect, feedback, and time comment.\n" \
              "2. Overall summary: strengths, weaknesses, and suggestions.\n" \
              "Format your response using HTML with <h3> for each question, <ul> for feedback, and <b> for highlights."

    # Call Gemini AI (or your model)
    ai_response = gemini_complete([{"role": "user", "content": prompt}])
    analysis = ai_response if isinstance(ai_response, str) else str(ai_response)

    return jsonify({"analysis": analysis})

@app.route('/set_generated_question', methods=['POST'])
def set_generated_question():
    data = request.json
    question_text = data.get("question")
    choices = data.get("choices")
    correct_answer = data.get("correct_answer")
    generated_text = data.get("generated_text")
    use_custom_source = data.get("use_custom_source", False)
    topic = data.get("topic")

    if not question_text or not choices or not correct_answer:
        return jsonify({"error": "Missing question data"}), 400

    session["generated_question"] = question_text
    session["generated_choices"] = choices
    session["correct_answer"] = correct_answer

    if use_custom_source and generated_text:
        session["custom_quiz_active"] = True
        session["quiz_source_type"] = "uploaded_file"
        # Store both the excerpt (for display) and the full text (for future excerpts)
        if "full_document_text" in session:
            session["generated_text"] = generated_text  # current excerpt shown
        else:
            # fallback, keep at least the excerpt
            session["full_document_text"] = generated_text
            session["generated_text"] = generated_text

    elif topic and not use_custom_source:
        # For prompt-based quizzes: store the topic and mark as topic-based
        session["current_topic"] = topic
        session["topic_quiz_active"] = True
        session["quiz_source_type"] = "prompt_topic"
        # Clear uploaded file quiz state
        session.pop("custom_quiz_active", None)
        session.pop("source_text", None)

    # Always store the current generated text for display
    if generated_text:
        session["generated_text"] = generated_text

    return jsonify({"message": "Generated question set in session"})

def select_relevant_excerpt_with_variety(full_text, used_excerpts=None):
    """
    Use AI to select a short, relevant excerpt from the document text,
    ensuring variety by avoiding previously used excerpts.
    """
    if used_excerpts is None:
        used_excerpts = []

    # Create a prompt that encourages variety
    variety_instruction = ""
    if used_excerpts:
        variety_instruction = f"\n\nPreviously used excerpts to AVOID repeating:\n" + "\n".join([f"- {excerpt[:100]}..." for excerpt in used_excerpts[-3:]])  # Show last 3 to avoid

    messages = [
        {
            "role": "system",
            "content": f"From the provided text, return only a short self-contained excerpt (2-4 sentences) that is DIFFERENT from any previously used excerpts. Focus on a different aspect, topic, or section of the document. No explanations.{variety_instruction}"
        },
        {"role": "user", "content": f"Select a NEW and DIFFERENT excerpt from this text:\n\n{full_text}"},
    ]
    return gemini_complete(messages)

def generate_varied_reading_comprehension_text(topic, question_number=1, previous_passages=None):
    """
    Generate reading comprehension text with variety based on question number and previous content.
    """
    if previous_passages is None:
        previous_passages = []

    # Create different aspects/angles for the same topic
    aspects = [
        f"basic introduction to {topic}",
        f"interesting facts about {topic}",
        f"how {topic} works or functions",
        f"the history or origin of {topic}",
        f"the importance or benefits of {topic}",
        f"different types or categories of {topic}",
        f"recent discoveries or developments about {topic}",
        f"practical applications of {topic}"
    ]

    # Select a different aspect based on question number
    aspect_index = (question_number - 1) % len(aspects)
    chosen_aspect = aspects[aspect_index]

    # Add instruction to avoid repetition if we have previous passages
    variety_instruction = ""
    if previous_passages:
        variety_instruction = f"\n\nPreviously covered content to AVOID repeating:\n" + "\n".join([f"- {passage[:80]}..." for passage in previous_passages[-2:]])

    messages = [
        {
            "role": "system",
            "content": f"You write short, simple passages for reading comprehension. Focus on {chosen_aspect}. Make each passage unique and different from previous ones. Respond with only the passage text (2-4 sentences).{variety_instruction}"
        },
        {"role": "user", "content": f"Write a short, concise passage focusing on {chosen_aspect}."},
    ]
    return gemini_complete(messages)

def generate_multiple_choice_question_with_context(text, question_number=1):
    """
    Generate MCQ with more variety based on question number and text content.
    """
    # Different question types to encourage variety
    question_types = [
        "main idea or topic",
        "specific detail mentioned",
        "cause and effect relationship",
        "comparison or contrast",
        "sequence or process",
        "purpose or function",
        "characteristic or feature",
        "location or setting"
    ]

    question_type = question_types[(question_number - 1) % len(question_types)]

    messages = [
        {
            "role": "system",
            "content": f"Create one MCQ focusing on {question_type} from a passage. Respond with ONLY valid JSON: {{\"question\": string, \"choices\": [string,string,string,string], \"correct_answer\": string}}. Make the question and all choices clearly distinct and avoid generic options."
        },
        {"role": "user", "content": f"Based on the following text, create an MCQ about {question_type}:\n\n{text}"},
    ]

    generated_text = gemini_complete(messages)

    print(f"Raw API response for MCQ (Question #{question_number}): {generated_text[:200]}...")

    try:
        mcq = json.loads(generated_text)
    except json.JSONDecodeError:
        try:
            import re
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                mcq = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            print(f"Failed to parse JSON from Gemini response: {e}")
            # Create a more varied fallback based on question number
            fallback_questions = [
                f"What is the main subject discussed in this text?",
                f"Which detail is mentioned in the passage?",
                f"What relationship is described in the text?",
                f"According to the passage, what happens?",
                f"What characteristic is highlighted in the text?"
            ]
            fallback_question = fallback_questions[(question_number - 1) % len(fallback_questions)]

            mcq = {
                "question": fallback_question,
                "choices": [
                    "First possible answer",
                    "Second possible answer",
                    "Third possible answer",
                    "Fourth possible answer"
                ],
                "correct_answer": "First possible answer"
            }

    # Validate the MCQ structure
    if not mcq.get("question") or not mcq.get("choices") or not mcq.get("correct_answer"):
        print(f"Invalid MCQ structure: {mcq}")
        mcq = {
            "question": f"Question #{question_number}: What does this passage discuss?",
            "choices": ["Topic A", "Topic B", "Topic C", "Topic D"],
            "correct_answer": "Topic A"
        }
    return mcq

def generate_multiple_choice_question_with_complexity(text, question_number=1, difficulty="medium"):
    """
    Generate MCQ with varying complexity levels.
    Difficulty levels: "easy", "medium", "hard"
    """
    # Question types, from simple to complex
    question_types = {
        "easy": [
            "recall basic facts or information",
            "identify main idea",
            "define key terms",
            "list important points",
            "describe what happened"
        ],
        "medium": [
            "explain relationships between concepts",
            "compare and contrast ideas",
            "classify information",
            "interpret meaning",
            "predict what might happen next"
        ],
        "hard": [
            "analyze cause and effect",
            "evaluate arguments or claims",
            "synthesize information from multiple parts",
            "justify a position or decision",
            "create new connections or insights"
        ]
    }

    # Select complexity based on progress
    if question_number <= 2:
        current_difficulty = "easy"
    elif question_number <= 4:
        current_difficulty = "medium"
    else:
        current_difficulty = "hard"

    # Get question types for current difficulty
    current_types = question_types[current_difficulty]
    question_type = current_types[question_number % len(current_types)]

    messages = [
        {
            "role": "system",
            "content": f"""Create one {current_difficulty}-level MCQ focusing on {question_type}.
            For {current_difficulty} questions:
            - Easy: Focus on direct recall and basic comprehension
            - Medium: Require analysis and application
            - Hard: Need evaluation and synthesis of information
            
            Respond with ONLY valid JSON: {{"question": string, "choices": [string,string,string,string], "correct_answer": string}}.
            Make choices distinctly different and avoid obvious wrong answers."""
        },
        {"role": "user", "content": f"Based on this text, create a {current_difficulty} question about {question_type}:\n\n{text}"}
    ]

    generated_text = gemini_complete(messages)
    
    try:
        mcq = json.loads(generated_text)
    except json.JSONDecodeError:
        # Your existing fallback logic with complexity levels
        fallback_questions = {
            "easy": [
                "What is directly stated in the text?",
                "Which basic fact is mentioned?",
                "What is the main topic?"
            ],
            "medium": [
                "How do the concepts relate to each other?",
                "What can be inferred from the passage?",
                "What is the underlying meaning?"
            ],
            "hard": [
                "What conclusions can be drawn?",
                "How would you evaluate the argument?",
                "What evidence supports the main claim?"
            ]
        }
        
        mcq = {
            "question": fallback_questions[current_difficulty][question_number % 3],
            "choices": [
                "First possible answer",
                "Second possible answer",
                "Third possible answer",
                "Fourth possible answer"
            ],
            "correct_answer": "First possible answer"
        }

    return mcq, current_difficulty

@app.route('/submit_answer_and_generate_new', methods=['POST'])
def submit_answer_and_generate_new():
    """Submit answer and generate new content with improved variety"""
    if "username" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data = request.json
    selected_answer = data.get("answer")

    if not selected_answer:
        return jsonify({"error": "No answer provided"}), 400

    # Check if answer is correct
    correct_answer = session.get("correct_answer")
    is_correct = selected_answer == correct_answer

    # Update score
    if is_correct:
        session["score"] = session.get("score", 0) + 1

    # Increment question count
    session["question_count"] = session.get("question_count", 0) + 1
    current_question_number = session["question_count"] + 1  # For next question

    # Check if quiz is finished
    if session["question_count"] >= 5:
        return jsonify({
            "score": session["score"],
            "quiz_finished": True
        })

    # Store the current passage in history for variety checking
    current_passage = session.get("generated_text", "")
    if current_passage:
        # Initialize or update passage history
        if "passage_history" not in session:
            session["passage_history"] = []
        session["passage_history"].append(current_passage)
        # Keep only last 3 passages to avoid memory buildup
        session["passage_history"] = session["passage_history"][-3:]

    # Clear current question data before generating new content
    session.pop("generated_question", None)
    session.pop("generated_choices", None)
    session.pop("correct_answer", None)

    # Generate new content with variety
    try:
        quiz_source_type = session.get("quiz_source_type")

        if quiz_source_type == "uploaded_file" and session.get("full_document_text"):
            # Pick a NEW excerpt from the full document each time
            full_text = session.get("full_document_text")
            used_excerpts = session.get("passage_history", [])

            # Try multiple times to get a different excerpt
            max_attempts = 3
            for attempt in range(max_attempts):
                excerpt = select_relevant_excerpt_with_variety(full_text, used_excerpts)

                # Check if this excerpt is significantly different from previous ones
                is_different = True
                for used_excerpt in used_excerpts[-2:]:  # Check last 2
                    if len(excerpt) > 0 and len(used_excerpt) > 0:
                        # Simple similarity check - if more than 70% of words are the same, it's too similar
                        excerpt_words = set(excerpt.lower().split())
                        used_words = set(used_excerpt.lower().split())
                        if len(excerpt_words & used_words) / max(len(excerpt_words), 1) > 0.7:
                            is_different = False
                            break

                if is_different or attempt == max_attempts - 1:
                    break

            mcq = generate_multiple_choice_question_with_context(excerpt, current_question_number)

            # Update session with new content
            session["generated_text"] = excerpt
            session["generated_question"] = mcq.get("question")
            session["generated_choices"] = mcq.get("choices")
            session["correct_answer"] = mcq.get("correct_answer")

            new_content = {
                "generated_text": excerpt,
                "question": mcq.get("question"),
                "choices": mcq.get("choices"),
                "correct_answer": mcq.get("correct_answer"),
                "topic": "Document Content"
            }

            return jsonify({
                "score": session["score"],
                "quiz_finished": False,
                "new_content": new_content,
                "use_custom_source": True
            })

        elif quiz_source_type == "prompt_topic" and session.get("current_topic"):
            # For topic-based quizzes, generate varied content
            current_topic = session.get("current_topic")
            previous_passages = session.get("passage_history", [])

            generated_text = generate_varied_reading_comprehension_text(
                current_topic, current_question_number, previous_passages
            )
            mcq = generate_multiple_choice_question_with_context(generated_text, current_question_number)

            # Update session with new content
            session["generated_text"] = generated_text
            session["generated_question"] = mcq.get("question")
            session["generated_choices"] = mcq.get("choices")
            session["correct_answer"] = mcq.get("correct_answer")

            new_content = {
                "generated_text": generated_text,
                "question": mcq.get("question"),
                "choices": mcq.get("choices"),
                "correct_answer": mcq.get("correct_answer"),
                "topic": current_topic
            }

            return jsonify({
                "score": session["score"],
                "quiz_finished": False,
                "new_content": new_content,
                "use_custom_source": False
            })

        else:
            # Fallback with variety
            fallback_topic = "general knowledge"
            previous_passages = session.get("passage_history", [])

            generated_text = generate_varied_reading_comprehension_text(
                fallback_topic, current_question_number, previous_passages
            )
            mcq = generate_multiple_choice_question_with_context(generated_text, current_question_number)

            # Update session with new content
            session["generated_text"] = generated_text
            session["generated_question"] = mcq.get("question")
            session["generated_choices"] = mcq.get("choices")
            session["correct_answer"] = mcq.get("correct_answer")
            session["current_topic"] = fallback_topic
            session["quiz_source_type"] = "prompt_topic"

            new_content = {
                "generated_text": generated_text,
                "question": mcq.get("question"),
                "choices": mcq.get("choices"),
                "correct_answer": mcq.get("correct_answer"),
                "topic": fallback_topic
            }

            return jsonify({
                "score": session["score"],
                "quiz_finished": False,
                "new_content": new_content,
                "use_custom_source": False
            })

    except Exception as e:
        print(f"Error generating new content: {str(e)}")
        return jsonify({"error": f"Failed to generate new content: {str(e)}"}), 500

@app.route('/start_quiz')
@login_required
def start_quiz():
    if "username" not in session:
        return redirect(url_for("home"))
    # Reset quiz session counters
    session["score"] = 0
    session["question_count"] = 0
    # Clear any existing question data
    session.pop("generated_question", None)
    session.pop("generated_choices", None)
    session.pop("correct_answer", None)
    session.pop("generated_text", None)
    session.pop("custom_quiz_active", None)
    session.pop("source_text", None)
    session.pop("current_topic", None)
    session.pop("quiz_source_type", None)
    session.pop("topic_quiz_active", None)
    session.pop("passage_history", None)  # Clear passage history for variety
    session.pop("full_document_text", None)  # Clear document text
    return redirect(url_for("quiz"))

@app.route("/start_group_exam", methods=["POST"])
@login_required
def start_group_exam():
    group_id = request.form.get("group_id")
    quiz_format = request.form.get("format")
    num_items = int(request.form.get("num_items", 5))
    timed = request.form.get("timed")
    time_limit = request.form.get("time_limit") if timed == "yes" else None
    host = session.get("username")

    # Get prompt or file
    if quiz_format == "prompt":
        prompt_text = request.form.get("prompt_text")
        source_type = "prompt"
        source_content = prompt_text
    elif quiz_format == "upload":
        file = request.files.get("upload_file")
        if file and allowed_file(file.filename):
            source_type = "upload"
            source_content = extract_text_from_uploaded_file(file)
        else:
            return render_template("group_exam.html", error="Invalid or missing file.")
    else:
        return render_template("group_exam.html", error="Invalid quiz format.")

    # Generate questions
    questions = []
    for i in range(num_items):
        if quiz_format == "prompt":
            passage = generate_varied_reading_comprehension_text(source_content, i+1)
        else:
            passage = select_relevant_excerpt_with_variety(source_content, [q["passage"] for q in questions])
        mcq = generate_multiple_choice_question_with_context(passage, i+1)
        questions.append({
            "passage": passage,
            "question": mcq["question"],
            "choices": json.dumps(mcq["choices"]),  # Store as JSON string
            "correct_answer": mcq["correct_answer"]
        })

    # Save group exam to DB
    db = connect_db()
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO group_exams (group_id, host_username, format, num_items, timed, time_limit, source_type, source_content)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (group_id, host, quiz_format, num_items, timed, time_limit, source_type, source_content))
    db.commit()

    # Save questions to DB
    for idx, q in enumerate(questions, start=1):
        cursor.execute("""
            INSERT INTO group_exam_questions (group_id, question_number, passage, question, choices, correct_answer)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (group_id, idx, q["passage"], q["question"], q["choices"], q["correct_answer"]))
    db.commit()

    db.commit()
    cursor.close()
    db.close()

    session["group_id"] = group_id
    return redirect(url_for("group_room"))

@app.route('/join_group_exam', methods=['POST'])
@login_required
def join_group_exam():
    group_id = request.form.get('group_id')
    username = session.get('username')

    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM group_exams WHERE group_id = %s", (group_id,))
    group = cursor.fetchone()

    if group:
        # Add user to group_exam_users if not already present
        cursor.execute("SELECT * FROM group_exam_users WHERE group_id = %s AND username = %s", (group_id, username))
        user_in_group = cursor.fetchone()
        if not user_in_group:
            cursor.execute("""
                INSERT INTO group_exam_users (group_id, username, finished)
                VALUES (%s, %s, %s)
            """, (group_id, username, False))
            db.commit()
        session['group_id'] = group_id
        cursor.close()
        db.close()
        return redirect(url_for('group_exam_take'))
    else:
        cursor.close()
        db.close()
        return render_template('homepage.html', error="Group ID not found.", first_name=session.get('first_name'), role=session.get('role'))

@app.route('/group_exam_take', methods=['GET', 'POST'])
@login_required
def group_exam_take():
    if "username" not in session or "group_id" not in session:
        return redirect(url_for("home"))

    group_id = session["group_id"]
    username = session["username"]

    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM group_exams WHERE group_id = %s", (group_id,))
    group = cursor.fetchone()
    if not group:
        cursor.close()
        db.close()
        return redirect(url_for("homepage"))

    num_items = group["num_items"]
    cursor.execute("SELECT * FROM group_exam_questions WHERE group_id = %s ORDER BY question_number ASC", (group_id,))
    questions = cursor.fetchall()

    # Track progress in session
    if "group_exam_score" not in session:
        session["group_exam_score"] = 0
    if "group_exam_question_count" not in session:
        session["group_exam_question_count"] = 0
    if "group_exam_current_index" not in session:
        session["group_exam_current_index"] = 0

    idx = session.get("group_exam_current_index", 0)

    if request.method == "POST":
        selected_answer = request.form.get("answer")
        correct_answer = questions[idx]["correct_answer"]

        if selected_answer == correct_answer:
            session["group_exam_score"] += 1

        session["group_exam_question_count"] += 1
        session["group_exam_current_index"] = idx + 1

        if session["group_exam_question_count"] >= num_items:
            # Save score to DB
            cursor.execute("""
                INSERT INTO group_exam_scores (group_id, username, score)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE score = %s, finished_at = NOW()
            """, (group_id, username, session["group_exam_score"], session["group_exam_score"]))
            # Mark user as finished
            cursor.execute("""
                UPDATE group_exam_users SET finished = TRUE WHERE group_id = %s AND username = %s
            """, (group_id, username))
            db.commit()

            # Fetch score from DB for display
            cursor.execute("SELECT score FROM group_exam_scores WHERE group_id = %s AND username = %s", (group_id, username))
            score_row = cursor.fetchone()
            score = score_row["score"] if score_row else session.get("group_exam_score", 0)

            # Clear user progress for next group exam
            session.pop("group_exam_score", None)
            session.pop("group_exam_question_count", None)
            session.pop("group_exam_current_index", None)

            cursor.close()
            db.close()
            return render_template(
                "result.html",
                total_questions=num_items,
                correct_answers=score,
                username=username
            )

        idx += 1
    else:
        # On GET, if not set, initialize
        if "remaining_time" not in session:
            session["remaining_time"] = session.get("time_limit", 0) * 60

    # Show current question
    if idx < len(questions):
        q = questions[idx]
        choices = json.loads(q["choices"])
        cursor.close()
        db.close()
        return render_template(
            "group_exam_take.html",
            question={"question_text": q["question"], "options": choices, "correct_answer": q["correct_answer"]},
            username=username,
            score=session.get("group_exam_score", 0),
            generated_text=q["passage"],
            group_id=group_id,
            time_limit=int(group.get("time_limit", 0)),  # <-- FIX: get from group, not session
            remaining_time=int(session.get("remaining_time", int(group.get("time_limit", 0)) * 60)),
            timed=group.get("timed", "no"),              # <-- FIX: get from group, not session
        )
    else:
        # If all questions answered, show result
        cursor.close()
        db.close()
        return render_template(
            "result.html",
            total_questions=num_items,
            correct_answers=session.get("group_exam_score", 0),
            username=username
        )

@app.route("/group_room")
@login_required
def group_room():
    if "username" not in session:
        return redirect(url_for("home"))
    group_id = session.get("group_id")

    db = connect_db()
    cursor = db.cursor(dictionary=True)

    # Get group info
    cursor.execute("SELECT * FROM group_exams WHERE group_id = %s", (group_id,))
    group = cursor.fetchone()
    if not group:
        cursor.close()
        db.close()
        return redirect(url_for("homepage"))

    # Get users (excluding host)
    cursor.execute("SELECT username FROM group_exam_users WHERE group_id = %s AND username != %s", (group_id, group["host_username"]))
    users = [row["username"] for row in cursor.fetchall()]

    # Get questions
    cursor.execute("SELECT * FROM group_exam_questions WHERE group_id = %s ORDER BY question_number ASC", (group_id,))
    questions = cursor.fetchall()
    for q in questions:
        q["choices"] = json.loads(q["choices"])

    # Get scores
    cursor.execute("SELECT username, score FROM group_exam_scores WHERE group_id = %s", (group_id,))
    scores = {row["username"]: row["score"] for row in cursor.fetchall()}

    cursor.close()
    db.close()

    return render_template(
        "group_room.html",
        group_id=group_id,
        users=users,
        questions=questions,
        scores=scores,
        host=group["host_username"],
        username=session["username"],
        time_limit=int(group.get("time_limit", 0)),   # <-- Add this
        timed=group.get("timed", "no"),               # <-- Add this
        remaining_time=int(session.get("remaining_time", int(group.get("time_limit", 0)) * 60)),  # Optional for timer
    )

app.config["GOOGLE_OAUTH_CLIENT_ID"] = os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
app.config["GOOGLE_OAUTH_CLIENT_SECRET"] = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET")

# Remove the duplicate app config lines and fix the blueprint
google_bp = make_google_blueprint(
    client_id=os.environ.get("GOOGLE_OAUTH_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET"),
    scope=["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"],
    redirect_to="oauth_success"  # Changed this - redirect to homepage after successful auth
)
app.register_blueprint(google_bp, url_prefix="/login")

# Remove the custom authorized route and use Flask-Dance's built-in handling
# Instead, create a handler that checks for successful OAuth
@app.route('/oauth_success')
def oauth_success():
    """Handle successful OAuth login"""
    if not google.authorized:
        return redirect(url_for("home"))

    try:
        # Get user info from Google
        resp = google.get("/oauth2/v2/userinfo")
        if not resp.ok:
            print(f"Failed to fetch user info: {resp.status_code}")
            return redirect(url_for("home"))

        user_info = resp.json()
        print(f"User info received: {user_info}")

        email = user_info.get("email")
        name = user_info.get("name", "")
        given_name = user_info.get("given_name", "")

        if not email:
            print("No email found in user info")
            return redirect(url_for("home"))

        # Database operations
        db = connect_db()
        cursor = db.cursor(dictionary=True)

        try:
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()

            if not user:
                # Create new user
                username = email.split("@")[0]
                cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
                if cursor.fetchone():
                    username = f"{username}_{random.randint(1000, 9999)}"

                cursor.execute("""
                    INSERT INTO users (username, email, role, first_name, password)
                    VALUES (%s, %s, %s, %s, %s)
                """, (username, email, "student", given_name or name, "oauth_google"))

                db.commit()
                session_username = username
                role = "student"
            else:
                session_username = user["username"]
                role = user["role"]

            # Set session
            session["username"] = session_username
            session["role"] = role
            session["guest"] = False
            session["score"] = 0
            session["question_count"] = 0
            session["user_id"] = user["user_id"] if user else cursor.lastrowid

            print(f"User logged in: {session_username}")

        except Exception as db_error:
            print(f"Database error: {db_error}")
            db.rollback()
            return redirect(url_for("home"))
        finally:
            cursor.close()
            db.close()

        return redirect(url_for("homepage"))

    except Exception as e:
        print(f"OAuth error: {e}")
        return redirect(url_for("home"))

# Add a simple Google login route
@app.route('/login_google')
def login_google():
    """Initiate Google login or handle if already authorized"""
    if google.authorized:
        return redirect(url_for("oauth_success"))
    return redirect(url_for("google.login"))

@app.route('/end_group_exam', methods=['POST'])
def end_group_exam():
    group_id = request.form.get('group_id')
    db = connect_db()
    cursor = db.cursor(dictionary=True)

    # Mark exam as ended
    cursor.execute("UPDATE group_exams SET exam_ended = %s WHERE group_id = %s", (True, group_id))

    # Get exam info
    cursor.execute("SELECT num_items FROM group_exams WHERE group_id = %s", (group_id,))
    group = cursor.fetchone()
    num_items = group["num_items"] if group else 0

    # Get all users in the group
    cursor.execute("SELECT username FROM group_exam_users WHERE group_id = %s", (group_id,))
    users = [row["username"] for row in cursor.fetchall()]

    for username in users:
        # Get current score and questions answered for this user
        cursor.execute("SELECT score FROM group_exam_scores WHERE group_id = %s AND username = %s", (group_id, username))
        score_row = cursor.fetchone()
        current_score = score_row["score"] if score_row else 0

        cursor.execute("SELECT finished FROM group_exam_users WHERE group_id = %s AND username = %s", (group_id, username))
        finished_row = cursor.fetchone()
        finished = finished_row["finished"] if finished_row else False

        if not finished:
            # Count how many questions answered (if you track this, otherwise assume all unanswered are wrong)
            cursor.execute("SELECT COUNT(*) AS answered FROM group_exam_scores WHERE group_id = %s AND username = %s", (group_id, username))
            _ = cursor.fetchone()  # <-- Always fetch the result, even if unused

            # Save score (unanswered = wrong, so score stays as is)
            cursor.execute("""
                INSERT INTO group_exam_scores (group_id, username, score)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE score = %s, finished_at = NOW()
            """, (group_id, username, current_score, current_score))

            # Mark user as finished
            cursor.execute("""
                UPDATE group_exam_users SET finished = TRUE WHERE group_id = %s AND username = %s
            """, (group_id, username))

    db.commit()
    cursor.close()
    db.close()
    return redirect(url_for('homepage'))

@app.route('/add_group_exam_time', methods=['POST'])
def add_group_exam_time():
    data = request.get_json()
    group_id = data.get('group_id')
    minutes = int(data.get('minutes', 0))
    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT time_limit FROM group_exams WHERE group_id = %s", (group_id,))
    group = cursor.fetchone()
    if not group:
        cursor.close()
        db.close()
        return jsonify(success=False, error="Group not found"), 404
    # Add minutes to current time_limit
    new_time_limit = int(group["time_limit"] or 0) + minutes
    cursor.execute("UPDATE group_exams SET time_limit = %s WHERE group_id = %s", (new_time_limit, group_id))
    db.commit()
    cursor.close()
    db.close()
    return jsonify(success=True, remaining_time=new_time_limit * 60)


@app.route('/records')
@login_required
def records():
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    
    # Get all exams with their questions
    cursor.execute("""
        SELECT 
            ge.group_id,
            ge.created_at,
            ge.source_content,
            ge.host_username,
            geq.question_number,
            geq.passage,
            geq.question,
            geq.choices,
            geq.correct_answer
        FROM group_exams ge
        LEFT JOIN group_exam_questions geq ON ge.group_id = geq.group_id
        ORDER BY ge.created_at DESC, geq.question_number ASC
    """)
    
    rows = cursor.fetchall()
    
    # Organize the data by group_id
    exams = {}
    for row in rows:
        group_id = row['group_id']
        if group_id not in exams:
            exams[group_id] = {
                'group_id': group_id,
                'created_at': row['created_at'],
                'source_content': row['source_content'],
                'host_username': row['host_username'],
                'questions': [],
                'examinees': []
            }
        
        if row['question'] is not None:  # Only add if there's a question
            exams[group_id]['questions'].append({
                'number': row['question_number'],
                'passage': row['passage'],
                'question': row['question'],
                'choices': json.loads(row['choices']) if row['choices'] else [],
                'correct_answer': row['correct_answer']
            })
    
    # Get scores for each exam - Modified query to remove user_id join
    for group_id in exams:
        cursor.execute("""
            SELECT username, score 
            FROM group_exam_scores 
            WHERE group_id = %s
        """, (group_id,))
        exams[group_id]['examinees'] = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return render_template('records.html', group_exams=list(exams.values()))

@app.route('/check_exam_ended')
def check_exam_ended():
    group_id = session.get("group_id")
    if not group_id:
        return jsonify({"error": "No group_id"}), 400
    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT exam_ended FROM group_exams WHERE group_id = %s", (group_id,))
    row = cursor.fetchone()
    cursor.close()
    db.close()
    if row:
        return jsonify({"exam_ended": bool(row["exam_ended"])})
    return jsonify({"exam_ended": False})

@app.route('/unselect_book', methods=['POST'])
@login_required
def unselect_book():
    data = request.get_json()
    book_id = data.get('book_id')
    
    if not book_id:
        return jsonify({"status": "error", "message": "No book ID provided"})
        
    conn = connect_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM selected_books WHERE id = %s AND user_id = %s", 
                      (book_id, session['user_id']))
        conn.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    finally:
        cursor.close()
        conn.close()

@app.route('/debug/session')
def debug_session():
    return {
        'session': dict(session),
        'google_authorized': google.authorized if 'google' in globals() else 'Not available'
    }

@app.route('/student_learning_plan', methods=['GET', 'POST'])
@login_required
def student_learning_plan():
    return(render_template("student_learning_plan.html"))

@app.route('/get_plan_progress/<int:plan_id>')
@login_required
def get_plan_progress(plan_id):
    user_id = session.get('user_id')
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    # Get topics for the plan
    cursor.execute("SELECT topic1, topic2 FROM learning_plans WHERE plan_id = %s", (plan_id,))
    plan = cursor.fetchone()
    topic1 = plan['topic1']
    topic2 = plan['topic2']

    # Get average score for quizzes matching topic1
    cursor.execute("""
        SELECT AVG(correct_answers / total_questions * 100) AS avg_score
        FROM user_progress
        WHERE user_id = %s AND topic = %s
    """, (user_id, topic1))
    topic1_avg = cursor.fetchone()['avg_score']
    topic1_avg = round(topic1_avg, 2) if topic1_avg is not None else None

    topic2_avg = None
    if topic2:
        cursor.execute("""
            SELECT AVG(score) as avg_score
            FROM custom_quizzes cq
            JOIN quiz_attempts qa ON cq.quiz_id = qa.quiz_id
            WHERE qa.user_id = %s AND cq.source_content = %s
        """, (user_id, topic2))
        topic2_avg = cursor.fetchone()['avg_score']
        topic2_avg = round(topic2_avg, 2) if topic2_avg is not None else None

    cursor.close()
    conn.close()
    return jsonify({'topic1_avg': topic1_avg, 'topic2_avg': topic2_avg})

@app.route('/search_learning_plans')
@login_required
def search_learning_plans():
    query = request.args.get('q', '').strip()
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    # Join with users to get teacher name
    cursor.execute("""
        SELECT lp.plan_id, lp.plan_title, lp.topic1, lp.topic2, lp.file1_path, lp.file2_path,
               u.username AS teacher_name
        FROM learning_plans lp
        JOIN users u ON lp.teacher_id = u.user_id
        WHERE lp.plan_title LIKE %s OR u.username LIKE %s
        ORDER BY lp.created_at DESC
    """, (f"%{query}%", f"%{query}%"))
    plans = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(plans)

@app.route('/get_applied_learning_plans')
@login_required
def get_applied_learning_plans():
    user_id = session.get('user_id')
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT lp.plan_id, lp.plan_title, lp.topic1, lp.topic2, lp.file1_path, lp.file2_path,
               u.username AS teacher_name, lpa.status
        FROM learning_plan_applications lpa
        JOIN learning_plans lp ON lpa.plan_id = lp.plan_id
        JOIN users u ON lp.teacher_id = u.user_id
        WHERE lpa.student_id = %s
        ORDER BY lpa.applied_at DESC
    """, (user_id,))
    plans = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(plans)

@app.route('/auto_quiz_setup', methods=['POST'])
@login_required
def auto_quiz_setup():
    data = request.get_json()
    topic = data.get('topic')
    file_path = data.get('file_path')
    user_id = session.get('user_id')

    # Load the PDF, extract text
    import PyPDF2
    pdf_path = os.path.join(app.root_path, 'static', file_path)
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        return jsonify({"status": "error", "message": "Could not read PDF."})

    # Auto-generate quiz questions
    num_items = 5  # or any number you want
    questions = []
    for i in range(num_items):
        passage = select_relevant_excerpt_with_variety(text, [q["passage"] for q in questions])
        mcq = generate_multiple_choice_question_with_context(passage, i+1)
        questions.append({
            "passage": passage,
            "question": mcq["question"],
            "choices": mcq["choices"],
            "correct_answer": mcq["correct_answer"]
        })

    # Save quiz to DB
    db = connect_db()
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO custom_quizzes (creator_username, format, num_items, timed, time_limit, source_type, source_content)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        session["username"],
        "upload",
        num_items,
        "no",
        0,
        "upload",
        topic
    ))
    quiz_id = cursor.lastrowid

    # Insert each question
    for idx, q in enumerate(questions, start=1):
        cursor.execute("""
            INSERT INTO custom_quiz_questions (quiz_id, question_number, passage, question, choices, correct_answer)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            quiz_id,
            idx,
            q["passage"],
            q["question"],
            json.dumps(q["choices"]),
            q["correct_answer"]
        ))

    db.commit()
    cursor.close()
    db.close()

    # Set up session for quiz
    session["quiz_id"] = quiz_id
    session["num_items"] = num_items
    session["score"] = 0
    session["question_count"] = 0
    session["current_question_index"] = 0
    session["quiz_topic"] = topic

    # Redirect directly to quiz
    return jsonify({"status": "success", "quiz_url": url_for('quiz')})

@app.route('/apply_learning_plan', methods=['POST'])
@login_required
def apply_learning_plan():
    data = request.get_json()
    plan_id = data.get('plan_id')
    student_id = session.get('user_id')
    if not plan_id or not student_id:
        return jsonify({"status": "error", "message": "Missing plan or user"}), 400

    conn = connect_db()
    cursor = conn.cursor()
    # Prevent duplicate applications
    cursor.execute("""
        SELECT * FROM learning_plan_applications WHERE plan_id = %s AND student_id = %s
    """, (plan_id, student_id))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        return jsonify({"status": "error", "message": "Already applied"}), 409

    cursor.execute("""
        INSERT INTO learning_plan_applications (plan_id, student_id) VALUES (%s, %s)
    """, (plan_id, student_id))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"status": "success"})

@app.route('/teacher_learning_plan', methods=['GET', 'POST'])
@login_required
def teacher_learning_plan():
    if session.get("role") not in ["teacher", "admin"]:
        return redirect(url_for("home"))

    message = None
    if request.method == "POST":
        plan_title = request.form.get("plan_title")
        topic1 = request.form.get("topic1")
        topic2 = request.form.get("topic2")
        file1 = request.files.get("file1")
        file2 = request.files.get("file2")

        # Save files and plan info
        uploads = []
        # Ensure static/uploads exists
        upload_folder = os.path.join(app.root_path, "static", "uploads")
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        for idx, file in enumerate([file1, file2], start=1):
            if file and allowed_file(file.filename):
                filename = secure_filename(f"{session['username']}_plan_{plan_title}_topic{idx}_{file.filename}")
                filepath = os.path.join(upload_folder, filename)
                file.save(filepath)
                # Save only the relative path for use with url_for('static', ...)
                uploads.append(f"uploads/{filename}")
            else:
                uploads.append(None)

        # Save to DB (create table learning_plans if needed)
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO learning_plans (teacher_id, plan_title, topic1, topic2, file1_path, file2_path, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
        """, (session["user_id"], plan_title, topic1, topic2, uploads[0], uploads[1]))
        conn.commit()
        cursor.close()
        conn.close()
        message = "Learning plan created successfully!"

    # Always fetch plans for the current teacher/admin
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT lp.*, 
            (SELECT GROUP_CONCAT(CONCAT(u.username, ':', lpa.application_id, ':', lpa.status) SEPARATOR ';')
            FROM learning_plan_applications lpa
            JOIN users u ON lpa.student_id = u.user_id
            WHERE lpa.plan_id = lp.plan_id) AS applicants
        FROM learning_plans lp
        WHERE lp.teacher_id = %s
        ORDER BY lp.created_at DESC
    """, (session["user_id"],))
    plans = cursor.fetchall()
    cursor.close()
    conn.close()

    # Parse applicants for easier use in template
    for plan in plans:
        plan["applicants_list"] = []
        if plan.get("applicants"):
            for entry in plan["applicants"].split(";"):
                if entry:
                    username, app_id, status = entry.split(":")
                    plan["applicants_list"].append({
                        "username": username,
                        "application_id": app_id,
                        "status": status
                    })

    return render_template("teacher_learning_plan.html", message=message, plans=plans)

@app.route('/update_applicant_status', methods=['POST'])
@login_required
def update_applicant_status():
    data = request.get_json()
    application_id = data.get('application_id')
    status = data.get('status')
    if not application_id or status not in ['accepted', 'denied']:
        return jsonify({"status": "error", "message": "Invalid request"}), 400

    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE learning_plan_applications SET status = %s WHERE application_id = %s
    """, (status, application_id))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"status": "success"})

@app.route('/plan_members/<int:plan_id>')
@login_required
def plan_members(plan_id):
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    # Get accepted members for this plan
    cursor.execute("""
        SELECT lpa.student_id AS user_id, u.username
        FROM learning_plan_applications lpa
        JOIN users u ON lpa.student_id = u.user_id
        WHERE lpa.plan_id = %s AND lpa.status = 'accepted'
    """, (plan_id,))
    members = cursor.fetchall()

    # Get plan topics
    cursor.execute("SELECT topic1, topic2 FROM learning_plans WHERE plan_id = %s", (plan_id,))
    plan = cursor.fetchone()
    topics = [plan['topic1']]
    if plan['topic2']:
        topics.append(plan['topic2'])

    # For each member, get progress and average scores
    for member in members:
        member['progress'] = {}
        member['average_scores'] = {}
        for topic in topics:
            # Progress: count of quizzes taken for this topic
            cursor.execute("""
                SELECT COUNT(*) as quiz_count
                FROM user_progress
                WHERE user_id = %s AND LOWER(topic) = %s
            """, (member['user_id'], topic.lower()))
            member['progress'][topic] = cursor.fetchone()['quiz_count']

            # Average score for this topic
            cursor.execute("""
                SELECT AVG(correct_answers / total_questions) * 100 as avg_score
                FROM user_progress
                WHERE user_id = %s AND LOWER(topic) = %s AND total_questions > 0
            """, (member['user_id'], topic.lower()))
            avg = cursor.fetchone()['avg_score']
            member['average_scores'][topic] = round(avg, 2) if avg else None

        # Get quizzes for this member and plan topics
        format_strings = ','.join(['%s'] * len(topics))
        query = f"""
            SELECT quiz_id, topic, correct_answers, total_questions, last_updated
            FROM user_progress
            WHERE user_id = %s AND LOWER(topic) IN ({format_strings})
            ORDER BY last_updated DESC
        """
        params = [member['user_id']] + [t.lower() for t in topics]
        cursor.execute(query, params)
        member['quizzes'] = cursor.fetchall()

    cursor.close()
    conn.close()
    return jsonify(members)

@app.route('/get_member_quizzes/<int:user_id>')
@login_required
def get_member_quizzes(user_id):
    plan_id = request.args.get('plan_id', type=int)
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    # If plan_id is provided, filter by plan topics
    if plan_id:
        cursor.execute("SELECT topic1, topic2 FROM learning_plans WHERE plan_id = %s", (plan_id,))
        plan = cursor.fetchone()
        topics = [plan['topic1']]
        if plan['topic2']:
            topics.append(plan['topic2'])
        # Prepare SQL for filtering by topics (case-insensitive)
        format_strings = ','.join(['%s'] * len(topics))
        query = f"""
            SELECT quiz_id, topic, correct_answers, total_questions, last_updated
            FROM user_progress
            WHERE user_id = %s AND LOWER(topic) IN ({format_strings})
            ORDER BY last_updated DESC
        """
        params = [user_id] + [t.lower() for t in topics]
        cursor.execute(query, params)
    else:
        # No plan_id, return all quizzes for user
        cursor.execute("""
            SELECT quiz_id, topic, correct_answers, total_questions, last_updated
            FROM user_progress
            WHERE user_id = %s
            ORDER BY last_updated DESC
        """, (user_id,))
    quizzes = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(quizzes) 

if __name__ == "__main__":
    app.run(debug=False,  ssl_context = 'adhoc')
