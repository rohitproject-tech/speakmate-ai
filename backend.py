import os
import re
import json
import base64
import sqlite3
import tempfile
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from groq import Groq
import edge_tts

load_dotenv()

app = FastAPI(title="SpeakMate AI")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DB_PATH = "speakmate.db"
sessions: dict[str, list[dict]] = {}


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            scenario TEXT,
            user_text TEXT,
            correction TEXT,
            wpm INTEGER,
            filler_count INTEGER,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()

SCENARIO_PROMPTS = {
    "casual": (
        "You are a friendly, warm English speaking partner having a casual daily conversation "
        "(hobbies, tech, college life, movies, travel). Keep replies natural, warm, and 2-3 sentences max. "
        "Always end with an engaging follow-up question."
    ),
    "hr_interview": (
        "You are an encouraging HR interviewer conducting a behavioral mock interview for a software engineering role. "
        "Ask one realistic interview question at a time (strengths, teamwork, conflict, career goals), give brief encouraging feedback, "
        "and guide the candidate to use the STAR method."
    ),
    "tech_pitch": (
        "You are a hackathon judge and senior engineer listening to a developer explain their software project. "
        "Ask sharp, constructive questions about architecture, problem-solving, and impact, helping them articulate technical ideas clearly."
    ),
    "debate": (
        "You are a thoughtful debate partner. Gently take the opposite viewpoint on everyday or tech topics "
        "(e.g., AI vs human creativity, remote work vs office) to push the user to express opinions, give reasons, and defend their stance."
    ),
}

BASE_INSTRUCTIONS = """
Your job is to do TWO things on every turn:
1. Respond naturally in character to keep the spoken conversation flowing (2-3 sentences max, ending with a question).
2. Act as a Shadow Coach: inspect the user's transcript for grammar slips, Indian-English literal phrasing, or weak vocabulary, and provide a smoother alternative.

If the user says "help me phrase" or asks how to express an idea, give 2 natural ways to say it in "correction", then smoothly continue the conversation in "reply".

Respond strictly in valid JSON with these exact keys:
{
  "reply": "Your conversational response to be spoken aloud.",
  "correction": "Natural rephrasing of the user's sentence, or 'Sounds natural!' if already good.",
  "tip": "Short 1-line explanation of the improvement, or empty string if none needed.",
  "vocabulary_upgrade": "One stronger word/phrase they could use next time (e.g., 'Instead of very good -> impressive'), or empty string."
}
"""

FILLER_WORDS = {"um", "uh", "like", "basically", "actually", "literally", "you know", "sort of", "kind of"}


def analyze_fluency(transcript: str, duration_sec: float) -> dict:
    words = re.findall(r"\b\w+\b", transcript.lower())
    word_count = len(words)
    wpm = round((word_count / max(duration_sec, 1.0)) * 60)

    found_fillers = []
    total_fillers = 0
    text_lower = transcript.lower()
    for filler in FILLER_WORDS:
        count = len(re.findall(r"\b" + re.escape(filler) + r"\b", text_lower))
        if count > 0:
            found_fillers.append(f"{filler} ({count}x)")
            total_fillers += count

    return {
        "word_count": word_count,
        "wpm": wpm,
        "fillers": found_fillers,
        "total_fillers": total_fillers,
    }


async def generate_tts_base64(text: str, voice: str, rate: str = "+0%") -> str:
    temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(temp_mp3)
    with open(temp_mp3, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    os.remove(temp_mp3)
    return audio_b64


@app.post("/api/start")
async def start_session(
    session_id: str = Form(...),
    scenario: str = Form("casual"),
    voice: str = Form("en-IN-NeerjaNeural"),
):
    system_msg = f"{SCENARIO_PROMPTS.get(scenario, SCENARIO_PROMPTS['casual'])}\n\n{BASE_INSTRUCTIONS}"
    sessions[session_id] = [{"role": "system", "content": system_msg}]

    openings = {
        "casual": "Hey there! Great to chat with you. How has your day been going so far?",
        "hr_interview": "Welcome! Thanks for joining today. To kick things off, could you tell me a little about yourself and what excites you about software engineering?",
        "tech_pitch": "Hi! I'd love to hear about what you're building. In a couple of sentences, what problem does your project solve?",
        "debate": "Let's warm up with a fun topic: do you think AI tools make developers better thinkers, or do they make us rely less on fundamentals?",
    }
    opening_text = openings.get(scenario, openings["casual"])
    sessions[session_id].append({"role": "assistant", "content": opening_text})

    audio_b64 = await generate_tts_base64(opening_text, voice)
    return {"reply": opening_text, "audio_base64": audio_b64}


@app.post("/api/turn")
async def process_turn(
    audio: UploadFile = File(...),
    session_id: str = Form(...),
    scenario: str = Form("casual"),
    voice: str = Form("en-IN-NeerjaNeural"),
    duration_sec: float = Form(3.0),
):
    if session_id not in sessions:
        system_msg = f"{SCENARIO_PROMPTS.get(scenario, SCENARIO_PROMPTS['casual'])}\n\n{BASE_INSTRUCTIONS}"
        sessions[session_id] = [{"role": "system", "content": system_msg}]

    suffix = os.path.splitext(audio.filename or "recording.webm")[1] or ".webm"
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_audio.write(await audio.read())
    temp_audio.close()

    unclear_words = []
    try:
        with open(temp_audio.name, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=f,
                language="en",
                response_format="verbose_json",
                timestamp_granularities=["word"],
                prompt="Um, uh, like, you know, I am practicing spoken English.",
            )
        user_text = transcription.text.strip()

        # Extract words where Whisper had lower acoustic confidence
        raw_words = getattr(transcription, "words", None) or []
        for w in raw_words:
            word_str = w.get("word", "").strip(".,!?\"'") if isinstance(w, dict) else getattr(w, "word", "").strip(".,!?\"'")
            prob = w.get("probability", 1.0) if isinstance(w, dict) else getattr(w, "probability", 1.0)
            if word_str and len(word_str) > 2 and prob < 0.65 and word_str.lower() not in unclear_words:
                unclear_words.append(word_str)
    finally:
        os.remove(temp_audio.name)

    if not user_text:
        return {"error": "No speech detected. Try speaking a bit louder."}

    fluency = analyze_fluency(user_text, duration_sec)
    sessions[session_id].append({"role": "user", "content": user_text})

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=sessions[session_id],
        response_format={"type": "json_object"},
        temperature=0.7,
    )

    parsed = json.loads(completion.choices[0].message.content)
    sessions[session_id].append({"role": "assistant", "content": parsed["reply"]})

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO turns (session_id, scenario, user_text, correction, wpm, filler_count, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            session_id,
            scenario,
            user_text,
            parsed.get("correction", "Sounds natural!"),
            fluency["wpm"],
            fluency["total_fillers"],
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()

    audio_b64 = await generate_tts_base64(parsed["reply"], voice)

    return {
        "user_text": user_text,
        "reply": parsed["reply"],
        "correction": parsed.get("correction", "Sounds natural!"),
        "tip": parsed.get("tip", ""),
        "vocabulary_upgrade": parsed.get("vocabulary_upgrade", ""),
        "unclear_words": unclear_words[:4],
        "fluency": fluency,
        "audio_base64": audio_b64,
    }


@app.post("/api/pronounce")
async def pronounce_phrase(
    text: str = Form(...),
    voice: str = Form("en-IN-NeerjaNeural"),
):
    """Plays any word or corrected sentence slowly and clearly so the user can shadow it."""
    audio_b64 = await generate_tts_base64(text, voice, rate="-12%")
    return {"audio_base64": audio_b64}


@app.post("/api/hint")
async def get_hint(session_id: str = Form(...)):
    history = sessions.get(session_id, [])
    last_ai_msg = next(
        (m["content"] for m in reversed(history) if m["role"] == "assistant"),
        "Tell me about your day.",
    )

    prompt = f"""
The English learner was just asked this question by their conversation partner:
"{last_ai_msg}"

Provide 2 natural, spoken English ways they can answer this question:
1. "simple": An easy, confident 1-sentence answer.
2. "advanced": A slightly richer, more expressive 2-sentence answer.

Respond strictly in valid JSON:
{{
  "simple": "...",
  "advanced": "..."
}}
"""
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.6,
    )
    return json.loads(completion.choices[0].message.content)


@app.post("/api/report")
async def session_report(session_id: str = Form(...)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT user_text, correction, wpm, filler_count FROM turns WHERE session_id = ?",
        (session_id,),
    )
    rows = c.fetchall()

    c.execute("SELECT AVG(wpm), COUNT(*) FROM turns")
    overall_avg_wpm, total_all_turns = c.fetchone()
    conn.close()

    if not rows:
        return {"error": "Speak at least one turn before generating a report!"}

    avg_wpm = round(sum(r[2] for r in rows) / len(rows))
    total_fillers = sum(r[3] for r in rows)
    transcript_summary = "\n".join(
        [f"- Said: '{r[0]}' | Coach: '{r[1]}'" for r in rows]
    )

    prompt = f"""
Analyze this English learner's speaking session ({len(rows)} turns, average {avg_wpm} WPM, {total_fillers} filler words):

{transcript_summary}

Respond strictly in valid JSON with:
{{
  "fluency_score": 82,
  "strength": "One specific thing they did well in this session.",
  "main_pattern_to_fix": "The #1 recurring grammar or phrasing habit noticed across their sentences.",
  "tomorrow_goal": "One actionable 5-minute speaking tip for their next session."
}}
"""
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.5,
    )
    report = json.loads(completion.choices[0].message.content)
    report["session_turns"] = len(rows)
    report["avg_wpm"] = avg_wpm
    report["total_fillers"] = total_fillers
    report["lifetime_turns"] = total_all_turns or len(rows)
    report["lifetime_avg_wpm"] = round(overall_avg_wpm or avg_wpm)
    return report


os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")