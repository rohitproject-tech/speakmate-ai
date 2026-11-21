Render link: https://speakmate-ai-6uak.onrender.com

# SpeakMate AI — Real-Time Conversational English Partner & Shadow Coach

SpeakMate AI is a low-latency, voice-to-voice English speaking coach designed to help learners overcome speaking hesitation through real-time conversation, instant grammar and phrasing refinements, pronunciation shadowing, and fluency analytics.

---

## Key Features

- **Low-Latency Voice Loop:** Sub-second Speech-to-Text (`whisper-large-v3-turbo`), structured dual-role LLM coaching, and natural neural Text-to-Speech (`edge-tts`) supporting Indian, US, and UK English voices.
- **Shadow Coach Feedback:** Evaluates every spoken turn in parallel—suggesting natural phrasing alternatives, vocabulary upgrades, and concise grammar tips without interrupting the flow of conversation.
- **Roleplay Scenarios:** Practice across **Daily Casual Chat**, **HR Mock Interviews**, **Tech / Project Pitches**, and **Debate & Opinions**.
- **"Help Me Answer" Scaffolded Hints:** Generates two ready-to-speak starter sentences (Simple & Expressive) whenever you freeze mid-conversation.
- **Word-Level Clarity & Shadowing:** Flags low-confidence words from acoustic transcription and provides slow-paced audio playback (` Shadow`) to practice accurate pronunciation.
- **Hands-Free Auto-Send:** Uses the Web Audio API to detect speech pauses (1.6s) and automatically send turns for a natural, call-like conversation flow.
- **Session Fluency Report Card:** Tracks Words Per Minute (WPM), filler words, and recurring patterns in SQLite (`speakmate.db`) with end-of-session diagnostic scorecards.

---

## System Architecture

```
[Browser Mic (Web Audio API)]
       │  (Audio Blob + Silence Detection)
       ▼
[FastAPI Backend (/api/turn)]
       ├──► [Groq Whisper-large-v3-turbo] ──► Transcript + Word Confidence + WPM/Fillers
       ├──► [Groq LLM (JSON Mode)]        ──► Conversational Reply + Shadow Coach Tips
       ├──► [Microsoft Edge-TTS]          ──► Base64 Neural Audio Response
       └──► [SQLite (speakmate.db)]       ──► Persistent Session & Progress Tracking
```
---
## Tech Stack
    Backend: Python 3.12, FastAPI, Uvicorn, SQLite
    AI & Speech: Groq API (whisper-large-v3-turbo & LLM Structured JSON), edge-tts
    Frontend: HTML5, Tailwind CSS, Web Audio API (MediaRecorder + AudioContext)
---
 ## Local Setup & Installation
 
    Clone the repository:
    Bash
    git clone [https://github.com/rohitproject-tech/speakmate-ai.git](https://github.com/rohitproject-tech/speakmate-ai.git)
    cd speakmate-ai

    Create and activate a virtual environment:
    Inside The Folder speakmate-ai
    Bash
    python -m venv venv
    
    # Windows:
    venv\Scripts\activate
    # macOS / Linux:
    source venv/bin/activate

    Install dependencies:
    Bash
    pip install -r requirements.txt

    Set up environment variables:
    Create a .env file in the root directory and add your Groq API key:
    Code snippet

    GROQ_API_KEY=your_groq_api_key_here

    Run the application:
    Bash
    uvicorn backend:app --reload

    Open http://127.0.0.1:8000 in Chrome or Edge, allow microphone permissions, and click New Session.
---
## 📁 Project Structure
```text

speakmate-ai/
├── backend.py          # FastAPI server (STT, dual-role LLM coach, TTS, SQLite analytics)
├── main.py             # Phase 1 standalone CLI voice-loop prototype
├── static/
│   └── index.html      # Split-screen web UI with Web Audio VAD & Report Card modal
├── requirements.txt    # Python dependencies
├── .gitignore          # Excludes venv, .env, and local SQLite database
└── README.md           # Project documentation
