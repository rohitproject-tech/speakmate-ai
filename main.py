import os
import sys
import json
import asyncio
import tempfile
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv
from groq import Groq
import edge_tts
import pygame

# Fix Windows cp1252 console encoding crash
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Voice options: "en-IN-NeerjaNeural" (Indian English), "en-US-AriaNeural" (US English)
VOICE = "en-IN-NeerjaNeural"
SAMPLE_RATE = 16000

SYSTEM_PROMPT = """
You are an encouraging, friendly English speaking partner and real-time language coach.
Your job is to do TWO things on every turn:
1. Keep a natural, engaging spoken conversation going with the user. Keep your spoken replies concise (2-3 sentences max) and always end with an open-ended question to keep them talking.
2. Silently check the user's input for grammar mistakes, awkward phrasing, or Indian-English literal translations, and suggest a smoother, natural phrasing.

If the user says "help me phrase" or asks how to say something, give them 2 natural ways to express their thought, then smoothly continue the conversation.

You MUST respond strictly in valid JSON with these exact keys:
{
  "reply": "Your natural conversational response to be spoken aloud.",
  "correction": "A brief, polite rephrasing of their sentence if needed, or 'Sounds natural!' if their sentence was good.",
  "tip": "One short 1-line explanation of why the correction sounds better (or empty string if no correction needed)."
}
"""

conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]


def record_audio_until_enter() -> str:
    """Records microphone audio until the user presses Enter."""
    print("\n[MIC] Listening... Speak now! (Press ENTER when you finish speaking)")
    recording = []

    def callback(indata, frames, time, status):
        recording.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback):
        input()  # Waits for Enter key

    audio_data = np.concatenate(recording, axis=0)
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_wav.name, SAMPLE_RATE, audio_data)
    return temp_wav.name


def transcribe_audio(wav_path: str) -> str:
    """Transcribes speech to text using Groq Whisper."""
    with open(wav_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=audio_file,
            language="en",
            prompt="Um, uh, like, you know, I want to improve my English speaking."
        )
    os.remove(wav_path)
    return transcription.text.strip()


def get_ai_response(user_text: str) -> dict:
    """Gets conversational reply + grammar coaching in JSON format."""
    conversation_history.append({"role": "user", "content": user_text})

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=conversation_history,
        response_format={"type": "json_object"},
        temperature=0.7,
    )

    raw_content = completion.choices[0].message.content
    parsed = json.loads(raw_content)

    # Keep only the spoken reply in history so the conversation context stays clean
    conversation_history.append({"role": "assistant", "content": parsed["reply"]})
    return parsed


async def speak_text(text: str):
    """Converts text to speech using Edge-TTS and plays it."""
    temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    communicate = edge_tts.Communicate(text, VOICE, rate="+0%")
    await communicate.save(temp_mp3)

    pygame.mixer.init()
    pygame.mixer.music.load(temp_mp3)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.1)
    pygame.mixer.quit()
    os.remove(temp_mp3)


async def main():
    print("=" * 55)
    print("SpeakMate AI - Phase 1 Prototype")
    print("Press ENTER to start/stop speaking. Ctrl+C to exit.")
    print("=" * 55)

    opening = "Hey! I'm your English speaking partner. What did you do today, or what topic would you like to practice?"
    print(f"\n[AI]: {opening}")
    await speak_text(opening)

    while True:
        input("\n>> Press ENTER to start talking...")
        wav_path = record_audio_until_enter()

        print("[...] Transcribing...")
        user_text = transcribe_audio(wav_path)

        if not user_text:
            print("[!] Didn't catch anything. Try speaking a bit louder.")
            continue

        print(f"\n[YOU]: \"{user_text}\"")

        ai_data = get_ai_response(user_text)

        # Display Shadow Coach feedback
        print("-" * 55)
        print(f"[COACH SUGGESTION]: {ai_data['correction']}")
        if ai_data.get("tip"):
            print(f"Why: {ai_data['tip']}")
        print("-" * 55)

        # Speak the conversational response
        print(f"[AI]: {ai_data['reply']}")
        await speak_text(ai_data["reply"])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession ended. Great practice today!")