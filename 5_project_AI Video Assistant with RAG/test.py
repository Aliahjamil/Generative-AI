from dotenv import load_dotenv
load_dotenv()
from utils.audio_processor import process_input
from core.transcriber import transcribe_all


source = "https://www.youtube.com/watch?v=0e.nezwg&t=223"
language = "hinglish" # change to "hinglish" to test Sarvam

chunks = process_input(source)
transcript = transcribe_all(chunks, language=language)
print("\n=== TRANSCRIPT ===\n")
print(transcript)