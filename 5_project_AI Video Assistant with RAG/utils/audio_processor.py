
import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# STEP 1: Download YouTube audio
def download_youtube_audio(url: str) -> str:

    output_path = os.path.join(
        DOWNLOAD_DIR, "%(title)s.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        filename = ydl.prepare_filename(info)
        filename = os.path.splitext(filename)[0] + ".wav"

    return filename


# STEP 2: Convert audio to WAV (Mono, 16kHz)
def convert_to_wav(input_path: str) -> str:

    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    audio = AudioSegment.from_file(input_path)

    audio = audio.set_channels(1).set_frame_rate(16000)

    audio.export(output_path, format="wav")

    return output_path


# STEP 3: Split audio into chunks
def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:

    audio = AudioSegment.from_wav(wav_path)

    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):

        chunk = audio[start:start + chunk_ms]

        chunk_path = f"{os.path.splitext(wav_path)[0]}_chunk_{i}.wav"

        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)

    return chunks

def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks

#I have developed an audio preprocessing pipeline that accepts audio input from either YouTube URLs or local audio files. For YouTube videos, I used **yt-dlp** to download the audio and **FFmpeg** to convert it into WAV format. For local audio files, I used **Pydub** to convert them into WAV format, mono (1 channel), and a 16 kHz sampling rate. Finally, I implemented an audio chunking mechanism that splits long recordings into smaller 10-minute WAV segments, making them easier to process using Whisper for speech-to-text transcription in my AI Video Assistant with RAG project.