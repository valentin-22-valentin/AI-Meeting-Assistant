import os
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from huggingface_hub import login
from faster_whisper.audio import decode_audio 

# Hugging Face API token 
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    print("Error: HF_TOKEN not found in the .env file!")
    exit()


print("Logging in to Hugging Face...")
login(token=hf_token)

print("Loading PyAnnote model...")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1"
)

audio_file = "test.mp3"
print(f"Decoding audio file with the faster-whisper engine: {audio_file}...")


audio_array = decode_audio(audio_file, sampling_rate=16000)

waveform = torch.from_numpy(audio_array).unsqueeze(0)

print("Running diarization... (This may take 1-2 minutes on CPU)")
diarization = pipeline({"waveform": waveform, "sample_rate": 16000})

print("\n--- SPEAKER TIMESTAMPS ---")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"Speaker: {speaker} | Time: {turn.start:.1f}s - {turn.end:.1f}s")
