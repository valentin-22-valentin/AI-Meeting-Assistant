from faster_whisper import WhisperModel

model_size = "base"

print("Loading model...")
model = WhisperModel(model_size, device="cpu", compute_type="int8")
print("Model loaded successfully!")

audio_file = "test.mp3" 

print(f"Processing audio file: {audio_file}")
segments, info = model.transcribe(audio_file, beam_size=5)

print(f"Detected: {info.language} language, with {info.language_probability:.2f} probability.")
print("-" * 50)


for segment in segments:
    print(f"[{segment.start:.2f}sec -> {segment.end:.2f}sec] {segment.text}")