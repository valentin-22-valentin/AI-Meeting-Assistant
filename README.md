#  AI Meeting Assistant

A hybrid AI pipeline that transcribes meeting audio and automatically extracts structured action items (attendees, decisions, tasks). Built with Python, it utilizes local processing for audio and cloud-based LLMs for advanced extraction.

##  Features
* **Speech-to-Text**: Converts audio to text using the lightweight `faster-whisper` model.
* **Information Extraction**: Uses the Llama-3.3-70B model via the Groq API to strictly extract structured JSON data.
* **Speaker Diarization (WIP)**: Integrates `pyannote.audio` to identify speaker turns, utilizing a custom workaround via Whisper's audio decoder to bypass Windows FFmpeg dependency issues.
* **Multilingual**: Supports generating output in both English and Hungarian.
* **User Interface**: Clean and simple web UI built with Streamlit.

##  Architecture & Tech Stack
This project follows a hybrid approach, balancing local privacy with cloud processing power:
* **Backend & Logic**: Python 3.x, Pydantic (Data validation)
* **Local Inference (ASR & Diarization)**: Faster-Whisper, PyAnnote.audio, PyTorch
* **Cloud Inference (LLM)**: Groq API 
* **Frontend**: Streamlit
* **Security**: python-dotenv

##  How to Run Locally

1. **Clone the repository**
2. **Create a virtual environment**: `python -m venv venv`
3. **Activate the environment**: 
   * Windows: `.\venv\Scripts\activate`
   * Mac/Linux: `source venv/bin/activate`
4. **Install dependencies**: `pip install -r requirements.txt`
5. **Configure Environment Variables**: Create a `.env` file in the root directory and add your API keys (Never commit this file!):
   ```env
   GROQ_API_KEY=your_groq_api_key
   HF_TOKEN=your_huggingface_read_token
   ```
6. **Hugging Face Setup**: Ensure you have accepted the user agreements for `pyannote/segmentation-3.0` and `pyannote/speaker-diarization-3.1` on Hugging Face using your token's account.
7. **Run the application**: `streamlit run app.py`
   * *To test the standalone diarization module: `python diarize_test.py`*

##  Roadmap / Current Status
This project is an active work-in-progress portfolio piece.
- [x] Basic ASR pipeline implementation
- [x] LLM integration with strict JSON output (Pydantic)
- [x] Streamlit UI and multilingual support
- [x] Local Speaker Diarization module setup
- [ ] **Next Step**: Align PyAnnote timestamps with Whisper transcription to provide fully speaker-labeled inputs to the LLM.
- [ ] Implement professional error handling for external API timeouts.
- [ ] Add automated unit testing for the extraction logic.