import streamlit as st
import json
import tempfile
import os
from faster_whisper import WhisperModel
from groq import Groq
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

#groq api key 
load_dotenv() 
api_key = os.getenv("GROQ_API_KEY")

class Task(BaseModel):
    assignee: str
    action: str
    deadline: str

class MeetingSummary(BaseModel):
    attendees: List[str]
    decisions: List[str]
    tasks: List[Task]

#  Web configration
st.set_page_config(page_title="AI Meeting Assistant", layout="wide")
st.title(" AI Meeting Assistant")

with st.sidebar:
    st.header(" Settings")
    output_language = st.selectbox("Output language", [ "English"])

# primary logic
if not api_key:

    st.warning("Error: GROQ_API_KEY not found! Please check your .env file or cloud settings.")
else:
    
    uploaded_file = st.file_uploader("Upload an audio file (mp3, wav)", type=['mp3', 'wav'])

    if uploaded_file is not None:
        if st.button("Start Processing"):
            
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:

                with st.spinner("1/2: Transcribing audio... (This may take a minute)"):
                    model = WhisperModel("base", device="cpu", compute_type="int8")
                    segments, info = model.transcribe(tmp_file_path, beam_size=5)
                    
                    transcript = " ".join([segment.text for segment in segments])
                
                st.success("Speech successfully transcribed!")
                with st.expander(" View raw transcript"):
                    st.write(transcript)

                # LLM calling to extract meeting data
                with st.spinner(f"2/2: Extracting data in {output_language}..."):
                    client = Groq(api_key=api_key)
                    
                    
                    system_prompt = f"You are a professional assistant. Extract the meeting data from the text. The output must strictly be a JSON object that matches this schema: {MeetingSummary.model_json_schema()}. Fill the JSON values in {output_language}!"
                    
                    response = client.chat.completions.create(
                        model="openai/gpt-oss-120b", 
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": transcript}
                        ],
                        response_format={"type": "json_object"}
                    )
                    
                    
                    parsed_json = json.loads(response.choices[0].message.content)

                # Displaying the extracted data in a structured format
                st.subheader(" Extracted Action Plan")
                
                st.write("**Attendees:**", ", ".join(parsed_json.get("attendees", [])))
                
                st.write("**Decisions:**")
                for decision in parsed_json.get("decisions", []):
                    st.markdown(f"- {decision}")
                
                st.write("**Tasks:**")
                
                st.table(parsed_json.get("tasks", []))

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
            
            finally:
                
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)