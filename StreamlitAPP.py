import os
import json
import traceback
import pandas as pd
import streamlit as st
from langchain.callbacks import get_openai_callback
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain

# Load JSON configuration
with open('Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

# Title
st.title("MCQs Creator Application with LangChain 🦜⛓️")

# --- User Inputs (outside form for simplicity) ---
uploaded_file = st.file_uploader("Upload a PDF or TXT file")
mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=50)
subject = st.text_input("Insert Subject", max_chars=20)
tone = st.text_input("Complexity Level Of Questions", max_chars=20, placeholder="Simple")
button = st.button("Create MCQs")  # simple button, no form

# --- Function to show download button separately ---
def download_quiz(df):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Quiz as CSV",
        data=csv,
        file_name="quiz.csv",
        mime="text/csv"
    )

# --- Generate and display MCQs ---
if button and uploaded_file is not None:
    with st.spinner("Generating quiz..."):
        try:
            text = read_file(uploaded_file)
            with get_openai_callback() as cb:
                response = generate_evaluate_chain(
                    {
                        "text": text,
                        "number": mcq_count,
                        "subject": subject,
                        "tone": tone,
                        "response_json": json.dumps(RESPONSE_JSON)
                    }
                )

                # Optional: print token usage in console
                print(f"Total Tokens: {cb.total_tokens}")
                print(f"Prompt Tokens: {cb.prompt_tokens}")
                print(f"Completion Tokens: {cb.completion_tokens}")
                print(f"Total Cost: {cb.total_cost}")

        except Exception as e:
            st.error(f"Error while generating quiz: {str(e)}")
            traceback.print_exc()
        else:
            if isinstance(response, dict):
                quiz = response.get("quiz", None)
                if quiz:
                    table_data = get_table_data(quiz)
                    if table_data:
                        df = pd.DataFrame(table_data)
                        df.index = df.index + 1

                        # Display table & review
                        st.table(df)
                        st.text_area(label="Review", value=response.get("review", ""))

                        # --- Call separate download function ---
                        download_quiz(df)
                    else:
                        st.error("Error in table data")
            else:
                st.write(response)
