import requests
import pathlib
import streamlit as st
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import json

# from dotenv import load_dotenv

# load_dotenv()  # take environment variables from .env.

import os

if not os.path.exists('temp_dir'):
    os.makedirs('temp_dir')


## Function to load OpenAI model and get respones
def get_gemini_response(prompt,data):
    model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=st.secrets["google_api_key"])
    agent = create_csv_agent(
            model, data, verbose=True)
    result=agent.run(prompt)
    print(type(result))
    print(result)
    return result


input_prompt = """
               You are an expert in annotating .tsv files.
               You will receive input .tsv file &
               you will have to analyze it and generate a .json data dictionary for that file. You will also identify missing values in the columns.\n
               For example, the tsv file is\n
               name,gender,age\n
                manya,f,20\n
                mama,f,41\n
                shivi,m,10\n
                rakesh,m,40\n
                The .json data dictionary will be :\n
                {
                    "name": {
                        "Description": "A participant ID",
                        "Annotations": {
                            "IsAbout": {
                                "Label": "Subject Unique Identifier",
                                "TermURL": "nb:ParticipantID"
                            },
                            "Identifies": "participant"
                        }
                    },
                    "gender": {
                        "Annotations": {
                            "IsAbout": {
                                "Label": "Sex",
                                "TermURL": "nb:Sex"
                            },
                            "Levels": {
                                "f": {
                                    "Label": "Female",
                                    "TermURL": "snomed:248152002"
                                },
                                "m": {
                                    "Label": "Male",
                                    "TermURL": "snomed:248153007"
                                }
                            },
                            "MissingValues": []
                        },
                        "Description": "gender of participants"
                    },
                    "age": {
                        "Annotations": {
                            "IsAbout": {
                                "Label": "Age",
                                "TermURL": "nb:Age"
                            },
                            "Transformation": {
                                "Label": "integer value",
                                "TermURL": "nb:FromInt"
                            },
                            "MissingValues": []
                        },
                        "Description": "age of the participants"
                    }
                }\n
                               Now read the dataframe provided to you and generate its .json data dictionary having the exact format as the one provided in the example in a way that it appears formatted when i print it

               """

# Setting Streamlit page config
st.set_page_config(page_title="Data Annotation", page_icon="🚀", layout="wide")

# Inline CSS
st.markdown(
    """
    <style>
        div.row-widget.stButton > button {
            margin: auto;
            display: block;
            transition: transform .2s;
        }
        div.row-widget.stButton > button:hover {
            transform: scale(1.05);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("LLM Data Annotation  ✏️")
st.write("An interactive tool to annotate your dataset, preview annotations, and save changes.")


uploaded_file = st.file_uploader("Upload a dataset (TSV)", type="tsv")

if uploaded_file is not None:
    # Save the uploaded file to a local directory
    file_path = os.path.join('temp_dir', uploaded_file.name)
    st.table(pd.read_csv(uploaded_file, sep='\t'))

    # Save the uploaded file to the file path
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Read the uploaded file into a pandas DataFrame
    if st.button("Annotate"):
        data = pd.read_csv(file_path)
        response=get_gemini_response(input_prompt,file_path)
        st.subheader("The .json data dictionary is")
        st.code(response)
