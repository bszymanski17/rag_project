import streamlit as st
from src.utils.tools import load_yaml
from src.orchestrator import main

st.title("IFC Analyst Assistant")
user_query = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["text"])

user_query = st.chat_input(placeholder="Ask question about IFC annual report...")

if user_query:    
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.chat_history.append({"role": "user", "text": user_query})

    with st.chat_message("assistant"):
        
        stream_generator = main(user_query)
        
        full_response = st.write_stream(stream_generator)
        
    st.session_state.chat_history.append({"role": "assistant", "text": full_response})


