import streamlit as st
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from src.orchestrator import main
from src.utils.load_settings import load_yaml

config = load_yaml("config/main.yaml")

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
        
        # stream_generator = main(user_query)
        stream_generator, raw_chunks = main(user_query, evaluation_mode=True)
        full_response = st.write_stream(stream_generator)

        if config["vector_db"]["chunk_approach"] == "visual_multimodal" and config["retriever"]["highlighted_regions"] and raw_chunks:
            from src.retrieval.similarity_map import load_colqwen2_model, get_highlighted_image
            from PIL import Image
            import torch

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model, processor = load_colqwen2_model(model_name= config["models"]["visual_model"],device=device)
            number_pages = config["retriever"]["highlighted_pages"]

            with st.expander("Source pages with highlighted regions", expanded=True):
                cols = st.columns(min(len(raw_chunks[:number_pages]), number_pages))
                for col, path in zip(cols, raw_chunks[:number_pages]):
                    page_num = path.split("page_")[-1].replace(".jpg", "")
                    with col:
                        st.caption(f"Page {page_num}")
                        try:
                            img = get_highlighted_image(path, user_query, model, processor, device)
                            st.image(img, width='stretch')
                        except Exception as e:
                            st.image(Image.open(path), width='stretch')
        
    st.session_state.chat_history.append({
        "role": "assistant",
        "text": full_response,
        "raw_chunks": raw_chunks,
        "query": user_query,
    })

