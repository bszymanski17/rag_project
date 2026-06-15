import streamlit as st
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from PIL import Image
import torch

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
        
        stream_generator, raw_chunks = main(user_query, evaluation_mode=True)
        full_response = st.write_stream(stream_generator)
        number_pages = config["retriever"].get("highlighted_pages", 3)

        if config["vector_db"]["chunk_approach"] == "visual_multimodal" and config["retriever"]["highlighted_regions"] and raw_chunks:
            from src.retrieval.similarity_map_generator import load_colqwen2_model, get_highlighted_image

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
                            img = get_highlighted_image(path, full_response, model, processor, device)
                            st.image(img, width='stretch')
                        except Exception as e:
                            st.image(Image.open(path), width='stretch')
        elif config["vector_db"]["chunk_approach"] == "visual_database" and raw_chunks:
            from src.retrieval.bounding_boxes_generator import draw_bounding_boxes_from_metadata
            from src.retrieval.visual_database_retrieval import rerank_patches_by_answer
            from src.retrieval.similarity_map_generator import load_colqwen2_model

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model, processor = load_colqwen2_model(device=device)
            strict_db_path = config["vector_db"]["db_path"]
            best_patches_dict = rerank_patches_by_answer(
                answer=full_response,
                raw_chunks=raw_chunks,
                db_path=strict_db_path,
                model=model,
                processor=processor,
                device=device,
            )

            with st.expander("Bounding Boxes", expanded=True):
                cols = st.columns(min(len(raw_chunks[:number_pages]), number_pages))
                for col, path in zip(cols, raw_chunks[:number_pages]):
                    page_num = int(path.split("page_")[-1].replace(".jpg", ""))
                    with col:
                        st.caption(f"Page {page_num}")
                        try:
                            winning_indices = best_patches_dict.get(page_num, [])
                            if winning_indices:
                                img = draw_bounding_boxes_from_metadata(
                                    image_path=path,
                                    winning_patch_indices=winning_indices,
                                    db_path=strict_db_path,
                                    page_num=page_num,
                                )
                                st.image(img, width='stretch')
                            else:
                                st.image(Image.open(path), width='stretch')
                        except Exception as e:
                            print(f"ERROR page {page_num}: {e}")
                            st.image(Image.open(path), width='stretch')
        
    st.session_state.chat_history.append({
        "role": "assistant",
        "text": full_response,
        "raw_chunks": raw_chunks,
        "query": user_query,
    })

