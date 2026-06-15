# Multimodal Financial RAG System

An advanced Retrieval-Augmented Generation pipeline designed for deep semantic auditing and precise information retrieval from complex, financial PDF document. This system moves beyond basic text extraction by intelligently parsing structural elements like tables and charts.

## Retrieval strategies & approaches

The system architecture was developed and evaluated across three distinct progressive strategies:

1. **Baseline (Text-Only):**
   A traditional approach utilizing `PyMuPDFLoader` to extract the PDF content page-by-page as a continuous stream of raw text. The text is sequentially divided into smaller segments using the `RecursiveCharacterTextSplitter` based strictly on predefined character limits. It treats all content uniformly, completely ignoring or flattening complex structural elements like tables or visual charts.

2. **Multimodal Split:**
   An advanced parsing approach using high-resolution layout analysis to detect tables and images. Instead of flattening them, these elements are passed to an LLM to generate keyword-rich, search-optimized textual summaries. These generated descriptions are then indexed into the vector database as independent chunks, isolated from the surrounding page context.

3. **Multimodal Inline:**
   The current state-of-the-art strategy for this pipeline. It leverages the same high-resolution parsing and LLM description engine as the split approach. However, instead of isolating the summaries, the generated descriptions are stitched directly back into the continuous text stream in their original sequence. The text splitter is applied only after this full enrichment, preserving vital structural context and preventing location blindness.

4. **Visual approach**
   This system implements a visual RAG pipeline where PDF pages are converted to high-resolution images and indexed using ColPali (ColQwen2), a late interaction multimodal model that generates patch-level embeddings to retrieve the most visually and semantically relevant pages. Retrieved pages are then passed as base64-encoded images directly to Gemini, which synthesizes the final answer from the visual context without any text extraction.

Additionally, enhancements such as rerankers and the integration of metadata (page numbers) were evaluated.

## Project Structure

```
.
├── .byaldi/                        # Local cache for ColPali models
├── .dockerignore                   
├── .gitignore                   
├── .env                            # API keys and environment secrets
├── Dockerfile                      # Blueprint configuration for the Linux container image
├── app.py                          # Streamlit chat interface
├── config/
│   ├── evaluation.yaml             # Evaluation configuration (paths, models)
│   └── main.yaml                   # Main configuration (models, retriever, vector DB settings)
├── documents/
│   ├── analysis/            
│   │   └── chunk_filtering_threshold_analysis.ipynb         
│   └── evaluaton/
│       ├── input/
│           └── RAG_evaluation_dataset.csv # evaluatuion dataset
│       └── output/ # evaluaton results         
├── prompts/
│   └── main.yaml                   # LLM prompt templates
├── knowledge/                      # Source PDF documents
├── images/
│   ├── pages/             # PDF pages converted to images (visual pipeline)
│   └── plots/             # Retrived plots from multimodal approach
├── schemas/
│   ├── config_schemas.py           # Configuration validation schemas
│    └──llm_output_schemas.py       # LLM output validation schemas
└── src/
    ├── orchestrator.py             # Main RAG pipeline orchestrator
    ├── indexing/
    │   ├── database_orchestrator.py          # Initializes vector DB (routes by chunk approach)
    │   ├── database_initializer.py           # Chroma/Qdrant/FAISS DB creation
    │   ├── visual_database_initializer.py    # Indexes image patches into database
    │   ├── visual_byaldi_initializer.py      # ColPali index creation (visual pipeline)
    │   ├── element_describer.py              # Multimodal element description via LLM
    │   └── chunk_strategies/
    │       ├── baseline/
    │       │   └── chunker.py                # Basic text chunking
    │       ├── multimodal_inline/
    │       │   ├── chunker.py                # Multimodal chunking with inline elements
    │       │   └── page_chunker.py           # Page-level chunking with metadata
    │       ├── multimodal_split/
    │       │   └── chunker.py                # Multimodal chunking with split elements
    │       └── visual_processing/
    │           └── pdf_converter.py          # PDF to image conversion
    ├── retrieval/
    │   ├── retrieval_orchestrator.py         # Routes retrieval by chunk approach
    │   ├── visual_retrieval.py               # ColPali visual semantic search
    │   ├── similarity_map.py                 # Source attribution heatmap generation
    │   ├── context_formatter.py              # Formats retrieved chunks for LLM prompt
    │   ├── query_metada_extractor.py         # Extracts page range filters from query
    │   └── bounding_boxes/
    │   │   ├── bounding_boxes_generator.py   # Draw bounding boxes (visual_database approach)
    │   │   └── similarity_map_generator.py   # Generate similarity heatmap (visual_multimodal approach)
    │   └── chunk_filtering/
    │       ├── reranker.py                   # Cross-encoder reranking (BGE)
    │       └── retrieval_filter.py           # Distance threshold filtering
    ├── generation/
    │   ├── answer_orchestrator.py            # Routes generation by chunk approach
    │   ├── answer_generator.py               # Text-based Gemini response generation
    │   └── answer_generator_visual.py        # Visual Gemini response
    ├── evaluation/
    │   ├── evaluation_pipeline.py            # End-to-end evaluation runner
    │   └── eval_utils.py                     # RAGAS metrics utilities
    └── utils/
        ├── load_settings.py                  # YAML config and env loader
        ├── logger_config.py                  # Logger setup
        └── image_utils.py                    # Image processing helpers       
```

## Setup & Installation

### 1. Install Dependencies

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory of the project. See the configuration example below.

### 3. Launch the Application

```bash
streamlit run app.py
```

---

## Configuration Setup

### Application Config (`config/main.yaml`)

Defines models, paths, chunking and retrieval parametrs,

```yaml
models:
  main_model: gemini-3.5-flash
  embedding_model: text-embedding-005
  judge_model: gemini-3.5-flash
  desc_model: gemini-3.5-flash
  metadata_model: gemini-2.5-fash
  visual_model: vidore/colqwen2-v1.0

retriever:
  top_k: 15
  distance_treshold: 0.3
  highlighted_regions: True # True / False
  highlighted_pages: 3


vector_db:
  # paths
  knowladge_path: knowledge/ifc-annual-report-2024-financials.pdf
  db_path: ./databases/chroma_databases/multimodal_visual
  interim_images_dir: images/pages

  # parameters
  chunk_size: 3000
  chunk_overlap: 500
  similarity_metric: "cosine"

  # approaches
  chunk_approach: visual_multimodal                         # baseline / multimodal_inline / multimodal_inline_metadata / multimodal_split
  provider: chroma                                          # chroma / qdrant / faiss
  qdrant_collection_name: multimodal_inline_3000_metadata
  elements_blacklist: []                                    # e.g. ["Image", "Table"] (Only for multimodal apporaches!)


reranker:
  use_reranker: False                                       # True/False
  top_n: 10

```

### Environment Secrets (`.env` Example)

The application expects database credentials, GCP environment and Langfuse configuration.

```env
# Google Cloud Platform Setup
GCP_PROJECT_ID=your-gcp-project-id

# Langfuse configuration 
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_BASE_URL=your_langfuse_base_url

```
