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
├── analysis/
│   └── chunk_analysis.ipynb         # Research notebook for testing thresholds for chunks
├── chroma_databases/                 # Local directory containing persistent Chroma DB vector collections
├── config/
│   ├── evaluation.yaml              # Configuration of evaluation pipeline
│   └── main.yaml                    # Main system configuration
├── evaluation/
│   ├── input/
│   │   └── RAG_evaluation_dataset.csv # Evaluation dataset containing questions, contexts, and ground truth answers
│   └── results/                     # RAGAS evaluation outputs across different strategies
│       ├── baseline/                # Test metrics for the raw text-only approach
│       ├── multimodal_inline/       # Test metrics for the inline approach
│       ├── multimodal_split/        # Test metrics for the isolated object descriptions approach
│       └── results_summary.md       # Consolidated markdown file with comparative metrics across all runs
├── images/                          # Target directory for graphics, charts, and tables extracted from the PDFs
├── knowledge/
│   └── ifc-annual-report-2024-financials.pdf # Source financial document acting as the RAG knowledge base
├── prompts/
│   └── main.yaml                    # Prompt templates
├── schemas/
│   ├── config_schemas.py            # Pydantic schemas for validating YAML configuration structures
│   └── llm_output_schemas.py        # Pydantic definitions enforcing structured outputs from the LLM
├── src/                             # Main application source code folder
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── eval_utils.py            # Helper functions for data aggregation and formatting
│   │   └── evaluation_pipeline.py   # Executable pipeline that triggers a full evaluation run using RAGAS
│   ├── generation/
│   │   └── answer_generator.py      # Module handling prompt assembly and final answer generation from the LLM
│   ├── indexing/                    # Core module handling document extraction, processing, and indexing
│   │   ├── chunk_strategies/        # Concrete implementation of the three progressive chunking strategies
│   │   │   ├── baseline/
│   │   │   │   └── chunker.py       # Implements the simple text-splitting stream using PyMuPDF
│   │   │   ├── multimodal_inline/
│   │   │   │   ├── chunker.py       # Implements continuous stream stitching of enriched metadata and text
│   │   │   │   └── page_chunker.py  # Advanced chunker that strictly obeys page boundaries and preserves location metadata
│   │   │   └── multimodal_split/
│   │   │       └── chunker.py       # Isolates table and image descriptions as standalone embedding chunks
│   │   ├── database_initializer.py  # Script for safe collection creation, embedding function bindings, and resets
│   │   ├── database_orchestrator.py # High-level coordinator managing chunk batch writes and metadata pushes to Chroma DB
│   │   └── element_describer.py     # Communication engine with LLM for generating text descriptions of visual elements
│   ├── retrieval/                   # Context fetcher and ranking engine
│   │   ├── context_formatter.py     # Formats retrieved chunks into a solid prompt context block
│   │   ├── query_metadata_extractor.py # Leverages a structured LLM to extract page-level metadata constraints
│   │   ├── reranker.py              # Executes a Cross-Encoder deep relevance evaluation to re-rank candidate chunks
│   │   ├── retrieval_filter.py      # Filters retrieved chunks based on semantic distance thresholds
│   │   └── retrieval_engine.py      # Handles initial Vector DB retrieval and orchestrates the retrieval part
│   ├── utils/                       # Shared utility functions
│   │   ├── image_utils.py           # Image operations 
│   │   ├── load_settings.py         # Handlers for robust loading of YAML configurations and environment variables
│   │   └── logger_config.py         # Configures unified console logging formats and reporting levels
│   ├── __init__.py
│   └── orchestrator.py              # Main system engine bridging the Retrieval and Generation pipelines (the core RAG pipeline)
├── .env                            
├── .gitignore                       
├── app.py                           # Application entry point script (UI interface)
├── README.md                        
└── requirements.txt                 
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
