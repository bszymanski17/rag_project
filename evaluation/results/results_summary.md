# Approaches evaluation
Across first 2 approaches, text chunking was performed using a chunk_size of 1000 and a chunk_overlap of 200 (except experiments in approach 3) and the retrieval phase involved fetching the top 20 chunks using the cosine distance metric, and then retaining only those chunks with a distance score below 0.3.

# 1.1 Baseline
This strategy utilizes a traditional, text-only approach where the PDF content is extracted page-by-page as a continuous stream of text using PyMuPDFLoader. The raw text is then sequentially divided into smaller segments using the RecursiveCharacterTextSplitter based strictly on predefined character limits and overlaps. It treats all content uniformly without identifying or preserving complex structural elements like tables or images.

## RAGAS results - summary
| Content category | Faithfulness | Answer correctness | Context recall |
| :--- | :---: | :---: | :---: |
| **Text stats** | 1.0000 | 0.8788 | 1.0000 |
| **Image stats** | 0.8333 | 0.6332 | 0.8333 |
| **Table stats** | 0.8333 | 0.8864 | 0.7778 |
| **Stats all** | **0.9387** | **0.7947** | **0.8548** |
#

The baseline performs exceptionally well, providing the correct answer in 31 out of 34 cases. The understated evaluation metrics are primarily due to the fact that the LLM-generated responses were frequently more precise and detailed than the ground truth.
The 3 errors where the model failed resulted from:
- Retrieval dilution: A keyword in the question mapped to multiple different sections, causing the retriever to fetch irrelevant chunks instead of the correct table.
- Lack of visual parsing: The current lack of chart processing for graphs that don't contain explicit numerical text labels (line plots).
- Chunking boundaries: Key context sentences being cut off mid-sentence due to rigid text splitting.

Potential steps to improve model performance
- split text in logically completed block of text,
- increasing chunk_size / chunk_overlap,
- sdding image processing.
#

# 1.2 Multimodal split
This approach uses UnstructuredPDFLoader in hi_res mode to parse the document into distinct elements like text, tables, and images. While standard text is processed like in baseline approach, tables and extracted images are passed to LLM to generate search-optimized textual summaries. These descriptions are then indexed into the vector database as completely independent, standalone chunks.
# 

## RAGAS results - summary (version 1)
| Content category | Faithfulness | Answer correctness | Context recall |
| :--- | :---: | :---: | :---: |
| **Text stats** | 0.7000 | 0.7884 | 0.9000 |
| **Image stats** | 0.6667 | 0.3258 | 0.3333 |
| **Table stats** | 0.7778 | 0.6189 | 0.3333 |
| **Stats all** | **0.7647** | **0.5892** | **0.5000** |

Out of 34 trials in this approach, the model answered 'I don't know' 9 times because the retrieved chunks lacked the correct information (un the remaining cases, it also occasionally provided incorrect answers). Eight of these instances involved text and/or images. Looking at the retrieved context, the model mostly fails to connect tables or images with their accompanying text, causing performance to drop significantly below the baseline despite adding multimodal processing. To improve the results, the prompts passed to the models generating descriptions were modified

- version 1 (table): 
```text
You are a financial data indexing assistant. Analyze this table and generate a keyword-rich, comprehensive description for text-based vector retrieval. Do not summarize vaguely. Explicitly list: 
- The exact title, headings, rows, and column names. 
- All precise numerical values, metrics, percentages, and totals (state the currency and scale, e.g., US$ in millions). - Key financial trends, variances, or data alignments shown. Output a detailed, fact-dense paragraph. Avoid conversational filler
```
- version 1 (image): 
```text
You are a financial visual asset indexing assistant. Analyze this chart/image and generate a keyword-rich, comprehensive description for text-based vector retrieval. 
Explicitly textually describe: 
- The chart type, exact title, labels, and legends. 
- All specific variables on the X-axis and Y-axis including timeframes (fiscal years, dates). 
- Every exact numerical data point, value, and trajectory plotted in the graphic. Output a detailed, fact-dense paragraph. Avoid conversational filler

```
#

- version 2 (table): 
```text
You are an expert financial data digitizer for a RAG database. Your ONLY task is to convert this table into a highly detailed text representation.
CRITICAL INSTRUCTION: DO NOT summarize vaguely. You MUST explicitly extract and write down EVERY SINGLE EXACT NUMBER, metric, percentage, year, row name, and column name present in the table. Format it as dense, factual sentences (e.g., "In 2023, Total Assets were $110,547 million. In 2024, Total Assets were $120,000 million."). Leave no data point behind.
```
version 2 (image): 
```text
You are an expert financial visual asset digitizer for a RAG database. Your ONLY task is to convert this chart/graph into a highly detailed text representation. 
CRITICAL INSTRUCTION: DO NOT summarize vaguely. Explicitly read the chart and list EVERY SINGLE exact numerical value, axis label, year, legend item, and category visible in the image. Format it as dense, factual sentences (e.g., "In 2023, Total Assets were $110,547 million. In 2024, Total Assets were $120,000 million."). Leave no data point behind.
```
#
## RAGAS results - summary (version 2)
| Content category | Faithfulness | Answer correctness | Context recall |
| :--- | :---: | :---: | :---: |
| **Text stats** | 0.9000 | 0.7378 | 0.9000 |
| **Image stats** | 0.6667 | 0.3124 | 0.0000 |
| **Table stats** | 0.7778 | 0.6578 | 0.4444 |
| **Stats all** | **0.7979** | **0.5799** | **0.4706** |

Modifying the prompt did not resolve the issue, and the resulting performance showed no improvement. The potential performance of the model could be improved not by sending descriptions separately, but by embedding them within the text in place of the objects they describe, and only then splitting the text into chunks (see Approach 1.3).
# 1.3 Multimodal inline
This strategy utilizes the exact same high-resolution parsing and LLM description pipeline as multimodal_split to extract and summarize tables and images. However, instead of saving these descriptions as isolated database entries, they are stitched directly back into the continuous document text stream in their original sequence. Only after this full enrichment is the unified text partitioned using the character splitter.

## chunk_size = 1000

| Category | Faithfulness | Answer Correctness | Context Recall |
| :--- | :---: | :---: | :---: |
| **Text** | 0.7833 | 0.7938 | 0.9000 |
| **Image** | 0.7500 | 0.5576 | 0.1667 |
| **Table** | 0.7778 | 0.7020 | 0.5556 |
| **All** | 0.8270 | 0.6814 | 0.5735 |
#
While this strategy yielded solid performance, some missing answers were directly caused by restrictive chunk sizes splitting sentences in half. Consequently, further testing was carried out with different chunk_size configurations.

## chunk_size = 2000
| Category | Faithfulness | Answer Correctness | Context Recall |
| :--- | :---: | :---: | :---: |
| **Text** | 0.9000 | 0.8066 | 0.9000 |
| **Image** | 0.8333 | 0.5425 | 0.6667 |
| **Table** | 1.0000 | 0.8832 | 0.7778 |
| **All** | 0.9382 | 0.7581 | 0.7500 |

## chunk_size = 3000
| Category | Faithfulness | Answer Correctness | Context Recall |
| :--- | :---: | :---: | :---: |
| **Text** | 0.9833 | 0.9129 | 1.0000 |
| **Image** | 0.8333 | 0.6005 | 0.5000 |
| **Table** | 1.0000 | 0.9475 | 0.8889 |
| **All** | 0.9539 | 0.8209 | 0.7941 |

## chunk_size = 5000
| Category | Faithfulness | Answer Correctness | Context Recall |
| :--- | :---: | :---: | :---: |
| **Text** | 0.9833 | 0.8872 | 1.0000 |
| **Image** | 1.0000 | 0.5840 | 0.8333 |
| **Table** | 1.0000 | 0.9754 | 0.8889 |
| **All** | 0.9510 | 0.8402 | 0.8824 |

As chunk_size increases, the results consistently improve. It appears that chunk_size = 3000 is the optimal choice, striking a balance between effectiveness and efficiency. In this scenario, we observe an increase in answer correctness of over 20% compared to the model with chunk_size = 1000. In only one instance did the model respond that it did not know the answer. For all other questions, it provided either correct or partially correct answers. Further expanding the context window continues to improve performance, but the rate of growth slows down.

# 1.4 Multimodal inline with modified separators"
To improve model performance, a different set of separators was also tested to achieve a more logical text split:
- ["\n\n", "\n", " ", ""]    -> ["\n\n\n","\n\n",". ",".\n","? ","! "," ",""]

In this approach, the chunk_size was set to 3000.

| Category | Faithfulness | Answer Correctness | Context Recall |
| :--- | :---: | :---: | :---: |
| **Text** | 1.0000 | 0.8514 | 1.0000 |
| **Image** | 1.0000 | 0.5173 | 0.6667 |
| **Table** | 1.0000 | 0.9473 | 1.0000 |
| **All** | 1.0000 | 0.8038 | 0.8529 |

Compared to the previous set of separators with the same chunk_size value, we see an increase in Faithfulness and Context Recall at the cost of a slight decrease in Answer Correctness. Following this change, the model appears more reliable, delivering better and more logically segmented chunks, while the decline in answer correctness is negligible.

## 1.5 Multimodal inline with metadata handling

This approach addressing the critical limitation of location blindness inherent. By combining the sequential, enriched text stream from the inline multimodal strategy with explicit page-level metadata tagging and structured query routing, the system gains full spatial awareness of the document layout. In this approach, the separators from section 1.4 were applied.

| Category | Faithfulness | Answer Correctness | Context Recall |
| :--- | :--- | :--- | :--- |
| text | 0.9000 | 0.8385 | 0.9000 |
| image | 1.0000 | 0.6773 | 0.8333 |
| table | 1.0000 | 0.9471 | 0.8889 |
| all | 0.9412 | 0.8244 | 0.8529 |

The results are highly robust compared to previous approaches. Furthermore, by leveraging metadata, we can scope our queries to content from specific pages or request the model to cite the exact page on which it based its answer.

| Question | Answer |
| :--- | :--- |
| What does page 4 contain? | Page 4 contains the introduction to the Management’s Discussion and Analysis discussing the IFC's financial results for the fiscal year ended June 30, 2024, as well as Table 1 containing Selected Financial Data. This table presents key investment indicators and a summary of operations for fiscal years 2022, 2023, and 2024 (including own-account commitments, fund mobilization, disbursements, net income or loss, and income available for designation), as well as selected balance sheet data as of June 30, 2023 and 2024 (total assets, liquid assets, investments, debt, and total capital). Furthermore, the page includes explanatory footnotes to the presented financial data and information about the Capital Utilization Ratio for the years 2023 and 2024. |
| Tell me what charts are located from page 10 to 20? | The following charts are located on pages 10 to 20:<br>- **On page 15:** Figure 10: “Disbursed Investment Portfolio Distribution by Region”.<br>- **On page 17:** Figure 11: “Carrying Value of Loan Portfolio” and Figure 12: “Currency Composite of Disbursed Loan Portfolio”.<br>- **On page 18:** Figure 13: “Carrying Value of Equity Investment Portfolio” and Figure 14: “Carrying Value of Debt Security Portfolio”. |
| What tables are located after page 130? | The following tables are located after page 130 in the provided text:<br>- **On page 139:** A table presenting Derivative Assets and Total Assets as of June 30, 2023, containing gross amounts reported in the consolidated balance sheet, financial instruments not subject to offsetting, collateral received, and net amounts.<br>- **On page 140:** A table presenting repurchase agreements for U.S. Treasury securities (Repurchase agreements - U.S. Treasury securities) and their total (Total Repurchase agreements) in millions of US dollars, broken down by maturity.<br>- **On page 144:** A table containing financial data for the Africa Region and the Central Asia and Türkiye Region broken down by country, including metrics such as Number of Enterprises, IFC, Loan & Guarantee Participations, and Total.<br>- **On page 145:** A table representing the Europe region, containing data for 17 countries (Albania, Armenia, Azerbaijan, Belarus, Bosnia and Herzegovina, Bulgaria, Croatia, Georgia, Kosovo, Moldova, Montenegro, North Macedonia, Poland, Romania, Russian Federation, Serbia, and Ukraine) regarding the number of enterprises, IFC, loan & guarantee participations, and total.<br>- **On page 146:** A table presenting worldwide investments (Worldwide) for individual countries and regional investments broken down by count (Count), value 1 (Value 1), value 2 (Value 2), and total value (Total Value). |
| What tables are located before page 10? | Before page 10 of the report, there is Table 3: “Summary of Financial Results” located on page 9. |
| What is the first row in the table on page 9? | There are two tables on page 9:<br>- **In Table 3 (Summary of Financial Results):** The first row (item) in the Consolidated Statement of Operations Highlights section is “Income from loans and guarantees, including realized gains and losses on loans and associated derivatives”.<br>- **In Table 4 (Key Financial Ratios):** The first row (item) is “Overall liquidity ratio (a)”. |
|What is the stated mission of the International Finance Corporation? State which page you took the answer from.|The mission of the International Finance Corporation is to end extreme poverty and boost shared prosperity on a livable planet. This information is from page 5.|


## 1.6 Re-ranker
Another attempt to improve performance involves adding a reranker to Approach 1.5. The system first casts a wide net across the vector space to retrieve the top 50 candidate chunks using the embedding model. Subsequently, a computationally intensive Cross-Encoder performs deep cross-attention evaluation over all 50 query-chunk pairs, re-ranking them dynamically and pruning the final payload down to the top 10 most relevant chunks forwarded to the LLM.

| Category | Faithfulness | Answer Correctness | Context Recall |
| :--- | :--- | :--- | :--- |
| text | 0.9833 | 0.8172 | 0.9000 |
| image | 1.0000 | 0.6606 | 0.8333 |
| table | 0.8889 | 0.7980 | 0.7778 |
| all | 0.9620 | 0.7140 | 0.7794 |

Replacing the previous filtering mechanism with a reranker led to a drop in performance. One potential cause could be an insufficient number of retrieved chunks; therefore, we will increase this value from 10 to 20.

| Category | Faithfulness | Answer Correctness | Context Recall |
| :--- | :--- | :--- | :--- |
| text | 0.9500 | 0.8463 | 1.0000 |
| image | 0.8333 | 0.5828 | 0.8333 |
| table | 1.0000 | 0.8482 | 0.8889 |
| all | 0.9242 | 0.7792 | 0.8676 |

After increasing the number of retrieved chunks, we observe a significant boost in both Answer Correctness and Context Recall metrics; however, the results still fall short compared to the approach without a reranker.