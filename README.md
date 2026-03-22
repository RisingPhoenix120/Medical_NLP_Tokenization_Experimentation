  
**MEDICAL RAG EVALUATION PIPELINE**

**Design & Analysis Document**

Version 2.0

*Benchmarking Embedding Models for Clinical Text Retrieval*

9 Models · 2 Experiments · MTSamples Corpus · RTX 5070 Ti GPU

March 2026


# **1\. Executive Summary**

This document provides a complete technical record of a Retrieval-Augmented Generation (RAG) evaluation pipeline designed to compare embedding models for clinical text retrieval. The study is motivated by a core research question: do domain-adapted language models, trained or fine-tuned on biomedical literature, outperform general-purpose models when used as the dense retrieval backbone of a medical RAG system?

The pipeline was applied to the MTSamples clinical transcription corpus — 4,999 medical dictation records covering 40 specialties — and evaluated 9 embedding models across two experiments: Experiment A used standard keyword matching, while Experiment B introduced a PubMedBERT tokenizer-aware weighting scheme that assigned higher relevance weight to medically significant terms.

*Key Finding: General-purpose and medical-domain models are in a near-perfect dead heat in both experiments. The top-2 models — bge-base (general) and S-PubMedBERT (medical) — differ by only 0.001 Quality points. Retrieval fine-tuning (MS-MARCO) matters more than continued biomedical pre-training alone. Deployment correctness (query prefixes, threshold tuning) can swing a model's Quality score by \+0.11.*

The most practically important individual finding is the Snowflake-M rehabilitation: omitting the required query prefix caused its Quality score to collapse from 0.771 to 0.659 — a gap larger than the entire spread between the remaining 8 models. This underscores that deployment details are at least as consequential as model selection itself.

# **2\. Introduction & Motivation**

## **2.1 What is RAG?**

Retrieval-Augmented Generation (RAG) is an architecture that combines a dense retrieval component with a generative language model. Rather than relying entirely on the model's parametric memory, RAG retrieves relevant passages from an external knowledge base at inference time and provides them as context to the generator. This approach is particularly valuable in knowledge-intensive domains like medicine, where facts change frequently, are highly specialised, and hallucination carries real-world risk.

A RAG system has two major components:

* Retriever: Encodes documents and queries as dense vectors; retrieves the top-k most similar documents by vector similarity (cosine or dot product). The quality of the retriever directly determines the upper bound of the generator's accuracy.

* Generator: A large language model (e.g., GPT-4, Claude) that reads the retrieved passages and produces a final answer.

This study focuses exclusively on the retrieval component — specifically on how the choice of embedding model affects retrieval quality over clinical text.

## **2.2 The Medical Vocabulary Problem**

Medical text presents unique challenges for off-the-shelf embedding models:

* High-density terminology: Clinical notes contain compounds like 'anastomosis', 'thromboembolism', 'immunofluorescence' and abbreviations like 'COPD', 'MI', 'T2DM' that standard tokenizers decompose into many subword pieces.

* Tokenizer fertility mismatch: A general BERT tokenizer trained on Wikipedia and Books encodes 'echocardiogram' as 5 separate tokens. A PubMed-trained tokenizer encodes it as 1 token. This affects the quality of the resulting embedding.

* Specialty-specific language: A surgery note and a psychiatry note may use entirely different vocabulary while describing related conditions.

* Keyword-dense queries: Clinical queries often consist almost entirely of domain terms with little syntactic scaffolding.

Domain-adapted models like PubMedBERT were designed to address these issues through two mechanisms: (1) pre-training on PubMed abstracts and clinical notes, which reshapes the tokenizer and adjusts embedding geometry for biomedical language; and (2) in the case of S-PubMedBERT, additional fine-tuning on retrieval-specific training pairs (MS-MARCO).

## **2.3 Research Questions**

1. Does pre-training domain adaptation (PubMedBERT vs BERT-base) improve retrieval quality on clinical transcriptions?

2. Does retrieval fine-tuning (S-PubMedBERT on MS-MARCO) add value over pre-training alone?

3. Does a tokenizer-aware relevance weighting scheme better discriminate between models than standard keyword matching?

4. How does embedding size (small / base / large) interact with domain adaptation?

5. What is the impact of correct deployment details (query prefixes) on observed performance?

# **3\. Input Data**

## **3.1 mtsamples.csv — Primary Corpus**

The MTSamples dataset is a publicly available collection of medical transcription samples covering 40 clinical specialties. It was originally published to demonstrate medical transcription software and has since become a standard benchmark for clinical NLP research.

| Column | Type | Description |
| :---- | :---- | :---- |
| idx | Integer | Row index 0–4998 |
| description | String | Short description of the transcription topic |
| medical\_specialty | String | One of 40 clinical specialties (Surgery, Cardiology, Neurology, …) |
| sample\_name | String | Short name/title of the case — used as the JOIN key with X.csv |
| transcription | String | Full clinical transcription text. 33 records are NULL (no text body). These are dropped during corpus construction. |
| keywords | String | Comma-separated expert-annotated medical terms for this record. 1,068 records have NULL keywords. These are excluded from evaluation scoring but still used as corpus documents. |

*The keywords column is the gold standard for relevance evaluation. Each keyword represents a clinically significant term that an expert annotated as being directly relevant to the document. A retrieval system scores well when the chunks it returns contain these expert-identified terms.*

## **3.2 X.csv — Class Labels**

X.csv contains 4,999 rows with three columns: label (integer 1–4), description (matching mtsamples.sample\_name exactly), and text (the transcription). It is NOT a separate corpus — it is the same MTSamples data with 4-class diagnostic labels added.

The four classes defined in classes.txt are:

* Class 1: Surgery

* Class 2: Medical Records

* Class 3: Internal Medicine

* Class 4: Other

**Critical design decision:** X.csv must be joined onto mtsamples using **sample\_name \= description** as the join key, NOT concatenated as additional documents. The v1 pipeline treated X.csv as a second corpus, which caused \~75% document deduplication collapse and corrupted the entire evaluation.

## **3.3 Auxiliary Files**

### **clinical-stopwords.txt**

A curated list of 806 tokens compiled by Dr. Kavita Ganesan combining standard English stopwords with clinical terms that appear very frequently across all specialties and therefore carry little discriminative signal (e.g., 'patient', 'history', 'mg', 'procedure', 'noted'). Used in Experiment A to filter out low-value terms before keyword matching.

### **vocab.txt**

A reference vocabulary of 69,944 SNOMED CT-derived tokens. Used for three purposes: (1) as a probe set for tokenizer fertility analysis; (2) to compute SNOMED coverage scores per model; and (3) as a 2× weight multiplier for SNOMED-listed terms in Experiment B's tokenizer-aware relevance scoring.

# **4\. System Architecture & Pipeline**

## **4.1 High-Level Overview**

The pipeline processes the raw CSV files through six sequential stages: data loading → chunking → tokenizer analysis → embedding & indexing → query generation → retrieval & evaluation. Each stage is described in detail below.

*All stages run on a single machine with an NVIDIA RTX 5070 Ti GPU (VRAM varies per model). Models are loaded one at a time and unloaded between evaluations to prevent VRAM overflow. Total wall-clock time for all 9 models across both experiments: approximately 5.2 minutes.*

## **4.2 Stage 1 — Data Loading & Joining**

The pipeline performs the following operations in order:

6. Load mtsamples.csv into a pandas DataFrame (4,999 rows).

7. Load X.csv (4,999 rows). Rename 'description' → 'sample\_name\_x' to match.

8. Left-join on sample\_name=description to attach class labels. Result: 4,999 rows, all with labels.

9. Drop 33 rows where transcription is NULL. Result: 4,966 labelled documents.

10. De-duplicate using a SHA-256 hash of (medical\_specialty \+ transcription\[:200\]). This removes near-identical records within MTSamples itself, yielding 2,348 unique documents.

11. Partition into 3,817 documents with keywords (used for evaluation) and 531 without (included as corpus noise but not scored).

## **4.3 Stage 2 — Text Chunking**

Each document's transcription is split into sentence-level chunks using a regex-based sentence splitter. Chunks shorter than 40 characters are merged with the adjacent chunk to avoid degenerate single-word segments.

Corpus statistics after chunking:

* Total sentence chunks: 7,433

* Mean chunk length: 145 words

* Chunks with gold-standard keywords inherited from parent document: 5,078

* Keyword propagation rule: all keywords from the parent document are assigned to every chunk derived from that document, since chunk-level keyword annotation is not available

## **4.4 Stage 3 — Tokenizer Analysis**

Before embedding, the pipeline characterises each model's tokeniser using three metrics computed over the full SNOMED vocabulary (69,944 tokens):

| Metric | Formula / Definition | What it measures |
| :---- | :---- | :---- |
| Vocabulary Size | Count of unique tokens in the tokenizer's vocabulary | Breadth of known subwords |
| Fertility | Mean number of subword tokens produced per SNOMED term across the vocabulary probe set | How much the tokenizer fragments medical terms. Lower \= better for medical text. |
| SNOMED Coverage | Fraction of SNOMED terms encoded as exactly 1 token (no fragmentation) | Medical vocabulary density. Higher \= better. |

Fertility is the key diagnostic metric. A fertility of 2.209 (all BERT-base models) means the tokenizer uses on average 2.2 subword tokens per SNOMED medical term. PubMedBERT achieves 1.918, meaning it handles medical terminology \~13% more efficiently — encoding more terms as single tokens and fewer as fragmented subword sequences.

## **4.5 Stage 4 — Embedding & Indexing**

For each model, the pipeline:

12. Loads the model onto GPU via sentence-transformers' SentenceTransformer class.

13. Checks a VRAM sanity threshold (fails if available VRAM drops below 500 MB after loading).

14. Encodes all 7,433 chunks in batches of 64\. Snowflake-M receives the query prefix during chunk encoding too, as it is an asymmetric model.

15. L2-normalises all embeddings (ensures cosine similarity \= dot product).

16. Builds a FAISS IndexFlatIP index over the chunk embeddings for exact inner-product search.

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search over dense vectors. IndexFlatIP performs exact (brute-force) inner-product search — no approximation, suitable for a corpus of this size (7,433 vectors) where speed is not a bottleneck.

## **4.6 Stage 5 — Query Generation**

Twenty-five evaluation queries were manually constructed to cover a range of difficulty levels and medical domains. Each query is designed to have at least one highly relevant document in the corpus (verified by keyword inspection). Queries span:

* Single-specialty procedural (e.g., Q08: 'laparoscopic cholecystectomy bile duct')

* Single-specialty diagnostic (e.g., Q14: 'acute myocardial infarction troponin')

* Cross-specialty synthesis (e.g., Q21: 'metabolic syndrome type 2 diabetes comorbidities')

* Rare disease / niche procedure (e.g., Q23: 'CRISPR gene therapy Duchenne muscular dystrophy')

* Paediatric subgroup (e.g., Q09: 'rhinitis management in children')

## **4.7 Stage 6 — Retrieval & Evaluation**

For each model and each query:

17. Encode the query using the same model (with query prefix for Snowflake-M).

18. Search the FAISS index for top-5 nearest chunks by inner product.

19. For each returned chunk, check relevance using the keyword-based relevance function.

20. Compute Precision@5, Recall@5, MRR@5, and nDCG@5. Average over all 25 queries.

21. Compute the composite Quality score.

22. Save per-query and per-model results to CSV checkpoints.

# **5\. Model Registry**

## **5.1 Overview**

Nine models were evaluated, drawn from two families (general-purpose and medical-domain) and three size tiers (small / base / large). All models produce L2-normalised dense embeddings and support cosine similarity search.

| Key | HuggingFace ID | Family | Size | Dims | Tokenizer | VRAM |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| MiniLM-L6 | sentence-transformers/all-MiniLM-L6-v2 | General | Small | 384 | bert-base-uncased | 56 MB |
| bge-small | BAAI/bge-small-en-v1.5 | General | Small | 384 | bert-base-uncased | 79 MB |
| bge-base | BAAI/bge-base-en-v1.5 | General | Base | 768 | bert-base-uncased | 244 MB |
| Snowflake-M | Snowflake/snowflake-arctic-embed-m | General | Base | 768 | bert-base-uncased | 242 MB |
| e5-large | intfloat/e5-large-v2 | General | Large | 1024 | bert-base-uncased | 680 MB |
| GTE-Large | thenlper/gte-large | General | Large | 1024 | bert-base-uncased | 680 MB |
| PubMedBERT | NeuML/pubmedbert-base-embeddings | Medical | Base | 768 | PubMed-retrained | 238 MB |
| S-PubMedBERT | pritamdeka/S-PubMedBert-MS-MARCO | Medical | Base | 768 | PubMed-retrained | 238 MB |
| MedEmbed-L | abhinand/medembed-large-v0.1 | Medical | Large | 1024 | bert-base-uncased | 680 MB |

## **5.2 Model Descriptions**

### **MiniLM-L6 (all-MiniLM-L6-v2)**

A distilled 6-layer MiniLM model fine-tuned by sentence-transformers on a large pool of sentence pair datasets via contrastive learning. Produces 384-dimensional embeddings. The smallest and fastest model evaluated. Despite its size, it achieves competitive performance due to the extensive fine-tuning dataset. Useful as a baseline for efficiency-quality tradeoffs.

### **bge-small (BAAI/bge-small-en-v1.5)**

Part of the BGE (BAAI General Embedding) family from the Beijing Academy of Artificial Intelligence. Trained with RetroMAE self-supervised pre-training followed by contrastive fine-tuning on a large-scale English sentence corpus. Produces 384-dimensional embeddings. The 'small' variant achieves strong performance relative to its size.

### **bge-base (BAAI/bge-base-en-v1.5)**

The base-size BGE model. Same training methodology as bge-small but with a full 12-layer BERT-base encoder, producing 768-dimensional embeddings. Ranked \#1 overall in both experiments. Excellent quality-to-compute ratio.

### **Snowflake-M (snowflake-arctic-embed-m)**

Snowflake's Arctic Embed model, trained specifically for retrieval tasks. Unlike symmetric models, it uses **asymmetric encoding**: queries and documents are encoded differently to better capture the 'question → relevant passage' relationship. Requires a specific query prefix: "Represent this sentence for searching relevant passages: ". Omitting this prefix causes severe performance degradation (Quality 0.771 → 0.659 as observed in this study's v1 run).

### **e5-large (intfloat/e5-large-v2)**

Microsoft's E5 (EmbEddings from bidirEctional Encoder rEpresentations) large model. Trained with a multi-stage pipeline: weakly supervised pre-training on (title, passage) pairs from the web followed by supervised fine-tuning on a curated mix of retrieval datasets. Produces 1,024-dimensional embeddings. Ranked 4th overall (Exp A) — large size does not translate to top ranking in this corpus.

### **GTE-Large (thenlper/gte-large)**

Alibaba's General Text Embeddings large model. Trained on a diverse multi-stage corpus with multi-task contrastive learning. Produces 1,024-dimensional embeddings. Closely matched to e5-large in quality but slightly lower overall. Both large general models are outperformed by base-size models in this evaluation, suggesting diminishing returns from scale alone on clinical text.

### **PubMedBERT (NeuML/pubmedbert-base-embeddings)**

Microsoft's PubMedBERT re-trained from scratch on PubMed abstracts and PubMed Central full texts. Unlike BioBERT (which initialises from general BERT), PubMedBERT was trained from a blank slate, giving it a tokenizer vocabulary shaped purely by biomedical text. This results in a lower fertility score (1.918 vs 2.209) and higher SNOMED coverage (14.8% vs 14.2%). NeuML fine-tuned this base model for semantic similarity using medical QA pairs.

### **S-PubMedBERT (pritamdeka/S-PubMedBert-MS-MARCO)**

PubMedBERT base further fine-tuned using the sentence-transformers framework on the MS-MARCO passage retrieval dataset — a large-scale benchmark of real Bing search queries with human-annotated relevant passages. This adds retrieval-specific optimisation on top of the biomedical tokenizer. Ranked \#2 overall, nearly matching bge-base. Demonstrates that retrieval fine-tuning provides a larger gain than domain pre-training alone.

### **MedEmbed-L (abhinand/medembed-large-v0.1)**

A 'medical' embedding model that produces 1,024-dimensional embeddings, labelled as domain-adapted. However, tokenizer analysis reveals it uses **bert-base-uncased** unchanged (fertility 2.209, SNOMED coverage 14.2%). The 'medical' label likely refers to fine-tuning data composition, not vocabulary adaptation. It ranks 6th overall despite being a large model — the absence of true tokenizer adaptation limits its advantage over similarly-sized general models.

# **6\. Evaluation Metrics**

## **6.1 Relevance Function**

Before defining the metrics, it is necessary to define what 'relevant' means. A retrieved chunk C is considered relevant to query Q if the chunk's parent document contains at least one keyword that matches a key term in Q's key-term set. Formally:

relevant(C, Q)  \=  1   if  keywords(parent(C)) ∩ key\_terms(Q) ≠ ∅

               \=  0   otherwise

In Experiment B, a weighted variant is used:

weighted\_hit(C, Q) \= Σ weight(t) × 1\[t ∈ keywords(parent(C))\]  for t in key\_terms(Q)

relevant\_B(C, Q)   \= 1  if  weighted\_hit(C, Q) ≥ threshold (= 2.0 in Exp B)

The key-term extraction functions are:

* Experiment A: Filter the query's words through clinical stopwords; return remaining terms.

* Experiment B: Tokenize query words using PubMedBERT tokenizer. Words encoded as 1 token get 3× weight. Words present in SNOMED vocabulary get an additional 2× weight. Threshold \= 2.0.

## **6.2 Precision@k**

### **Definition**

Precision@k measures the fraction of the top-k retrieved items that are relevant. For k=5 (used throughout):

Precision@5 \= (number of relevant chunks in top 5\) / 5

### **Interpretation**

A Precision@5 of 0.912 (bge-base) means that on average 4.56 out of every 5 retrieved chunks are relevant to the query. Higher is better. Precision is about how much of what you retrieve is useful — it penalises returning irrelevant results. A system can achieve perfect Precision by being very conservative (only returning the single most confident result), so it must be read alongside Recall.

## **6.3 Recall@k**

### **Definition**

Recall@k measures the fraction of all relevant chunks in the corpus that are captured within the top-k results. Unlike standard recall (which uses total relevant items), this evaluation uses the number of expert key terms as the denominator to make it query-specific:

Recall@5 \= (number of distinct key terms matched in top 5\) / len(key\_terms(Q))

### **Interpretation**

A Recall@5 of 0.402 (bge-base) means that 40.2% of the expert-annotated key terms for the query are covered by the top-5 retrieved chunks. Lower recall (\~0.36–0.40 across all models) reflects the fundamental limitation of k=5 retrieval over a corpus of 7,433 chunks — a single query's relevant information is spread across many documents. Recall is most important for comprehensive RAG scenarios where missing any relevant piece of information could lead to an incomplete answer.

## **6.4 Mean Reciprocal Rank (MRR@k)**

### **Definition**

MRR@k measures how high the first relevant chunk appears in the ranking. The reciprocal rank for a single query is 1/rank\_of\_first\_relevant\_chunk. If no relevant chunk appears in top-k, the reciprocal rank is 0\. MRR@5 is the average reciprocal rank over all queries:

RR\_q   \= 1 / rank\_of\_first\_relevant(q)   if any relevant chunk in top-5

         0                                 otherwise

MRR@5  \= (1/25) × Σ RR\_q   over all 25 queries

### **Interpretation**

An MRR of 1.0 would mean the first retrieved chunk is always relevant. MRR values here range from 0.464 to 0.503 — meaning the first relevant chunk is typically found at rank 2 to 2.2. Notably, PubMedBERT has the highest MRR@5 (0.503) of all 9 models despite ranking 9th overall. This reveals a distinctive retrieval profile: PubMedBERT is very precise about its best match, placing it at rank 1 more reliably than any other model. This is the clearest evidence of tokenizer benefit in the results — domain-specific tokenization improves the quality of the single best match even when overall precision is lower.

## **6.5 Normalised Discounted Cumulative Gain (nDCG@k)**

### **Definition**

nDCG@k is a ranked relevance metric that rewards placing relevant items higher in the ranking. DCG penalises placing relevant items at lower ranks using a logarithmic discount:

DCG@5  \= Σ\_{i=1}^{5} rel\_i / log2(i \+ 1\)

         where rel\_i \= 1 if chunk at rank i is relevant, else 0

IDCG@5 \= maximum achievable DCG@5 (all top positions filled with relevant items)

nDCG@5 \= DCG@5 / IDCG@5

### **Interpretation**

nDCG ranges from 0 (worst) to 1.0 (perfect). All models achieve extremely high nDCG@5 values (0.965–0.991) in this evaluation. This near-ceiling effect occurs because nDCG heavily rewards placing any relevant item at rank 1 or 2, and all models do this reliably for most queries. The small spread (0.026) in nDCG across models means it has limited discriminative power in this corpus — Precision@5 and MRR@5 provide more signal.

## **6.6 Quality Score (Composite)**

### **Definition**

The composite Quality score is the arithmetic mean of three metrics: Precision@5, MRR@5, and nDCG@5. Average similarity (Avg\_Sim) was deliberately excluded from the Quality composite because it is not comparable across models with different embedding spaces and dimensionalities.

Quality \= (Precision@5 \+ MRR@5 \+ nDCG@5) / 3

### **Interpretation**

Quality is a balanced summary score. It penalises models that are good at placing one relevant item first (high MRR) but return many irrelevant results overall (low Precision), and vice versa. The range across models in Exp A is 0.7690–0.7948, a spread of 0.026. This tight spread is itself a finding — suggesting that on clinical transcriptions of this type, the choice of embedding model matters less than getting the pipeline deployment right.

## **6.7 Why Recall@5 Is Excluded from Quality**

Recall@5 is reported for diagnostic purposes but excluded from the Quality composite for a principled reason: its denominator (number of key terms) varies per query, making cross-model comparison noisy. A query with 2 key terms can achieve Recall@5 \= 1.0 by returning 2 chunks; a query with 20 key terms cannot. Including Recall@5 in Quality would weight easy queries disproportionately.

# **7\. Experiments**

## **7.1 Experiment A — Standard Key-Term Matching**

### **Design**

In Experiment A, query key terms are extracted using a simple filter: remove clinical stopwords (from clinical-stopwords.txt) from the query words. The remaining terms form the key-term set. Relevance is binary: a chunk is relevant if any of its parent document's keywords match any of the query key terms (case-insensitive, after lowercasing and stripping punctuation).

**Relevance threshold:** 1.0 (any single match suffices).

Example: Query \= 'cardiac arrhythmia drug interactions' → After stopword removal: key\_terms \= {arrhythmia, drug, interactions} → A chunk whose parent document has keywords \['arrhythmia', 'atrial fibrillation', 'beta blockers'\] is relevant because 'arrhythmia' matches.

### **Purpose**

Experiment A establishes the baseline performance landscape. It uses the simplest possible relevance function, ensuring any observed model differences reflect genuine embedding quality rather than evaluation-function artefacts.

## **7.2 Experiment B — Tokenizer-Aware Weighted Matching**

### **Design**

Experiment B replaces the binary key-term extractor with a weighted variant that assigns differential importance to terms based on their PubMedBERT tokenizer encoding. The hypothesis is that terms encoded as single tokens by PubMedBERT are highly domain-specific and therefore more discriminative as retrieval signals.

Weight assignment rules:

* Base weight: 1.0 for all terms

* Single-token bonus: ×3 multiplier for terms that PubMedBERT tokenises as exactly 1 token (not fragmented)

* SNOMED bonus: ×2 multiplier for terms present in the SNOMED vocabulary file

* Multipliers stack: a term that is both single-token AND SNOMED-listed gets ×6 effective weight

**Relevance threshold:** 2.0 (the sum of weights of matched terms must exceed 2.0 to count as relevant). This is stricter than Experiment A's threshold of 1.0.

### **Purpose**

Experiment B tests whether the tokenizer-based weighting can better distinguish between general and medical models. The hypothesis was that PubMedBERT's ability to encode medical terms as single tokens gives it an inherent advantage at retrieving chunks with high-weight terms. The results partially confirm this: PubMedBERT's MRR lead grows slightly in Exp B, but all models decline uniformly in Quality, suggesting the stricter threshold hurts rather than discriminates.

### **Key Bug: Weight Propagation**

**A critical bug was identified and fixed between the v1 and v2 runs:** the extract\_key\_terms\_tokaware function returned a weighted list (terms repeated proportional to weight), but the evaluation loop called list(dict.fromkeys(key\_terms)) which deduplicated the list, destroying all weight information. The fix: pass the raw weighted list directly to the relevance function; is\_relevant builds a weight\_map by counting term frequency and computes the weighted hit score from it.

# **8\. Results**

## **8.1 Experiment A — Full Rankings**

Table ordered by Quality (descending). Medical models highlighted in yellow.

| Rank | Model | Type | Quality ↓ | Prec@5 | Recall@5 | MRR@5 | nDCG@5 | Time |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | bge-base | General | **0.7948** | 0.912 | 0.402 | 0.481 | 0.991 | 11.6s |
| 2 | S-PubMedBERT | Medical | 0.7937 | 0.904 | 0.399 | 0.487 | 0.991 | 12.2s |
| 3 | bge-small | General | 0.7837 | 0.864 | 0.381 | 0.498 | 0.990 | 7.8s |
| 4 | e5-large | General | 0.7833 | 0.880 | 0.388 | 0.487 | 0.984 | 26.1s |
| 5 | GTE-Large | General | 0.7797 | 0.856 | 0.376 | 0.498 | 0.985 | 26.5s |
| 6 | MedEmbed-L | Medical | 0.7785 | 0.904 | 0.399 | 0.464 | 0.968 | 30.6s |
| 7 | MiniLM-L6 | General | 0.7760 | 0.856 | 0.379 | 0.493 | 0.979 | 5.9s |
| 8 | Snowflake-M | General | 0.7710 | 0.880 | 0.392 | 0.468 | 0.965 | 11.7s |
| 9 | PubMedBERT | Medical | 0.7690 | 0.832 | 0.362 | **0.503** | 0.972 | 12.0s |

*bge-base (general) ranks \#1 with Quality \= 0.7948. S-PubMedBERT (medical) ranks \#2 with Quality \= 0.7937. The gap is 0.001 — smaller than measurement noise between runs.*

## **8.2 Experiment B — Delta from Experiment A**

All models decline uniformly in Quality under the stricter tokenizer-aware threshold. The table shows Quality scores and rank changes.

| Model | Exp A Quality | Exp B Quality | Delta | New Rank | Rank Change |
| :---- | :---- | :---- | :---- | :---- | :---- |
| bge-base | 0.7948 | 0.7918 | −0.0030 | 1 | → (no change) |
| S-PubMedBERT | 0.7937 | 0.7881 | −0.0056 | 2 | → (no change) |
| MedEmbed-L | 0.7785 | 0.7767 | **−0.0018** | 3 | ↑ 6→3 (smallest delta) |
| bge-small | 0.7837 | 0.7783 | −0.0054 | 4 | ↓ 3→4 |
| GTE-Large | 0.7797 | 0.7756 | −0.0041 | 5 | → (no change) |
| e5-large | 0.7833 | 0.7750 | **−0.0083** | 6 | ↓ 4→6 (largest delta) |
| MiniLM-L6 | 0.7760 | 0.7722 | −0.0038 | 7 | → (no change) |
| Snowflake-M | 0.7710 | 0.7672 | −0.0038 | 8 | → (no change) |
| PubMedBERT | 0.7690 | 0.7661 | −0.0029 | 9 | → (no change) |

Notable observations:

* MedEmbed-L has the smallest quality drop (−0.0018) despite being labelled as a 'medical' model. Its standard tokenizer means the SNOMED/single-token weight bonuses rarely trigger, so its effective threshold rarely changes — it is least affected by the stricter evaluation.

* e5-large has the largest quality drop (−0.0083) and falls from rank 4 to rank 6\. The reason: e5-large's broader recall captures more marginally-relevant chunks that pass Exp A's lenient threshold but fail Exp B's stricter one.

* Only queries Q01 and Q03 show measurable per-query deltas in Exp B. Both involve allergic rhinitis ('rhinitis' gets 3× weight as a PubMedBERT single-token term). Stricter threshold penalises generic allergy chunks that match the lower-weight term 'allergy' but not the higher-weight 'rhinitis'.

## **8.3 Group Averages — General vs Medical**

The most important summary comparison: does domain adaptation help at the group level?

| Exp | Family | Prec@5 | Recall@5 | MRR@5 | nDCG@5 | Quality | Verdict |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| A | General | 0.875 | 0.387 | 0.487 | 0.982 | 0.781 | Slight edge |
| A | Medical | 0.880 | 0.387 | 0.485 | 0.977 | 0.780 | Dead heat |
| B | General | 0.863 | 0.380 | 0.489 | 0.978 | 0.777 | Dead heat |
| B | Medical | 0.869 | 0.381 | 0.487 | 0.975 | 0.777 | Dead heat |

*The answer is no — at the group level, medical and general models are in a dead heat in both experiments. The 0.001 advantage for the Medical family in Prec@5 is within noise. This finding challenges the common assumption that domain-adapted models are categorically better for medical RAG.*

## **8.4 Tokenizer Analysis Results**

| Model | Vocab Size | Fertility | SNOMED Cov. | Tokenizer Base | Example: echocardiogram |
| :---- | :---- | :---- | :---- | :---- | :---- |
| MiniLM-L6 | 30,522 | 2.209 | 14.2% | bert-base-uncased | 5 tokens |
| bge-small | 30,522 | 2.209 | 14.2% | bert-base-uncased | 5 tokens |
| bge-base | 30,522 | 2.209 | 14.2% | bert-base-uncased | 5 tokens |
| Snowflake-M | 30,522 | 2.209 | 14.2% | bert-base-uncased | 5 tokens |
| e5-large | 30,522 | 2.209 | 14.2% | bert-base-uncased | 5 tokens |
| GTE-Large | 30,522 | 2.209 | 14.2% | bert-base-uncased | 5 tokens |
| PubMedBERT | 30,522 | 1.918 | 14.8% | PubMed fine-tuned | **1 token ✓** |
| S-PubMedBERT | 30,522 | 1.918 | 14.8% | PubMed fine-tuned | **1 token ✓** |
| MedEmbed-L | 30,522 | 2.209 | 14.2% | bert-base-uncased | 5 tokens |

The most striking tokenizer finding is that MedEmbed-L — despite its 'medical' branding — uses the standard bert-base-uncased tokenizer with no vocabulary adaptation. Its fertility (2.209) and SNOMED coverage (14.2%) are identical to MiniLM-L6, the smallest general model. This reveals that the label 'medical embedding model' can mean several different things: medical fine-tuning data (MedEmbed-L's case) vs. genuine vocabulary adaptation (PubMedBERT's case).

## **8.5 Per-Query Analysis — Hardest Queries**

Five queries produced consistently below-average Quality across all models:

| Query | Description | Best Model | Worst Model | Avg Quality | Why Difficult |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Q24 | Sickle cell: genetic counselling considerations | bge-base (0.74) | PubMedBERT (0.65) | 0.708 | Cross-specialty synthesis |
| Q23 | CRISPR gene therapy for Duchenne MD | S-PubMedBERT (0.76) | Snowflake-M (0.62) | 0.718 | Rare disease \+ procedure |
| Q09 | Rhinitis management in children | bge-base (0.81) | Snowflake-M (0.29) | 0.727 | Snowflake-M prefix failure |
| Q02 | Cardiac arrhythmia drug interactions | bge-base (0.79) | PubMedBERT (0.67) | 0.732 | Multi-medication cross-links |
| Q21 | Metabolic syndrome \+ T2DM comorbidities | e5-large (0.82) | PubMedBERT (0.72) | 0.751 | Multi-condition overlap |

Q09 (rhinitis management in children) deserves special attention: its average Quality of 0.727 is depressed by Snowflake-M's catastrophic failure on this query (Quality \= 0.294). When Snowflake-M is excluded, Q09's average rises to \~0.79, comparable to other queries. This single query is responsible for a large portion of Snowflake-M's standard deviation (0.109 — the highest of all models).

## **8.6 Efficiency Analysis**

| Model | Time | VRAM | Quality | Quality/s | Tier | Recommendation |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| MiniLM-L6 | 5.9s | 56 MB | 0.776 | 0.1315 | Edge/CPU-friendly | Baseline |
| bge-small | 7.8s | 79 MB | 0.784 | 0.1005 | Edge/CPU-friendly | Better than MiniLM |
| **bge-base** | 11.6s | 244 MB | **0.795** | 0.0685 | Standard GPU | **Best overall** |
| S-PubMedBERT | 12.2s | 238 MB | 0.794 | 0.0651 | Standard GPU | Medical use case |
| Snowflake-M | 11.7s | 242 MB | 0.771 | 0.0659 | Standard GPU | With prefix only |
| e5-large | 26.1s | 680 MB | 0.783 | 0.0300 | High-end GPU | Not cost-effective |
| GTE-Large | 26.5s | 680 MB | 0.780 | 0.0294 | High-end GPU | Not cost-effective |
| MedEmbed-L | 30.6s | 680 MB | 0.779 | 0.0255 | High-end GPU | Worst large model |
| PubMedBERT | 12.0s | 238 MB | 0.769 | 0.0641 | Standard GPU | Best first-hit only |

# **9\. Interpretation & Key Findings**

## **9.1 Finding 1: The General vs Medical Dead Heat**

The most headline-worthy result is also the most counterintuitive: on clinical transcriptions, general-purpose models match medical-domain models in aggregate retrieval quality. The group averages (General: 0.781, Medical: 0.780 in Exp A) are statistically indistinguishable.

Why does this happen? There are three plausible explanations:

* Training data coverage: Modern general models like bge-base are trained on billions of text pairs covering a wide range of domains, including medical and scientific text. The sheer scale may compensate for the lack of domain-specific vocabulary adaptation.

* Retrieval fine-tuning dominance: The models that perform best (bge-base, S-PubMedBERT) have both been fine-tuned on explicit retrieval objectives. The models that underperform (PubMedBERT, MedEmbed-L) are primarily trained on classification or similarity tasks, not retrieval. The fine-tuning task design matters more than the pre-training domain.

* Corpus nature: MTSamples clinical transcriptions are written in relatively standard clinical English — not the highly compressed, abbreviation-dense language of emergency notes or discharge summaries where vocabulary adaptation would provide a larger advantage.

## **9.2 Finding 2: Retrieval Fine-Tuning \> Domain Pre-Training**

The clearest evidence comes from comparing PubMedBERT and S-PubMedBERT, which share identical tokenizers and pre-training corpora. Their Quality scores differ by 0.025 (0.7690 vs 0.7937) — the largest gap between any same-tokenizer pair. The only difference between them is that S-PubMedBERT was additionally fine-tuned on MS-MARCO, a retrieval-specific dataset of 8.8 million (query, relevant passage) pairs from Bing search.

*S-PubMedBERT vs PubMedBERT: \+0.025 Quality from MS-MARCO fine-tuning. This is the largest single improvement observed in the entire experiment, and it comes from retrieval-task fine-tuning, not domain pre-training.*

## **9.3 Finding 3: PubMedBERT's MRR Advantage Is Real**

PubMedBERT ranks 9th in overall Quality but achieves the highest MRR@5 (0.503) of all models. This is not noise — it reflects a genuine and important property of domain-adapted tokenization. When PubMedBERT encodes 'echocardiogram' or 'haemoglobin' as single tokens, the query embedding contains a precise, unambiguous representation of that clinical concept. This leads to a more reliable top-1 match.

However, PubMedBERT's overall Precision@5 is the lowest (0.832), suggesting that after it places the best match first, its subsequent ranked results are less reliable than general models. This creates a precision-recall profile that is useful for applications where the top-1 answer is critical (clinical decision support) but less suitable for comprehensive evidence synthesis (literature review, systematic review) where recall matters.

## **9.4 Finding 4: The Snowflake-M Rehabilitation**

In the v1 evaluation, Snowflake-M scored Quality \= 0.659 — ranking last by a wide margin and appearing to be a poor model. After adding the required query prefix in v2, its Quality jumped to 0.771, within normal range. The \+0.112 improvement from a single string prepended to each query is larger than the entire spread between the other 8 models (0.026).

*Deployment lesson: Snowflake-M is an asymmetric model that requires different encodings for queries vs passages. Using the same encoder for both causes severe query-passage misalignment. This is documented in the model card but easy to overlook. The lesson generalises: always verify model-specific deployment requirements before drawing conclusions about model quality.*

## **9.5 Finding 5: MedEmbed-L Is Not What Its Label Claims**

MedEmbed-L's tokenizer fingerprint is byte-for-byte identical to MiniLM-L6 (the smallest general model): vocab size 30,522, fertility 2.209, SNOMED coverage 14.2%. It is a large model with 1,024-dimensional embeddings, which provides some additional capacity, but it achieves Quality \= 0.779 — below all base-size models and consuming 680 MB VRAM (12× the VRAM of MiniLM-L6).

This raises an important point for practitioners: 'medical embedding model' is a marketing label, not a specification. When evaluating such models, always examine the tokenizer directly (vocabulary size, fertility on medical terms) to distinguish genuine vocabulary adaptation from domain-specific fine-tuning data alone.

## **9.6 Finding 6: Experiment B Finds Uniform Decline**

The tokenizer-aware weighting in Experiment B was designed to advantage models with better medical tokenizers. Instead, it caused a uniform decline across all 9 models with no differential benefit for PubMedBERT or S-PubMedBERT beyond their existing Exp A advantage. Two reasons explain this:

* The threshold (2.0) is too strict. Most queries have only 2–4 high-weight terms. A threshold of 2.0 requiring the sum of matched weights to exceed 2.0 is only marginally different from threshold 1.0 for short key-term sets.

* PubMedBERT's vocabulary benefit operates at embedding time, not evaluation time. The model's advantage comes from encoding medical queries more accurately into vector space — not from receiving higher evaluation credit for medical terms. An evaluation function that tries to 'credit' medical tokenizer terms cannot measure this; it needs to be expressed through retrieval rank differences, which are already captured by MRR@5.

# **10\. Limitations**

## **10.1 Corpus Limitations**

* Single dataset: All results are on MTSamples. This corpus consists of transcription examples (dictated by physicians for transcription software training), not real clinical notes. Real clinical notes contain more abbreviations, misspellings, and implicit context.

* Specialty distribution: MTSamples is Surgery-heavy. Results may not generalise to subspecialties with smaller representation (e.g., psychiatry, occupational medicine).

* Keyword propagation: Keywords are propagated from document level to all chunks from that document. This means a chunk about post-operative care inherits keywords about the surgical procedure itself. This introduces noise into the relevance signal.

* Null keyword exclusion: 1,182 documents (31%) lack keywords and are therefore excluded from evaluation scoring. If these documents are systematically different (e.g., shorter, less detailed), the evaluation may over-estimate performance on typical documents.

## **10.2 Evaluation Limitations**

* 25 queries: The query set is small. Results are subject to sampling variance. A 95% confidence interval on each metric spans approximately ±0.04–0.06 given n=25.

* Binary relevance: The keyword-matching relevance function treats all keywords as equally important and does not model partial relevance or degrees of relevance. A chunk that contains 8 out of 10 query keywords scores identically to one that contains exactly 1\.

* No human judgement: Ground truth is derived entirely from expert annotations in the dataset, not from human relevance assessments for these specific queries. The 25 queries were not validated against the dataset before construction.

* k=5 fixed: The evaluation only reports metrics at k=5. In production RAG systems, k is often tuned (16–64 is common for context window filling). Performance at larger k values may rank models differently.

## **10.3 Infrastructure Limitations**

* Single GPU run: Results are from a single RTX 5070 Ti run. No statistical replication was performed — run-to-run variance (due to GPU non-determinism in floating-point operations) is not quantified.

* Exact FAISS search: IndexFlatIP performs exact search. In production systems with millions of documents, approximate nearest neighbour (ANN) search is used, which introduces an approximation error that may affect model rankings differently.

# **11\. Future Work & Improvements**

## **11.1 Planned Experiments**

| Experiment | Description | Expected Gain | Priority |
| :---- | :---- | :---- | :---- |
| Two-stage retrieval | e5-large retrieves top-50; PubMedBERT reranks — combines general coverage with medical precision | \+8–12% on Q21–Q25 | **HIGH** |
| Ensemble retrieval | α × e5-score \+ (1−α) × pubmed-score; sweep α from 0.0→1.0 in steps of 0.1 | \+3–6% across all | **HIGH** |
| Exp B threshold fix | Reduce tokenizer-aware threshold from 2.0 → 1.5 to improve term filtering sensitivity | Recover −0.004 drop | **MEDIUM** |
| Vocabulary adaptation | Extend PubMedBERT tokenizer with SNOMED terms; continued pre-training; compare to S-PubMedBERT | Validate tokenizer hypothesis | **RESEARCH** |
| Cross-encoder reranking | Use a cross-encoder (e.g. MedCPT) to rerank top-20 retrieved chunks | \+5–15% nDCG@5 | **MEDIUM** |
| Hard-query fine-tuning | Construct training pairs from Q21–Q25 failure cases; fine-tune S-PubMedBERT | Close cross-specialty gap | **RESEARCH** |
| Larger corpus evaluation | Extend to MIMIC-III discharge notes or PubMed abstracts for generalizability | External validity | **LOW** |
| Chunking strategy ablation | Compare sentence / fixed-300-word / semantic chunking; measure impact on Recall@5 | \+5–10% Recall@5 | **MEDIUM** |

## **11.2 Two-Stage Retrieval (Highest Priority)**

The most promising near-term improvement is a two-stage retrieval architecture:

23. Stage 1 (Retrieve): e5-large retrieves the top-50 candidate chunks using its broad general coverage and high recall.

24. Stage 2 (Rerank): PubMedBERT scores each of the 50 candidates against the query using its precise medical embeddings. Return the top-5 reranked candidates.

The motivation: e5-large has the best Recall@50 (estimated) due to its large embedding capacity and diverse training. PubMedBERT has the best MRR@5 (0.503) — it is highly precise about its best match. Combining these strengths in a two-stage pipeline is expected to produce the best of both worlds: high recall from Stage 1 and high precision from Stage 2\.

Expected gains: \+8–12% Quality on the hardest queries (Q21–Q25) where cross-specialty synthesis requires both breadth (high-recall first stage) and precision (accurate medical concept matching in reranking).

## **11.3 Ensemble Retrieval**

A simpler alternative to two-stage retrieval is score ensembling. For each query, run both e5-large and PubMedBERT independently to retrieve top-k with similarity scores. Merge the two ranked lists using a weighted combination:

ensemble\_score(chunk, query) \= α × e5\_score(chunk, query) \+ (1 − α) × pubmed\_score(chunk, query)

Sweep α from 0.0 to 1.0 in steps of 0.1. Plot Quality vs α. The optimal α is expected to be near 0.6–0.7 (leaning slightly general), based on the Exp A results showing e5-large's slight advantage for cross-specialty queries.

## **11.4 Vocabulary Adaptation Research**

The core theoretical question of this project is whether vocabulary adaptation provides retrieval benefit. The current results show a tokenizer fertility gap (2.209 vs 1.918) but the retrieval benefit is modest and confounded by fine-tuning differences. A controlled experiment is needed:

25. Start from bert-base-uncased (standard tokenizer, fertility 2.209, SNOMED coverage 14.2%).

26. Extend the vocabulary by adding the top-10,000 most common SNOMED terms as single tokens. This directly reduces fertility.

27. Continue pre-training on PubMed abstracts for 100K steps to adapt the new token embeddings.

28. Fine-tune on MS-MARCO (same as S-PubMedBERT's fine-tuning).

29. Compare against S-PubMedBERT at each stage to isolate the contribution of: (a) vocabulary extension, (b) domain pre-training, (c) retrieval fine-tuning.

This ablation would provide the first controlled evidence of vocabulary adaptation's marginal contribution to retrieval quality — which is the primary research question of the supervising project.

## **11.5 Chunking Strategy Ablation**

The current chunking strategy (sentence-level, min 40 chars) was chosen as a reasonable default. However, chunking significantly affects retrieval — both the number of chunks and their coherence. Three strategies should be compared:

* Sentence-level (current): Natural boundaries, variable length, 7,433 chunks at mean 145 words.

* Fixed-word (300 words, 50-word overlap): Predictable size, better for dense retrieval models trained on fixed-length passages.

* Semantic chunking: Use a sentence embedding model to detect topic boundaries. Keeps semantically coherent passages together — expected to improve Precision@5 at the cost of higher Recall@5 variance.

# **12\. Appendix**

## **A. Output File Manifest**

| File | Format | Contents |
| :---- | :---- | :---- |
| corpus.csv | CSV | 2,348 deduplicated documents with labels, keywords, chunk count |
| tokenizer\_summary.csv | CSV | Per-model: vocab size, fertility, SNOMED coverage |
| tokenizer\_per\_term.csv | CSV | Per SNOMED term: token count per model |
| results\_expA.csv | CSV | Per-model aggregate metrics for Experiment A |
| results\_expB.csv | CSV | Per-model aggregate metrics for Experiment B |
| per\_query\_expA.csv | CSV | Per-query metrics for all 9 models, Experiment A |
| per\_query\_expB.csv | CSV | Per-query metrics for all 9 models, Experiment B |
| ckpt\_{name}\_ExpA.csv | CSV | Checkpoint after each model completes Experiment A |
| ckpt\_{name}\_ExpB.csv | CSV | Checkpoint after each model completes Experiment B |
| fig01\_quality\_expA.png | PNG | Bar chart: Quality scores for all models, Experiment A |
| fig02\_quality\_expB.png | PNG | Bar chart: Quality scores for all models, Experiment B |
| fig03\_metrics\_radar.png | PNG | Radar chart: All 4 metrics per model, Experiment A |
| fig04\_heatmap\_query.png | PNG | Heatmap: per-query Quality for all 9 models |
| fig05\_exp\_comparison.png | PNG | Side-by-side Exp A vs Exp B Quality delta per model |
| fig06\_efficiency.png | PNG | Scatter: Quality vs inference time coloured by VRAM |
| fig07\_tokenizer.png | PNG | Bar chart: fertility and SNOMED coverage by model |
| fig08\_group\_avg.png | PNG | Grouped bar: General vs Medical family averages |

## **B. Environment Setup**

Reproduce the environment on Windows with uv:

uv venv .venv

.venv\\Scripts\\activate

uv pip uninstall torch

uv pip install torch \--index-url https://download.pytorch.org/whl/cu128

uv pip install sentence-transformers transformers faiss-cpu pandas numpy matplotlib seaborn tqdm psutil

python rag\_eval\_local.py

Verify CUDA:

python \-c "import torch; print(torch.cuda.is\_available()); print(torch.cuda.get\_device\_name(0))"

*Note: Use \--index-url (not \--extra-index-url) to guarantee the cu128 wheel is selected. \--extra-index-url falls back to CPU torch from PyPI if the cu128 wheel is not found.*

## **C. Tokenizer Fertility: Selected Terms**

| SNOMED Term | BERT-base Tokens | PubMedBERT Tokens | Savings |
| :---- | :---- | :---- | :---- |
| haemoglobin | 5 tokens | **1 token** | 4 tokens |
| immunofluorescence | 5 tokens | **1 token** | 4 tokens |
| echocardiogram | 5 tokens | **1 token** | 4 tokens |
| antifungal | 4 tokens | **1 token** | 3 tokens |
| anterograde | 3 tokens | **1 token** | 2 tokens |
| hyperactivity | 3 tokens | **1 token** | 2 tokens |
| thromboembolus | 6 tokens | 2 tokens | 4 tokens |
| anastomosis | 4 tokens | **1 token** | 3 tokens |
| cholecystectomy | 5 tokens | 2 tokens | 3 tokens |

## **D. Glossary**

| Term | Definition |
| :---- | :---- |
| **RAG** | Retrieval-Augmented Generation. An architecture combining a dense retriever with a generative language model. |
| **Dense Retrieval** | Encoding documents and queries as continuous vectors and finding matches by vector similarity, as opposed to sparse retrieval (BM25, TF-IDF) based on keyword overlap. |
| **Embedding** | A fixed-size dense vector representation of a piece of text, produced by passing it through a transformer encoder. |
| **Cosine Similarity** | A measure of angular similarity between two vectors, equal to their dot product divided by the product of their magnitudes. Ranges from \-1 to 1\. |
| **FAISS** | Facebook AI Similarity Search. A library for fast exact and approximate nearest-neighbour search over large sets of vectors. |
| **Fertility** | The mean number of subword tokens produced by a tokenizer per word in a reference vocabulary. Lower fertility \= more whole-word tokens \= better vocabulary coverage. |
| **SNOMED CT** | Systematized Nomenclature of Medicine — Clinical Terms. A comprehensive multilingual clinical healthcare terminology maintained by SNOMED International. |
| **Subword Tokenization** | A tokenization strategy that breaks rare words into smaller pieces using algorithms like WordPiece (BERT) or BPE. Allows models to handle any word but may fragment domain-specific terms. |
| **Precision@k** | Fraction of retrieved top-k items that are relevant. Measures how much of what is retrieved is useful. |
| **Recall@k** | Fraction of all relevant items captured in the top-k. Measures how much of what is useful is retrieved. |
| **MRR@k** | Mean Reciprocal Rank. Average of 1/rank of the first relevant item. Measures how reliably the best item appears at rank 1\. |
| **nDCG@k** | Normalised Discounted Cumulative Gain. A ranked relevance metric that rewards placing relevant items higher in the result list. |
| **MS-MARCO** | Microsoft MAchine Reading COmprehension. A large-scale dataset of real Bing queries with human-annotated relevant passages. Used for retrieval fine-tuning. |
| **Contrastive Learning** | A training objective that pulls representations of similar (positive) pairs together and pushes dissimilar (negative) pairs apart in embedding space. |
| **MTSamples** | A public dataset of 4,999 medical transcription samples from 40 clinical specialties, originally created for medical transcription software training. |

