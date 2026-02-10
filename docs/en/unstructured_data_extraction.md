---
layout: docs
header: true
seotitle: Unstructured Data Extraction
title: Unstructured Data Extraction
permalink: /docs/en/unstructured_data_extraction
key: docs-concepts
modify_date: "2026-02-09"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

Spark NLP provides comprehensive capabilities for extracting and processing unstructured data from various document formats at enterprise scale using its `Reader2X` components and related annotators handle common document AI tasks, with comparisons to other frameworks.

</div><div class="h3-box" markdown="1">

### Complete Text Coverage from Complex Documents

#### Problem
Enterprise pipelines require extracting **every piece of visible text** from documents, including navigation menus, footers, captions, tables, figure titles, and metadata fields. Capturing all visible text is essential for traceable, auditable corpora where any omission could lead to information loss or compliance gaps.

</div><div class="h3-box" markdown="1">

#### Spark NLP Solution
To clean text extracted from HTML using Spark NLP, we leveraged the following annotators:

```python
from sparknlp.reader.reader2doc import Reader2Doc
from sparknlp.annotator import DocumentNormalizer, SentenceDetectorDLModel
from pyspark.ml import Pipeline

reader2doc = Reader2Doc() \
    .setContentType('text/html') \
    .setContentPath(directory) \
    .setOutputCol('document')

normalizer = DocumentNormalizer() \
    .setInputCols(['document']) \
    .setOutputCol('normalized') \
    .setAutoMode("HTML_CLEAN") \
    .setPatterns([(":")])

sentence_detector = SentenceDetectorDLModel() \
    .pretrained() \
    .setInputCols(['normalized']) \
    .setOutputCol('sentences') \
    .setExplodeSentences(True)

pipeline = Pipeline(stages=[reader2doc, normalizer, sentence_detector])

model = pipeline.fit(empty_df)
result_df = model.transform(empty_df)
```

</div><div class="h3-box" markdown="1">

#### Benefits

{:.list1}
- **Complete coverage**: Extracts the full visible text layer without filtering
- **Scalable processing**: Built on Apache Spark for distributed processing
- **Unified pipeline**: Flows directly into tokenizers, embeddings, and NLP models
- **Traceability**: Maintains metadata (source path, page number, character offsets)

**Use Cases**: Enterprise-scale ingestion, full-text indexing, document alignment, compliance auditing

</div><div class="h3-box" markdown="1">

### Maintaining Structural Context for Data-Rich Documents

#### Problem
In healthcare, finance, insurance, and legal domains, critical insights are embedded in **structured elements** like tables and figures. Without preservation of structural context (headers, captions, section hierarchy), downstream NLP systems struggle to interpret the extracted information.

</div><div class="h3-box" markdown="1">

#### Spark NLP Solution

```python
from sparknlp.reader.reader2table import Reader2Table

reader2doc = Reader2Table() \
    .setContentType('text/html') \
    .setContentPath('html_docs/EHR-2025-12-000002.html') \
    .setOutputCol('table') \
    .setExplodeDocs(True)

pipeline = Pipeline(stages=[reader2doc])

model = pipeline.fit(empty_df)
result_df = model.transform(empty_df)
```

</div><div class="h3-box" markdown="1">

#### JSON Output (structured data)

```
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|result                                                                                                                                                                                                                                                                                                                                                                                                                                         |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[{"caption":"","header":["Test","Result","Units","Reference Range","Status"],"rows":[["PSA","0.32","ng/mL","0-4.0","Excellent"],["Testosterone","125","ng/dL","300-1000","Recovering"],["Hemoglobin","14.3","g/dL","13.5-17.5","Normal"],["WBC","7.2","K/uL","4.5-11.0","Normal"],["Creatinine","0.9","mg/dL","0.7-1.3","Normal"],["ALT","22","U/L","7-56","Normal"]]}]                                                                        |
|[{"caption":"","header":["Test","Result","Units","Reference Range","Status"],"rows":[["Testosterone","105","ng/dL","300-1000","Recovering"],["Hemoglobin","12.3","g/dL","13.5-17.5","Normal"],["Creatinine","0.7","mg/dL","0.7-1.3","Normal"]]}]                                                                                                                                                                                               |
|[{"caption":"","header":["Medication","Dose","Frequency","Indication","Status"],"rows":[["Atorvastatin (Lipitor)","10 mg PO","Daily","Hyperlipidemia","Active"],["Aspirin","81 mg PO","Daily","Cardiovascular prophylaxis","Active"],["Vitamin D3","2000 IU PO","Daily","Bone health","Active"],["Calcium carbonate","500 mg PO","BID","Bone health (post-ADT)","Active"],["Multivitamin","1 tab PO","Daily","Nutritional support","Active"]]}]|
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

</div><div class="h3-box" markdown="1">

#### HTML Output (for rendering)

```python
reader2doc = Reader2Table() \
    .setContentType('text/html') \
    .setContentPath('html_docs/EHR-2025-12-000002.html') \
    .setOutputCol('table') \
    .setOutputFormat('html-table') \
    .setExplodeDocs(True)
```

```
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[<table class="lab-table"><thead><tr><th>Test</th><th>Result</th><th>Units</th><th>Reference Range</th><th>Status</th></tr></thead><tbody><tr><td>PSA</td><td><strong>0.32</strong></td><td>ng/mL</td><td>0-4.0</td><td><span class="status-badge status-active">Excellent</span></td></tr><tr><td>Testosterone</td><td><strong>125</strong></td><td>ng/dL</td><td>300-1000</td><td><span class="status-badge status-completed">Recovering</span></td></tr><tr><td>Hemoglobin</td><td><strong>14.3</strong></td><td>g/dL</td><td>13.5-17.5</td><td><span class="status-badge status-active">Normal</span></td></tr><tr><td>WBC</td><td><strong>7.2</strong></td><td>K/uL</td><td>4.5-11.0</td><td><span class="status-badge status-active">Normal</span></td></tr><tr><td>Creatinine</td><td><strong>0.9</strong></td><td>mg/dL</td><td>0.7-1.3</td><td><span class="status-badge status-active">Normal</span></td></tr><tr><td>ALT</td><td><strong>22</strong></td><td>U/L</td><td>7-56</td><td><span class="status-badge status-active">Normal</span></td></tr></tbody></table>]|
|[<table class="lab-table"><thead><tr><th>Test</th><th>Result</th><th>Units</th><th>Reference Range</th><th>Status</th></tr></thead><tbody><tr><td>Testosterone</td><td><strong>105</strong></td><td>ng/dL</td><td>300-1000</td><td><span class="status-badge status-completed">Recovering</span></td></tr><tr><td>Hemoglobin</td><td><strong>12.3</strong></td><td>g/dL</td><td>13.5-17.5</td><td><span class="status-badge status-active">Normal</span></td></tr><tr><td>Creatinine</td><td><strong>0.7</strong></td><td>mg/dL</td><td>0.7-1.3</td><td><span class="status-badge status-active">Normal</span></td></tr></tbody></table>]                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|[<table class="lab-table"><thead><tr><th>Medication</th><th>Dose</th><th>Frequency</th><th>Indication</th><th>Status</th></tr></thead><tbody><tr><td>Atorvastatin (Lipitor)</td><td>10 mg PO</td><td>Daily</td><td>Hyperlipidemia</td><td><span class="status-badge status-active">Active</span></td></tr><tr><td>Aspirin</td><td>81 mg PO</td><td>Daily</td><td>Cardiovascular prophylaxis</td><td><span class="status-badge status-active">Active</span></td></tr><tr><td>Vitamin D3</td><td>2000 IU PO</td><td>Daily</td><td>Bone health</td><td><span class="status-badge status-active">Active</span></td></tr><tr><td>Calcium carbonate</td><td>500 mg PO</td><td>BID</td><td>Bone health (post-ADT)</td><td><span class="status-badge status-active">Active</span></td></tr><tr><td>Multivitamin</td><td>1 tab PO</td><td>Daily</td><td>Nutritional support</td><td><span class="status-badge status-active">Active</span></td></tr></tbody></table>]                                                                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```
</div><div class="h3-box" markdown="1">

#### Metadata Output

```
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|metadata                                                                                                                                                                                                 |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[orderTableIndex -> 1, nearestHeader -> 🔬 Most Recent Laboratory Results (10/22/2016), pageNumber -> 1, domPath -> /html[1]/body[1]/div[1]/div[3]/div[4]/table[1], elementType -> Table, sentence -> 8}]|
|[orderTableIndex -> 2, nearestHeader -> History Laboratory Results (10/22/2016), pageNumber -> 1, domPath -> /html[1]/body[1]/div[1]/div[3]/div[4]/table[2], elementType -> Table, sentence -> 10}]      |
|[orderTableIndex -> 1, nearestHeader -> 💊 Current Medications, pageNumber -> 1, domPath -> /html[1]/body[1]/div[1]/div[3]/div[5]/table[1], elementType -> Table, sentence -> 12}]                       |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

</div><div class="h3-box" markdown="1">

#### Benefits

This output captures rich structural information alongside extracted content:

- **DOM paths** (e.g., `/html[1]/body[1]/div[3]/div[5]/table[1]`) that identify exactly where in the HTML document a table or image came from.
- **Nearest section header context** so that a table is semantically linked to its surrounding narrative (“Laboratory Results”, “Current Medications”, etc.).
- **Order and hierarchy metadata** such as `orderTableIndex`, allowing precise reconstruction of document structure.
- A structured **JSON** representation of tables (with headers, rows, captions, and field metadata)
- A **HTML** representation for visualization, rendering, or further processing.

**Use Cases**: These enriched representations help downstream NLP tasks such as:

- **Table-aware question answering**: Models like TAPAS leverage structured table data to answer natural language questions over tables with high accuracy, something that plain text extraction alone cannot support.
- **Contextual table interpretation**: Structural metadata enables models to understand why a table occurs where it does, improving joint inference between narrative text and tabular data, which is known to boost extraction quality when the context is considered.
- **Semantic integration with knowledge graphs and IE systems**: By preserving layout and section cues, extracted table data can be merged into structured knowledge representations with clear provenance.

</div><div class="h3-box" markdown="1">

### Processing Millions of Documents Efficiently

#### Problem
Modern organizations must process **massive volumes of unstructured documents**, including PDFs, HTML pages, contracts, medical records, and regulatory filings. These documents often number in the millions and arrive continuously through ingestion pipelines and compliance workflows.

**Why traditional approaches fail**

*Single-machine or sequential processing* does not scale with growing data volumes. Processing times increase as files are handled one by one, pipelines become fragile and fail mid-run, infrastructure costs rise due to inefficient resource usage, and NLP workflows become harder to scale as *tokenization, NER, and classification* are added.

For data engineering teams supporting **real-time compliance, analytics, and document intelligence**, this is no longer a minor inefficiency. It becomes a **core scalability bottleneck** that limits reliability and impact.

</div><div class="h3-box" markdown="1">

#### Spark NLP Solution
**Spark NLP is built natively on Apache Spark**, bringing distributed data processing to text analytics and NLP workloads. This enables *text extraction, normalization, and NLP tasks* to run in parallel across clusters, allowing **millions of documents** to be processed efficiently, reproducibly, and at scale.

```python
from sparknlp.reader.reader_assembler import ReaderAssembler

# Ingest all files in directory using ReaderAssembler
reader_assembler = ReaderAssembler() \
    .setContentPath(directory) \
    .setOutputCol("document")

pipeline = Pipeline(stages=[reader_assembler])
model = pipeline.fit(empty_df)

# Process and save as Parquet
df = model.transform(empty_df)
df.select("document_text.result").write.mode("overwrite").parquet(output)
```

</div><div class="h3-box" markdown="1">

#### Benchmark Results

Processing **60 mixed-format documents** on single machine:

{:.list1}
- Spark NLP achieved **~2× faster throughput** than sequential processing
- Automatic parallelization across all available CPU cores
- Same pipeline scales to Spark clusters with linear scalability

</div><div class="h3-box" markdown="1">

#### Benefits

{:.list1}
- **Scalable architecture**: Workloads partitioned across Spark executors
- **Fault tolerance**: Automatic checkpointing and resilient distributed datasets (RDDs)
- **Unified pipeline integration**: Ingestion, extraction, tokenization, and NLP in single Spark job
- **Operational efficiency**: Designed for terabytes of daily data

**Use Cases**: Enterprise document ingestion, batch processing, compliance workflows, large-scale ETL

</div><div class="h3-box" markdown="1">

### Document Format Support

Spark NLP provides comprehensive support for common document formats through its `Reader2X` components.

</div><div class="h3-box" markdown="1">

#### Supported Formats

{:.table-model-big}
| Format | Spark NLP Components | Description |
|--------|---------------------|-------------|
| **PDF** | Reader2Doc, Reader2Table, Reader2Image | Extract text and images, handles complex layouts |
| **HTML** | Reader2Doc, Reader2Table, Reader2Image | Parse structure, extract tables, preserve DOM context |
| **DOCX** | Reader2Doc, Reader2Table, Reader2Image | Support text, tables, and images from Word documents |
| **PPTX** | Reader2Doc, Reader2Table, Reader2Image | Extract slide content, notes, tables, and images |
| **XLSX** | Reader2Doc, Reader2Table, Reader2Image | Parse spreadsheets, extract structured data |
| **CSV** | Reader2Doc, Reader2Table | Read tabular data with proper schema |
| **Email (MSG, EML)** | Reader2Doc, Reader2Table, Reader2Image | Parse headers, body, and attachments |
| **XML** | Reader2Doc, Reader2Table, Reader2Image | Preserve structure, control tag handling |
| **Markdown** | Reader2Doc, Reader2Table, Reader2Image | Parse text and embedded images |
| **Plain Text** | Reader2Doc | Simple text ingestion |

</div><div class="h3-box" markdown="1">

### Data Preparation and Cleaning

Spark NLP's `DocumentNormalizer` provides powerful text cleaning and normalization capabilities that scale to large datasets.

</div><div class="h3-box" markdown="1">

#### Encoding Conversion

```python
normalizer = DocumentNormalizer() \
    .setInputCols(['document']) \
    .setOutputCol('normalized') \
    .setEncoding("UTF-8")
```

{:.list1}
- Converts byte strings into text strings
- Fully Spark-native for distributed processing
- Integrates into NLP pipelines via DocumentNormalizer

</div><div class="h3-box" markdown="1">

#### Remove Non-ASCII Characters

```python
normalizer = DocumentNormalizer() \
    .setInputCols(['document']) \
    .setOutputCol('normalized') \
    .setPresetPattern('CLEAN_NON_ASCII')
    # OR
    # .setAutoMode('HTML_CLEAN')
```

</div><div class="h3-box" markdown="1">

#### Clean Bullets and Dashes

```python
normalizer = DocumentNormalizer() \
    .setInputCols(['document']) \
    .setOutputCol('normalized') \
    .setPresetPattern('CLEAN_BULLETS') \
    # OR
    # .setAutoMode('DOCUMENT_CLEAN')
```

</div><div class="h3-box" markdown="1">

#### Clean Ordered Bullets

```python
normalizer = DocumentNormalizer() \
    .setInputCols(['document']) \
    .setOutputCol('normalized') \
    .setPresetPattern('CLEAN_ORDERED_BULLETS') \
    # OR
    # .setAutoMode('DOCUMENT_CLEAN')
```

{:.list1}
- Explicit support for ordered list bullets (1., 2., a., b., etc.)
- Implemented via preset patterns
- Can be composed with other document cleaners
- Clear semantics for removing enumerated list markers

</div><div class="h3-box" markdown="1">

#### Remove Punctuation

```python
normalizer = DocumentNormalizer() \
    .setInputCols(['document']) \
    .setOutputCol('normalized') \
    .setPresetPattern('REMOVE_PUNCTUATION') \
    # OR
    # .setAutoMode('SOCIAL_CLEAN')
```

</div><div class="h3-box" markdown="1">

#### Custom Pattern Cleaning

```python
# Remove specific prefix patterns
normalizer = DocumentNormalizer() \
    .setPatterns(Array("(?i)^(SUMMARY|DESCRIPTION):")) \
    .setAction("clean") \
    .setReplacement(" ") \
    .setPolicy("pretty_all")

# Remove postfix patterns
normalizer = DocumentNormalizer() \
    .setPatterns(Array("(?i)(END|STOP)$")) \
    .setAction("clean") \
    .setReplacement(" ") \
    .setPolicy("pretty_all")
```

</div><div class="h3-box" markdown="1">

#### Text Translation

```python
from sparknlp.annotator import MarianTransformer

# German to English translation
translator = MarianTransformer.pretrained("opus_mt_de_en", "xx") \
    .setInputCols(["sentence"]) \
    .setOutputCol("translation")

# French to English translation
translator = MarianTransformer.pretrained("opus_mt_fr_en", "xx") \
    .setInputCols(["sentence"]) \
    .setOutputCol("translation")

# Spanish to English translation
translator = MarianTransformer.pretrained("opus_mt_es_en", "xx") \
    .setInputCols(["sentence"]) \
    .setOutputCol("translation")
```

{:.list1}
- Uses neural machine translation (MarianTransformer)
- Supports many language pairs (200+ models available)
- Production-grade and scalable across Spark clusters
- Higher translation quality than rule-based approaches
- GPU acceleration recommended for large-scale processing

</div><div class="h3-box" markdown="1">

#### Auto Modes

DocumentNormalizer provides preset cleaning modes for common scenarios:

{:.table-model-big}
| Auto Mode | Purpose | Includes |
|-----------|---------|----------|
| `HTML_CLEAN` | Clean HTML content | Remove HTML tags, clean non-ASCII, normalize Unicode |
| `DOCUMENT_CLEAN` | General document cleaning | Clean bullets, dashes, trailing punctuation |
| `SOCIAL_CLEAN` | Social media text | Remove punctuation, normalize social media patterns |
| `LIGHT_CLEAN` | Minimal cleaning | Clean trailing punctuation only |

</div><div class="h3-box" markdown="1">

### Entity Extraction

Spark NLP provides token-aware entity extraction that scales to large datasets.

</div><div class="h3-box" markdown="1">

#### Date Extraction

```python
from sparknlp.annotator import DateMatcher

date_matcher = DateMatcher() \
    .setInputCols(['document', 'token']) \
    .setOutputCol('date') \
    .setOutputFormat("yyyy-MM-dd HH:mm:ss")
```

{:.list1}
- Handles relaxed and relative dates
- Normalized output format
- Semantic date parsing

</div><div class="h3-box" markdown="1">

#### Email and Contact Extraction

```python
from sparknlp.annotator import EntityRulerModel

# Extract email addresses
entity_ruler = EntityRulerModel \
    .pretrained() \
    .setAutoMode("EMAIL_ENTITIES")

# Extract phone numbers
entity_ruler = EntityRulerModel \
    .pretrained() \
    .setAutoMode("CONTACT_ENTITIES")

# Extract IP addresses
entity_ruler = EntityRulerModel \
    .pretrained() \
    .setAutoMode("NETWORK_ENTITIES")

# Extract hostnames and IP address labels
entity_ruler = EntityRulerModel \
    .pretrained() \
    .setAutoMode("NETWORK_ENTITIES") \
    .setRegexEntities(Array(
        "IP_ADDRESS_PATTERN",
        "HOSTNAME_PATTERN"
    ))
```

{:.list1}
- Token-based extraction with offsets
- Can be combined with other communication entities
- Production-ready for network entity extraction
- Supports both IP addresses and associated hostnames/labels

</div><div class="h3-box" markdown="1">

#### Custom Entity Patterns

```python
entity_ruler = EntityRulerModel \
    .pretrained() \
    .setRegexEntities(Array(
        "EMAIL_ADDRESS_PATTERN",
        "US_PHONE_NUMBERS_PATTERN",
        "MAPI_ID_PATTERN"
    ))
```

{:.list1}
- Token-aware extraction with offsets and metadata
- Integrates with other entity extraction pipelines
- Scales efficiently to large corpora
- Provides rich annotation metadata

</div><div class="h3-box" markdown="1">

### Text Chunking

Spark NLP provides flexible chunking strategies for preparing text for downstream processing.

</div><div class="h3-box" markdown="1">

#### Character-Based Chunking

```python
from sparknlp.annotator import DocumentCharacterTextSplitter

splitter = DocumentCharacterTextSplitter() \
    .setInputCols(["document"]) \
    .setOutputCol("chunks") \
    .setChunkSize(1000) \
    .setChunkOverlap(100) \
    .setExplodeSplits(True)
```

</div><div class="h3-box" markdown="1">

#### Token-Based Chunking

```python
from sparknlp.annotator import DocumentTokenSplitter

splitter = DocumentTokenSplitter() \
    .setInputCols(["document", "token"]) \
    .setOutputCol("chunks") \
    .setNumTokens(512) \
    .setTokenOverlap(50) \
    .setExplodeSplits(True)
```

</div><div class="h3-box" markdown="1">

#### Features

{:.list1}
- Configurable split patterns with regex support
- Control over overlap between chunks
- Can preserve or remove separators
- Explode chunks to rows for parallelism
- Deterministic behavior for reproducibility

**Use Cases**: LLM context preparation, semantic search indexing, document summarization

</div><div class="h3-box" markdown="1">

### Comparison: Spark NLP vs Other Frameworks

#### Architecture Philosophy

**Spark NLP**:

{:.list1}
- Built natively on Apache Spark for distributed processing
- Explicit reader separation (Doc / Table / Image)
- Strong typing of outputs
- Designed for large-scale production pipelines

**Other Frameworks**:

{:.list1}
- Typically single-node Python libraries
- Unified API across file types with automatic inference
- Focus on simplicity and ease of use
- Better for small to medium datasets

</div><div class="h3-box" markdown="1">

#### Spark NLP vs Unstructured.io: Practical Trade-offs

{:.table-model-big}
| Aspect | Spark NLP | Unstructured.io |
|--------|-----------|-----------------|
| **Processing Model** | Distributed (Spark) | Single-node |
| **Scalability** | Linear with cluster size | Limited to single machine |
| **Text Coverage** | Complete extraction | Semantic filtering applied |
| **Structural Context** | Full DOM paths and metadata | Limited context preservation |
| **Performance (60 docs)** | ~2× faster | Baseline |
| **API Complexity** | More configuration | Simpler API |
| **Pipeline Integration** | Native Spark integration | Requires external orchestration |
| **Use Case** | Enterprise scale, compliance | Prototyping, small datasets |

Check the full comparison in this blog post: [Evaluating Document AI Frameworks: Spark NLP vs Unstructured for Large-Scale Text Processing](https://medium.com/spark-nlp/evaluating-document-ai-frameworks-spark-nlp-vs-unstructured-for-large-scale-text-processing-0d50874982cd)

</div><div class="h3-box" markdown="1">

#### When to Use Spark NLP

Choose Spark NLP when you need:

{:.list1}
- Processing millions of documents
- Complete text extraction without filtering
- Rich structural and positional metadata
- Integration with existing Spark/Hadoop infrastructure
- Distributed processing and fault tolerance
- Production-grade scalability and reliability
- Traceable, auditable document processing

</div><div class="h3-box" markdown="1">

#### When to Consider Alternatives

Consider lighter frameworks when:

{:.list1}
- Processing small datasets (< 1000 documents)
- Prototyping or exploratory analysis
- No existing Spark infrastructure
- Semantic content extraction preferred over completeness
- Simple API more important than configurability

</div><div class="h3-box" markdown="1">

<!-- ### Resources

**Documentation**
- [Spark NLP Official Documentation](https://sparknlp.org/)
- [Reader2Doc API Reference](https://sparknlp.org/api/python/reference/autosummary/sparknlp/reader/reader2doc/index.html)
- [Reader2Table API Reference](https://sparknlp.org/api/python/reference/autosummary/sparknlp/reader/reader2table/index.html)
- [Reader2Image API Reference](https://sparknlp.org/api/python/reference/autosummary/sparknlp/reader/reader2image/index.html)
- [ReaderAssembler API Reference](https://sparknlp.org/api/python/reference/autosummary/sparknlp/reader/reader_assembler/index.html)
- [DocumentNormalizer API Reference](https://sparknlp.org/api/python/reference/autosummary/sparknlp/annotator/document_normalizer/index.html)

**Example Notebooks**
- [HTML Text Extraction Benchmark](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/open-source-nlp/24.0.Benchmark_Unstructured_Sparknlp_Html.ipynb)
- [Table Structural Context Extraction](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/open-source-nlp/25.0.Table_Structural_Context_Unstructured_Sparknlp.ipynb)
- [Files Ingestion Benchmark](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/open-source-nlp/26.0.Benchmark_Unstructured_and_SparkNLP_Files_Ingestion.ipynb)

**Blog Posts**
- [Evaluating Document AI Frameworks: Spark NLP vs Unstructured for Large-Scale Text Processing](https://medium.com/spark-nlp/evaluating-document-ai-frameworks-spark-nlp-vs-unstructured-for-large-scale-text-processing-0d50874982cd)

**Academic References**
- [Document Layout Analysis](https://aclanthology.org/P13-2116/)
- [TAPAS: Question Answering from Tables](https://www.johnsnowlabs.com/tapas-question-answering-from-tables-in-spark-nlp/)

</div><div class="h3-box" markdown="1">

### Where to look for more information

For questions or issues, visit the [Spark NLP GitHub repository](https://github.com/JohnSnowLabs/spark-nlp) or post on [Stack Overflow](https://stackoverflow.com/questions/tagged/spark-nlp) with the `spark-nlp` tag. -->

</div>
