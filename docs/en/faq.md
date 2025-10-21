---
layout: docs
header: true
seotitle: Spark NLP - FAQ
title: Spark NLP - FAQ
permalink: /docs/en/faq
key: docs-faq
modify_date: "2023-09-14"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

### How to use Spark NLP?

To use Spark NLP in Python, follow these steps:

1. **Installation**:

    ```python
    pip install spark-nlp
    ```

    if you don't have PySpark you should also install the following dependencies:

    ```python
    pip install pyspark numpy
    ```

2. **Initialize SparkSession with Spark NLP**:

   ```python
   import sparknlp

   spark = sparknlp.start()
   ```
   
   #### Optional Parameters
   - **gpu** (`bool`, default `False`) – Enable GPU acceleration.  
   - **apple_silicon** (`bool`, default `False`) – Support for macOS Apple Silicon.  
   - **aarch64** (`bool`, default `False`) – Support for Linux Aarch64.  
   - **memory** (`str`, default `"16G"`) – Spark driver memory.  
   - **cache_folder** (`str`) – Folder for pretrained models (`~/cache_pretrained` by default).  
   - **log_folder** (`str`) – Temp folder for embeddings/logs (local, HDFS, or DBFS only).  
   - **params** (`dict`) – Custom Spark config options.  
   - **cluster_tmp_dir** (`str`) – Where annotator logs are saved.  
   - **real_time_output** (`bool`, default `False`) – Show JVM output live.  
   - **output_level** (`int`, default `1`) – Logging verbosity.  

3. **Use Annotators**:
   Spark NLP offers a variety of annotators (e.g., Tokenizer, SentenceDetector, Lemmatizer). To use them, first create the appropriate pipeline.

   Example using a Tokenizer:

   ```python
   from sparknlp.base import DocumentAssembler
   from sparknlp.annotator import Tokenizer
   from pyspark.ml import Pipeline

   documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

   tokenizer = Tokenizer() \
      .setInputCols(["document"]) \
      .setOutputCol("token")

   pipeline = Pipeline(stages=[documentAssembler, tokenizer])
   ```

4. **Transform Data**:
   Once you have a pipeline, you can transform your data.

   ```python
   result = pipeline.fit(data).transform(data)
   ```

5. **Explore and Utilize Models**:
   Spark NLP offers a wide range of [pretrained models](https://sparknlp.org/models) for tasks like Named Entity Recognition (NER), sentiment analysis, and more. You can easily plug these into your pipeline and customize as needed.

6. **Further Reading**:
   Dive deeper into the [official documentation](https://sparknlp.org/docs/en/install) for more detailed examples, a complete list of annotators and models, and best practices for building NLP pipelines.

</div><div class="h3-box" markdown="1">

### Is Spark NLP free?

Short answer: 100%! Free forever inculding any commercial use.

Longer answer: Yes, Spark NLP is an open-source library and can be used freely. It's released under the Apache License 2.0. Users can use, modify, and distribute it without incurring costs.

</div><div class="h3-box" markdown="1">

### What is the difference between spaCy and Spark NLP?

Both spaCy and Spark NLP are popular libraries for Natural Language Processing, but Spark NLP shines when it comes to scalability and distributed processing. Here are some key differences between the two:

1. **Scalability & Distributed Processing**:
   - **Spark NLP**: Built on top of Apache Spark, it's designed for distributed processing and handling large datasets at scale. This makes it especially suitable for big data processing tasks that need to run on a cluster.
   - **spaCy**: Designed for processing data on a single machine and it's not natively built for distributed computing.

2. **Language Models & Pretrained Pipelines**:
   - **Spark NLP**: Offers over 18,000 diverse pre-trained models and pipelines for over 235 languages, making it easy to get started on various NLP tasks. It also makes it easy to import your custom models from Hugging Face in TensorFlow and ONNX formats. Spark NLP also offeres a large number of state-of-the-art Large Language Models (LLMs) like BERT, RoBERTa, ALBERT, T5, OpenAI Whisper, and many more for Text Embeddings (useful for RAG), Named Entity Recognition, Text Classification, Answering, Automatic Speech Recognition, and more. These models can be used out of the box or fine-tuned on your own data.
   - **spaCy**: Provides support for multiple languages with its models and supports tasks like tokenization, named entity recognition, and dependency parsing out of the box. However, spaCy doesn't have any Models Hub and the number of offered models out of the box is limited.

3. **Licensing & Versions**:
   - **Spark NLP**: The core library is open-source under the Apache License 2.0, making it free for both academic and commercial use.
   - **spaCy**: Open-source and released under the MIT license.

</div><div class="h3-box" markdown="1">

### What are the Spark NLP models?

Spark NLP provides a range of models to tackle various NLP tasks. These models are often pre-trained on large datasets and can be fine-tuned or used directly for inference. Some of the primary categories and examples of Spark NLP models include:

1. **Named Entity Recognition (NER)**  
   - Pre-trained models for recognizing entities such as persons, organizations, and locations.  
   - Specialized models for domains like healthcare to detect medical entities.  

2. **Text Classification**  
   - Models for sentiment analysis, topic classification, intent detection, and more.  

3. **Word Embeddings**  
   - Word2Vec, GloVe, and transformer embeddings (BERT, RoBERTa, etc.).  
   - Generate embeddings for words or sentences to power downstream tasks.  

4. **Language Models**  
   - Pre-trained transformer-based models like BERT, ALBERT, ELECTRA.  
   - Can be fine-tuned for domain-specific tasks.  

5. **Dependency Parsing**  
   - Models that analyze grammatical structure and word relationships.  

6. **Spell Checking and Correction**  
   - Detect and correct spelling mistakes in text.  

7. **Sentence Embeddings**  
   - Models for sentence-level vector representations, such as Universal Sentence Encoder.  

8. **Translation and Language Detection**  
   - Detect the language of a given text or translate between languages.  

9. **Text Matching**  
   - Models for textual similarity, duplicate detection, or paraphrase detection.  

10. **Question Answering (QA)**  
   - Extractive QA models built on transformers (e.g., BERT, DistilBERT).  

11. **Summarization**  
   - Models for abstractive or extractive text summarization.  

12. **Entity Resolution / Normalization**  
   - Map recognized entities to canonical forms (e.g., ICD-10 codes in healthcare).  

13. **Zero-Shot & Few-Shot Classification**  
   - Transformer-based models that classify text into categories without task-specific training.  

14. **Large Language Models (LLMs)**  
   - General-purpose models (e.g., Phi, Qwen, LLaMA, Mistral, Falcon, GPT-style, and MPT).  
   - Used for text generation, summarization, rewriting, reasoning, and more.  

15. **Vision-Language Models (VLMs)**  
   - Multimodal models combining text and images (e.g., BLIP, CLIP, LLaVA, Kosmos-2).  
   - Tasks include image captioning, document understanding, OCR post-processing, and visual QA.  

16. **Pretrained Pipelines**  
   - Ready-to-use pipelines combining multiple models and annotators.  
   - Allow quick text processing without building a custom pipeline.  

For the latest list of models, detailed documentation, and instructions on how to use them, visiting the [Official Spark NLP Models Hub](https://sparknlp.org/models) would be beneficial.

</div><div class="h3-box" markdown="1">

### What are the main functions of Spark NLP?

Spark NLP offers a comprehensive suite of functionalities tailored for natural language processing tasks via large language models. Some of the main functions and capabilities include:

1. **Text Tokenization**:
   - Segmenting text into words, phrases, or other meaningful elements called tokens.

2. **Named Entity Recognition (NER)**:
   - Identifying and classifying named entities in text, such as names of people, places, organizations, dates, etc.

3. **Document Classification**:
   - Categorizing documents or chunks of text into predefined classes or topics.

4. **Sentiment Analysis**:
   - Determining the sentiment or emotion expressed in a piece of text, typically as positive, negative, or neutral.

5. **Dependency Parsing**:
   - Analyzing the grammatical structure of a sentence to establish relationships between words.

6. **Lemmatization and Stemming**:
   - Reducing words to their base or root form. For example, “running” becomes “run”.

7. **Spell Checking and Correction**:
   - Identifying and correcting spelling mistakes in text.

8. **Word and Sentence Embeddings**:
   - Transforming words or sentences into numerical vectors, useful for many machine learning tasks.

9. **Language Detection and Translation**:
   - Detecting the language of a given text and translating text between different languages.

10. **Text Matching and Similarity**:

    - Calculating the similarity between pieces of text or determining if texts are duplicates or paraphrases of one another.

11. **Chunking**:

    - Extracting short, meaningful phrases from the text.

12. **Stop Words Removal**:

    - Removing commonly used words that don't carry significant meaning in most contexts (e.g., "and", "the", "is").

13. **Normalization**:

    - Converting text into a standard or regular form, such as converting all letters to lowercase or removing special characters.

14. **Pre-trained Pipelines**:

    - Ready-to-use workflows combining multiple functions, allowing users to process text without creating a custom sequence of operations.

15. **Customizable Workflows**:

    - Building custom pipelines by chaining different annotators and models to create a tailored NLP processing sequence.

Spark NLP is designed to be highly scalable and can handle large-scale text processing tasks efficiently by leveraging the distributed processing capabilities of Apache Spark.

To fully grasp the breadth of functions and learn how to use them, users are encouraged to explore the [official Spark NLP documentation](https://nlp.johnsnowlabs.com/docs/en/quickstart).

</div><div class="h3-box" markdown="1">

### Where can I get prebuilt versions of Spark NLP?

Prebuilt versions of Spark NLP can be obtained through multiple channels, depending on your development environment and platform:

1. **PyPI (for Python Users)**:
   You can install Spark NLP using `pip`, the Python package installer.

   ```bash
   pip install spark-nlp
   ```

2. **Maven Central (for Java/Scala Users)**:
   If you are using Maven, you can add the following dependency to your `pom.xml`:

   ```xml
   <dependency>
      <groupId>com.johnsnowlabs.nlp</groupId>
      <artifactId>spark-nlp_2.12</artifactId>
      <version>LATEST_VERSION</version>
   </dependency>
   ```

   Make sure to replace `LATEST_VERSION` with the desired version of Spark NLP.

3. **Spark Packages**:
   For those using the `spark-shell`, `pyspark`, or `spark-submit`, you can include Spark NLP directly via Spark Packages:

   ```bash
   --packages com.johnsnowlabs.nlp:spark-nlp_2.12:LATEST_VERSION
   ```

4. **Pre-trained Models & Pipelines**:
   Apart from the library itself, Spark NLP provides a range of pre-trained models and pipelines. These can be found on the [Spark NLP Model Hub](https://sparknlp.org/models).

Always make sure to consult the [official documentation](https://sparknlp.org/docs/en/quickstart) or the [GitHub repository](https://github.com/JohnSnowLabs/spark-nlp/) for the latest instructions and versions available.

</div>