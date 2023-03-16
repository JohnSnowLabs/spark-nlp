# Spark NLP Examples

This is the directory for examples on how to use Spark NLP in various environments.

These include examples for Python, Scala, Java and Docker.

For an introduction into using Spark NLP, take a look at the [Quick
Start](python/quick_start.ipynb). If you are planning to use Spark NLP on Google Colab,
see [Quick Start on Google Colab](python/quick_start_google_colab.ipynb). The notebook
[Spark NLP Basics](python/annotation/text/english/spark-nlp-basics) covers the basics of
Spark NLP.

For more use-cases and advanced examples, take a look at the following table of contents.

## Table Of Contents

- [Python Examples](python)
  - [Using Annotators](python/annotation)
    - [Audio Processing](python/annotation/audio)
    - [Image Processing](python/annotation/image)
    - [Text Processing](python/annotation/text)
      - [Chinese](python/annotation/text/chinese)
      - [English](python/annotation/text/english)
        - [Assembling Documents](python/annotation/text/english/document-assembler)
        - [Assembling Tokens to Documents](python/annotation/text/english/token-assembler)
        - [Chunking](python/annotation/text/english/chunking)
        - [Co-reference Resolution](python/annotation/text/english/coreference-resolution)
        - [Document Normalization](python/annotation/text/english/document-normalizer)
        - [Embeddings](python/annotation/text/english/embeddings)
        - [Graph Extraction](python/annotation/text/english/graph-extraction)
        - [Keyword Extraction](python/annotation/text/english/keyword-extraction)
        - [Language Detection](python/annotation/text/english/language-detection)
        - [Matching text using Regex](python/annotation/text/english/regex-matcher)
        - [Model Downloader](python/annotation/text/english/model-downloader)
        - [Named Entity Recognition](python/annotation/text/english/named-entity-recognition)
        - [Pretrained Pipelines](python/annotation/text/english/pretrained-pipelines)
        - [Question Answering](python/annotation/text/english/question-answering)
        - [Sentence Detection](python/annotation/text/english/sentence-detection)
        - [Sentiment Detection](python/annotation/text/english/sentiment-detection)
        - [Stemming](python/annotation/text/english/stemmer)
        - [Stop Words Cleaning](python/annotation/text/english/stop-words)
        - [Text Matching](python/annotation/text/english/text-matcher-pipeline)
        - [Text Similarity](python/annotation/text/english/text-similarity)
        - [Tokenization Using Regex](python/annotation/text/english/regex-tokenizer)
      - [French](python/annotation/text/french)
      - [German](python/annotation/text/german)
      - [Italian](python/annotation/text/italian)
      - [Multilingual](python/annotation/text/multilingual)
      - [Portuguese](python/annotation/text/portuguese)
      - [Spanish](python/annotation/text/spanish)
  - [Training Annotators](python/training)
    - [Chinese](python/training/chinese)
    - [English](python/training/english)
      - [Document Embeddings with Doc2Vec](python/training/english/doc2vec)
      - [Matching Entities with EntityRuler](python/training/english/entity-ruler)
      - [Named Entity Recognition with CRF](python/training/english/crf-ner)
      - [Named Entity Recognition with Deep Learning](python/training/english/dl-ner)
        - [Creating NerDL Graphs](python/training/english/dl-ner/nerdl-graph)
      - [Sentiment Analysis](python/training/english/sentiment-detection)
      - [Text Classification](python/training/english/classification)
      - [Word embeddings with Word2Vec](python/training/english/word2vec)
    - [French](python/training/french)
    - [Italian](python/training/italian)
  - [Transformers in Spark NLP](python/transformers)
  - [Logging](python/logging)
- [Scala Examples](scala)
  - [Training Annotators](scala/training)
  - [Using Annotators](scala/annotation)
- [Java Examples](java)
- [SparkNLP Setup with Docker](docker)
- [Utilities](util)
