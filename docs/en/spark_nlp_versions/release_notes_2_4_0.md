---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 2.4.0
permalink: /docs/en/spark_nlp_versions/release_notes_2_4_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

### 2.4.0

#### John Snow Labs Spark-NLP 2.4.0: New TensorFlow 1.15, Universal Sentence Encoder, Elmo, faster Word Embeddings & more

We are very excited to finally release Spark NLP v2.4.0! This has been one of the largest releases we have ever made since the inception of the library! The new release of Spark NLP `2.4.0` has been migrated to TensorFlow `1.15.0` which takes advantage of the latest deep learning technologies and pre-trained models.

Major features and improvements

* **NEW:** TensorFlow 1.15.0 now works behind Spark NLP. This brings implicit improvements in performance, accuracy, and functionalities
* **NEW:** UniversalSentenceEncoder annotator with 2 pre-trained models from TF Hub
* **NEW:** ElmoEmbeddings with a pre-trained model from TF Hub
* **NEW:** All our pre-trained models are now cross-platform!
* **NEW:** For the first time, all the multi-lingual models and pipelines are available for Windows users (French, German and Italian)
* **NEW:** MultiDateMatcher capable of matching more than one date per sentence (Extends DateMatcher algorithm)
* **NEW:** BigTextMatcher works best with large amounts of input data
* BertEmbeddings improvements with 5 new models from TF Hub
* RecursivePipelineModel as an enhanced PipelineModel allows Annotators to access previous annotators in the pipeline for more ML strategies
* LazyAnnotators: A new Param in Annotators allows them to stand idle in the Pipeline and do nothing. Can be called by other Annotators in a RecursivePipeline
* RocksDB is now available as a flexible API called `Storage`. Allows any annotator to have it's own distributed local index database
* Now our Tensorflow pre-trained models are cross-platform. Enabling multi-language models and other improvements to Windows users.
* Improved IO performance in general for handling embeddings
* Improved cache cleanup and GC by liberating open files utilized in RocksDB (to be improved further)
* Tokenizer and SentenceDetector Params minLength and MaxLength to filter out annotations outside these bounds
* Tokenizer improvements in splitChars and simplified rules
* DateMatcher improvements
* TextMatcher improvements preload algorithm information within the model for faster prediction
* Annotators the utilize embeddings have now a strict validation to be using exactly the embeddings they were trained with
* Improvements in the API allow Annotators with Storage to save and load their RocksDB database independently and let it be shared across Annotators and let it be shared across Annotators

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <strong>Version 2.4.0</strong>
    </li>
    <li>
        <a href="release_notes_2_4_1">Version 2.4.1</a>
    </li>
</ul>

<ul class="pagination pagination_big">
  <li><a href="release_notes_3_4_0">3.4.0</a></li>
  <li><a href="release_notes_3_3_4">3.3.4</a></li>
  <li><a href="release_notes_3_3_3">3.3.3</a></li>
  <li><a href="release_notes_3_3_2">3.3.2</a></li>
  <li><a href="release_notes_3_3_1">3.3.1</a></li>
  <li><a href="release_notes_3_3_0">3.3.0</a></li>
  <li><a href="release_notes_3_2_3">3.2.3</a></li>
  <li><a href="release_notes_3_2_2">3.2.2</a></li>
  <li><a href="release_notes_3_2_1">3.2.1</a></li>
  <li><a href="release_notes_3_2_0">3.2.0</a></li>
  <li><a href="release_notes_3_1_3">3.1.3</a></li>
  <li><a href="release_notes_3_1_2">3.1.2</a></li>
  <li><a href="release_notes_3_1_1">3.1.1</a></li>
  <li><a href="release_notes_3_1_0">3.1.0</a></li>
  <li><a href="release_notes_3_0_3">3.0.3</a></li>
  <li><a href="release_notes_3_0_2">3.0.2</a></li>
  <li><a href="release_notes_3_0_1">3.0.1</a></li>
  <li><a href="release_notes_3_0_0">3.0.0</a></li>
  <li><a href="release_notes_2_7_5">2.7.5</a></li>
  <li><a href="release_notes_2_7_4">2.7.4</a></li>
  <li><a href="release_notes_2_7_3">2.7.3</a></li>
  <li><a href="release_notes_2_7_2">2.7.2</a></li>
  <li><a href="release_notes_2_7_1">2.7.1</a></li>
  <li><a href="release_notes_2_7_0">2.7.0</a></li>
  <li><a href="release_notes_2_6_5">2.6.5</a></li>
  <li><a href="release_notes_2_6_4">2.6.4</a></li>
  <li><a href="release_notes_2_6_3">2.6.3</a></li>
  <li><a href="release_notes_2_6_2">2.6.2</a></li>
  <li><a href="release_notes_2_6_1">2.6.1</a></li>
  <li><a href="release_notes_2_6_0">2.6.0</a></li>
  <li><a href="release_notes_2_5_5">2.5.5</a></li>
  <li><a href="release_notes_2_5_4">2.5.4</a></li>
  <li><a href="release_notes_2_5_3">2.5.3</a></li>
  <li><a href="release_notes_2_5_2">2.5.2</a></li>
  <li><a href="release_notes_2_5_1">2.5.1</a></li>
  <li><a href="release_notes_2_5_0">2.5.0</a></li>
  <li><a href="release_notes_2_4_5">2.4.5</a></li>
  <li><a href="release_notes_2_4_4">2.4.4</a></li>
  <li><a href="release_notes_2_4_3">2.4.3</a></li>
  <li><a href="release_notes_2_4_2">2.4.2</a></li>
  <li><a href="release_notes_2_4_1">2.4.1</a></li>
  <li class="active"><a href="release_notes_2_4_0">2.4.0</a></li>
</ul>