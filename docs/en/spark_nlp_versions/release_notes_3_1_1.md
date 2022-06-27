---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 3.1.1
permalink: /docs/en/spark_nlp_versions/release_notes_3_1_1
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

### 3.1.1

#### John Snow Labs Spark-NLP 3.1.1: New and improved ALBERT with support for external Transformers, real-time metrics in Python notebooks, bug fixes, and many more improvements!

Overview

We are pleased to release Spark NLP 🚀  3.1.1! We have a new and much-improved ALBERT annotator with support for HuggingFace 🤗  models in Spark NLP. We managed to make AlbertEmbeddings almost 7x times faster on GPU compare to prior releases!

As always, we would like to thank our community for their feedback, questions, and feature requests.

New Features

* Migrate AlbertEmbeddings to TensorFlow v2. This allows the importing of HuggingFace ALBERT models to Spark NLP
* Migrate AlbertEmbeddings to BatchAnnotate to allow better performance on accelerated hardware such as GPU
* Enable stdout/stderr in real-time for child processes via `sparknlp.start()`. Thanks to PySpark 3.x, this is now possible with `sparknlp.start(real_time_output=True)` to have the outputs of Spark NLP (such as metrics during training) right in your Jupyter, Colab, and Kaggle notebooks.
* Complete examples for all annotators in Scaladoc APIs https://github.com/JohnSnowLabs/spark-nlp/pull/5668

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.1.2)**

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_1_0">Version 3.1.0</a>
    </li>
    <li>
        <strong>Version 3.1.1</strong>
    </li>
    <li>
        <a href="release_notes_3_1_2">Version 3.1.2</a>
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
  <li class="active"><a href="release_notes_3_1_1">3.1.1</a></li>
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
  <li><a href="release_notes_2_4_0">2.4.0</a></li>
</ul>