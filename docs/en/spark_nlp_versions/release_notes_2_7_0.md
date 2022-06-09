---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 2.7.0
permalink: /docs/en/spark_nlp_versions/release_notes_2_7_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

### 2.7.0

#### John Snow Labs Spark-NLP 2.7.0: New T5 and MarianMT seq2seq transformers, detect up to 375 languages, word segmentation, over 720+ models and pipelines, support for 192+ languages, and many more!

Overview

We are very excited to release Spark NLP 2.7.0! This has been one of the biggest releases we have ever done that we are so proud to share it with our community!

In this release, we are bringing support to state-of-the-art Seq2Seq and Text2Text transformers. We have developed annotators for Google T5 (Text-To-Text Transfer Transformer) and MarianMNT for Neural Machine Translation with over 646 pretrained models and pipelines.

This release also comes with a refactored and brand new models for language detection and identification. They are more accurate, faster, and support up to 375 languages.

The 2.7.0 release has over 720+ new pretrained models and pipelines while extending our support of multi-lingual models to 192+ languages such as Chinese, Japanese, Korean, Arabic, Persian, Urdu, and Hebrew.

As always, we would like to thank our community for their feedback and support.

Major features and improvements

* **NEW:** Introducing MarianTransformer annotator for machine translation based on MarianNMT models. Marian is an efficient, free Neural Machine Translation framework mainly being developed by the Microsoft Translator team (646+ pretrained models & pipelines in 192+ languages)
* **NEW:** Introducing T5Transformer annotator for Text-To-Text Transfer Transformer (Google T5) models to achieve state-of-the-art results on multiple NLP tasks such as Translation, Summarization, Question Answering, Sentence Similarity, and so on
* **NEW:** Introducing brand new and refactored language detection and identification models. The new LanguageDetectorDL is faster, more accurate, and supports up to 375 languages
* **NEW:** Introducing WordSegmenter annotator, a trainable annotator for word segmentation of languages without any rule-based tokenization such as Chinese, Japanese, or Korean
* **NEW:** Introducing DocumentNormalizer annotator cleaning content from HTML or XML documents, applying either data cleansing using an arbitrary number of custom regular expressions either data extraction following the different parameters
* **NEW:** [Spark NLP Display](https://github.com/JohnSnowLabs/spark-nlp-display) for visualization of different types of annotations
* Add support for new multi-lingual models in UniversalSentenceEncoder annotator
* Add support to Lemmatizer to be trained directly from a DataFrame instead of a text file
* Add training helper to transform CoNLL-U into Spark NLP annotator type columns

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.7.0)**

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_2_6_5">Version 2.6.5</a>
    </li>
    <li>
        <strong>Version 2.7.0</strong>
    </li>
    <li>
        <a href="release_notes_2_7_1">Version 2.7.1</a>
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
  <li class="active"><a href="release_notes_2_7_0">2.7.0</a></li>
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