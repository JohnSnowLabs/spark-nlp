---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 2.5.0
permalink: /docs/en/spark_nlp_versions/release_notes_2_5_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

### 2.5.0

#### John Snow Labs Spark-NLP 2.5.0: ALBERT & XLNet transformers, state-of-the-art spell checker, multi-class sentiment detector, 80+ new models & pipelines in 14 new languages & more

Overview

When we started planning for Spark NLP 2.5.0 release a few months ago the world was a different place!

We have been blown away by the use of Natural Language Processing for early outbreak detections, question-answering chatbot services, text analysis of medical records, monitoring efforts to minimize the virus spread, and many more.

In that spirit, we are honored to announce Spark NLP 2.5.0 release! Witnessing the world coming together to fight coronavirus has driven us to deliver perhaps one of the biggest releases we have ever made.

As always, we thank our community for their feedback, bug reports, and contributions that made this release possible.

Major features and improvements

* **NEW:** A new AlbertEmbeddings annotator with 4 available pre-trained models
* **NEW:** A new XlnetEmbeddings annotator with 2 available pre-trained models
* **NEW:** A new ContextSpellChecker annotator, the state-of-the-art annotator for spell checking
* **NEW:** A new SentimentDL annotator for multi-class sentiment analysis. This annotator comes with 2 available pre-trained models trained on IMDB and Twitter datasets
* **NEW:** Support for 14 new languages with 80+ pretrained models and pipelines!
* Add new PubTator reader to convert automatic annotations of the biomedical datasets into DataFrame
* Introducing a new outputLogsPath param for NerDLApproach, ClassifierDLApproach and SentimentDLApproach annotators
* Refactored CoNLLGenerator to actually use NER labels from the DataFrame
* Unified params in NerDLModel in both Scala and Python
* Extend and complete Scaladoc APIs for all the annotators

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.5.0)**

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_2_4_5">Version 2.4.5</a>
    </li>
    <li>
        <strong>Version 2.5.0</strong>
    </li>
    <li>
        <a href="release_notes_2_5_1">Version 2.5.1</a>
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
  <li class="active"><a href="release_notes_2_5_0">2.5.0</a></li>
  <li><a href="release_notes_2_4_5">2.4.5</a></li>
  <li><a href="release_notes_2_4_4">2.4.4</a></li>
  <li><a href="release_notes_2_4_3">2.4.3</a></li>
  <li><a href="release_notes_2_4_2">2.4.2</a></li>
  <li><a href="release_notes_2_4_1">2.4.1</a></li>
  <li><a href="release_notes_2_4_0">2.4.0</a></li>
</ul>