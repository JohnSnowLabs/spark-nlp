---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 3.3.0
permalink: /docs/en/spark_nlp_versions/release_notes_3_3_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

### 3.3.0

#### John Snow Labs Spark-NLP 3.3.0: New ALBERT, XLNet, RoBERTa, XLM-RoBERTa, and Longformer for Token Classification, 50x times faster to save models, new ways to discover pretrained models and pipelines, new state-of-the-art models, and lots more!

Overview

We are very excited to release Spark NLP 🚀 3.3.0! This release comes with new ALBERT, XLNet, RoBERTa, XLM-RoBERTa, and Longformer existing or fine-tuned models for Token Classification on HuggingFace 🤗 , up to 50x times faster saving Spark NLP models & pipelines, no more 2G limitation for the size of imported TensorFlow models, lots of new functions to filter and display pretrained models & pipelines inside Spark NLP, bug fixes, and more!

We are proud to say Spark NLP 3.3.0 is still compatible across all major releases of Apache Spark used locally, by all Cloud providers such as EMR, and all managed services such as Databricks. The major releases of Apache Spark include Apache Spark 3.0.x/3.1.x (`spark-nlp`), Apache Spark 2.4.x (`spark-nlp-spark24`), and Apache Spark 2.3.x (`spark-nlp-spark23`).

As always, we would like to thank our community for their feedback, questions, and feature requests.

Major features and improvements

* **NEW:** Starting Spark NLP 3.3.0 release there will be `no limitation of size` when you import TensorFlow models! You can now import TF Hub & HuggingFace models larger than 2 Gigabytes of size.
* **NEW:** Up to **50x faster** saving Spark NLP models and pipelines!  We have improved the way we package TensorFlow SavedModel while saving Spark NLP models & pipelines. For instance, it used to take up to 10 minutes to save the `xlm_roberta_base` model before Spark NLP 3.3.0, and now it only takes up to 15 seconds!
* **NEW:** Introducing **AlbertForTokenClassification** annotator in Spark NLP 🚀. `AlbertForTokenClassification` can load ALBERT Models with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. This annotator is compatible with all the models trained/fine-tuned by using `AlbertForTokenClassification` or `TFAlbertForTokenClassification` in HuggingFace 🤗
* **NEW:** Introducing **XlnetForTokenClassification** annotator in Spark NLP 🚀. `XlnetForTokenClassification` can load XLNet Models with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. This annotator is compatible with all the models trained/fine-tuned by using `XLNetForTokenClassificationet` or `TFXLNetForTokenClassificationet` in HuggingFace 🤗
* **NEW:** Introducing **RoBertaForTokenClassification** annotator in Spark NLP 🚀. `RoBertaForTokenClassification` can load RoBERTa Models with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. This annotator is compatible with all the models trained/fine-tuned by using `RobertaForTokenClassification` or `TFRobertaForTokenClassification` in HuggingFace 🤗
* **NEW:** Introducing **XlmRoBertaForTokenClassification** annotator in Spark NLP 🚀. `XlmRoBertaForTokenClassification` can load XLM-RoBERTa Models with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. This annotator is compatible with all the models trained/fine-tuned by using `XLMRobertaForTokenClassification` or `TFXLMRobertaForTokenClassification` in HuggingFace 🤗
* **NEW:** Introducing **LongformerForTokenClassification** annotator in Spark NLP 🚀. `LongformerForTokenClassification` can load Longformer Models with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. This annotator is compatible with all the models trained/fine-tuned by using `LongformerForTokenClassification` or `TFLongformerForTokenClassification` in HuggingFace 🤗
* **NEW:** Introducing new ResourceDownloader functions to easily look for pretrained models & pipelines inside Spark NLP (Python and Scala). You can filter models or pipelines via `language`, `version`, or the name of the `annotator`

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.3.0)**

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_2_3">Version 3.2.3</a>
    </li>
    <li>
        <strong>Version 3.3.0</strong>
    </li>
    <li>
        <a href="release_notes_3_3_1">Version 3.3.1</a>
    </li>
</ul>

<ul class="pagination pagination_big">
  <li><a href="release_notes_3_4_0">3.4.0</a></li>
  <li><a href="release_notes_3_3_4">3.3.4</a></li>
  <li><a href="release_notes_3_3_3">3.3.3</a></li>
  <li><a href="release_notes_3_3_2">3.3.2</a></li>
  <li><a href="release_notes_3_3_1">3.3.1</a></li>
  <li class="active"><a href="release_notes_3_3_0">3.3.0</a></li>
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
  <li><a href="release_notes_2_4_0">2.4.0</a></li>
</ul>