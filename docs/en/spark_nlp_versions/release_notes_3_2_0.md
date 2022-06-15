---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 3.2.0
permalink: /docs/en/spark_nlp_versions/release_notes_3_2_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

### 3.2.0

#### John Snow Labs Spark-NLP 3.2.0: New Longformer embeddings, BERT and DistilBERT for Token Classification, GraphExctraction, Spark NLP Configurations, new state-of-the-art multilingual NER models, and lots more!

Overview

We are very excited to release Spark NLP 🚀 3.2.0! This is a big release with new Longformer models for long documents, BertForTokenClassification & DistilBertForTokenClassification for existing or fine-tuned models on HuggingFace, GraphExctraction & GraphFinisher to find relevant relationships between words, support for multilingual Date Matching, new Pydoc for Python APIs, and so many more!

As always, we would like to thank our community for their feedback, questions, and feature requests.

Major features and improvements

* **NEW:** Introducing **LongformerEmbeddings** annotator. `Longformer` is a transformer model for long documents. Longformer is a BERT-like model started from the RoBERTa checkpoint and pretrained for MLM on long documents. It supports sequences of length up to 4,096.

We have trained two NER models based on Longformer Base and Large embeddings:

| Model | Accuracy | F1 Test | F1 Dev |
|:------|:----------|:------|:--------|
|ner_conll_longformer_base_4096  | 94.75% | 90.09 | 94.22
|ner_conll_longformer_large_4096 | 95.79% | 91.25 | 94.82

* **NEW:** Introducing **BertForTokenClassification** annotator. `BertForTokenClassification` can load BERT Models with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. This annotator is compatible with all the models trained/fine-tuned by using `BertForTokenClassification` or `TFBertForTokenClassification` in HuggingFace 🤗
* **NEW:** Introducing **DistilBertForTokenClassification** annotator. `DistilBertForTokenClassification` can load BERT Models with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. This annotator is compatible with all the models trained/fine-tuned by using `DistilBertForTokenClassification` or `TFDistilBertForTokenClassification` in HuggingFace 🤗
* **NEW:** Introducing **GraphExctraction** and **GraphFinisher** annotators to extract a dependency graph between entities. The **GraphExtraction** class takes e.g. extracted entities from a `NerDLModel` and creates a dependency tree that describes how the entities relate to each other. For that, a triple store format is used. Nodes represent the entities and the edges represent the relations between those entities. The graph can then be used to find relevant relationships between words
* **NEW:** Introducing support for multilingual **DateMatcher** and **MultiDateMatcher** annotators. These two annotators will support **English**, **French**, **Italian**, **Spanish**, **German**, and **Portuguese** languages
* **NEW:** Introducing new **Python APIs** and fully documented **Pydoc**
* **NEW:** Introducing new **Spark NLP configurations** via spark.conf() by deprecating `application.conf` usage. You can easily change Spark NLP configurations in SparkSession. For more examples please visti [Spark NLP Configuration](https://github.com/JohnSnowLabs/spark-nlp#spark-nlp-configuration)
* Add support for Amazon S3 to `log_folder` Spark NLP config and `outputLogsPath` param in `NerDLApproach`, `ClassifierDlApproach`, `MultiClassifierDlApproach`, and `SentimentDlApproach` annotators
* Added examples to all Spark NLP Scaladoc
* Added examples to all Spark NLP Pydoc
* Welcoming new Databricks runtimes to our Spark NLP family:
  * Databricks 8.4 ML & GPU
* Fix printing a wrong version return in sparknlp.version()

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.2.0)**

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_1_3">Version 3.1.3</a>
    </li>
    <li>
        <strong>Version 3.2.0</strong>
    </li>
    <li>
        <a href="release_notes_3_2_1">Version 3.2.1</a>
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
  <li class="active"><a href="release_notes_3_2_0">3.2.0</a></li>
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