---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 3.4.0
permalink: /docs/en/spark_nlp_versions/release_notes_3_4_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

### 3.4.0

#### John Snow Labs Spark-NLP 3.4.0: New OpenAI GPT-2, new ALBERT, XLNet, RoBERTa, XLM-RoBERTa, and Longformer for Sequence Classification, support for Spark 3.2, new distributed Word2Vec, extend support to more Databricks & EMR runtimes, new state-of-the-art transformer models, bug fixes, and lots more!

Overview

We are very excited to release Spark NLP 3.4.0! This has been one of the biggest releases we have ever done and we are so proud to share this with our community at the dawn of 2022! 🎉

Spark NLP 3.4.0 extends the support for Apache Spark 3.2.x major releases on Scala 2.12. We now support all 5 major Apache Spark and PySpark releases of 2.3.x, 2.4.x, 3.0.x, 3.1.x, and 3.2.x at once helping our community to migrate from earlier Apache Spark versions to newer releases without being worried about Spark NLP end of life support. We also extend support for new Databricks and EMR instances on Spark 3.2.x clusters.

This release also comes with a brand new GPT2Transformer using OpenAI GPT-2 models for prediction at scale,  new ALBERT, XLNet, RoBERTa, XLM-RoBERTa, and Longformer annotators to use existing or fine-tuned models for Sequence Classification, new distributed and trainable Word2Vec annotators, new state-of-the-art transformer models in many languages, a new param to useBestModel in NerDL during training, bug fixes, and lots more!

As always, we would like to thank our community for their feedback, questions, and feature requests.

Major features and improvements

* **NEW:** Introducing **GPT2Transformer** annotator in Spark NLP 🚀  for Text Generation purposes. `GPT2Transformer` uses OpenAI GPT-2 models from HuggingFace 🤗  for prediction at scale in Spark NLP 🚀 . `GPT-2` is a transformer model trained on a very large corpus of English data in a self-supervised fashion. This means it was trained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was trained to guess the next word in sentences
* **NEW:** Introducing **RoBertaForSequenceClassification** annotator in Spark NLP 🚀. `RoBertaForSequenceClassification` can load RoBERTa Models with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks. This annotator is compatible with all the models trained/fine-tuned by using `RobertaForSequenceClassification` for **PyTorch** or `TFRobertaForSequenceClassification` for **TensorFlow** models in HuggingFace 🤗
* **NEW:** Introducing **XlmRoBertaForSequenceClassification** annotator in Spark NLP 🚀. `XlmRoBertaForSequenceClassification` can load XLM-RoBERTa Models with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks. This annotator is compatible with all the models trained/fine-tuned by using `XLMRobertaForSequenceClassification` for **PyTorch** or `TFXLMRobertaForSequenceClassification` for **TensorFlow** models in HuggingFace 🤗
* **NEW:** Introducing **LongformerForSequenceClassification** annotator in Spark NLP 🚀. `LongformerForSequenceClassification` can load ALBERT Models with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks. This annotator is compatible with all the models trained/fine-tuned by using `LongformerForSequenceClassification` for **PyTorch** or `TFLongformerForSequenceClassification` for **TensorFlow** models in HuggingFace 🤗
* **NEW:** Introducing **AlbertForSequenceClassification** annotator in Spark NLP 🚀. `AlbertForSequenceClassification` can load ALBERT Models with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks. This annotator is compatible with all the models trained/fine-tuned by using `AlbertForSequenceClassification` for **PyTorch** or `TFAlbertForSequenceClassification` for **TensorFlow** models in HuggingFace 🤗
* **NEW:** Introducing **XlnetForSequenceClassification** annotator in Spark NLP 🚀. `XlnetForSequenceClassification` can load XLNet Models with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks. This annotator is compatible with all the models trained/fine-tuned by using `XLNetForSequenceClassification` for **PyTorch** or `TFXLNetForSequenceClassification` for **TensorFlow** models in HuggingFace 🤗
* **NEW:** Introducing trainable and distributed Word2Vec annotators based on Word2Vec in Spark ML. You can train Word2Vec in a cluster on multiple machines to handle large-scale datasets and use the trained model for token-level classifications such as NerDL
* Introducing `useBestModel` param in NerDLApproach annotator. This param in the NerDLApproach preserves and restores the model that has achieved the best performance at the end of the training. The priority is metrics from testDataset (micro F1), metrics from validationSplit (micro F1), and if none is set it will keep track of loss during the training
* Support Apache Spark and PySpark 3.2.x on Scala 2.12. Spark NLP by default is shipped for Spark 3.0.x/3.1.x, but now you have `spark-nlp-spark32` and `spark-nlp-gpu-spark32` packages
* Adding a new param to sparknlp.start() function in Python for Apache Spark 3.2.x (`spark32=True`)
* Update Colab and Kaggle scripts for faster setup. We no longer need to remove Java 11 in order to install Java 8 since Spark NLP works on Java 11. This makes the installation of Spark NLP on Colab and Kaggle as fast as `pip install spark-nlp pyspark==3.1.2`
* Add new scripts/notebook to generate custom TensroFlow graphs for `ContextSpellCheckerApproach` annotator
* Add a new `graphFolder` param to `ContextSpellCheckerApproach` annotator. This param allows to train ContextSpellChecker from a custom made TensorFlow graph
* Support DBFS file system in `graphFolder` param. Starting Spark NLP 3.4.0 you can point NerDLApproach or ContextSpellCheckerApproach to a TF graph hosted on Databricks
* Add a new feature to all classifiers (`ForTokenClassification` and `ForSequenceClassification`) to retrieve classes from the pretrained models
```python
sequenceClassifier = XlmRoBertaForSequenceClassification \
      .pretrained('xlm_roberta_base_sequence_classifier_ag_news', 'en') \
      .setInputCols(['token', 'document']) \
      .setOutputCol('class')

print(sequenceClassifier.getClasses())

#Sports, Business, World, Sci/Tech
```

* Add `inputFormats` param to DateMatcher and MultiDateMatcher annotators. DateMatcher and MultiDateMatcher can now define a list of acceptable input formats via date patterns to search in the text. Consequently, the output format will be defining the output pattern for the unique output format.

```python
date_matcher = DateMatcher() \
    .setInputCols(['document']) \
    .setOutputCol("date") \
    .setInputFormats(["yyyy", "yyyy/dd/MM", "MM/yyyy"]) \
    .setOutputFormat("yyyyMM") \ #previously called `.setDateFormat`
    .setSourceLanguage("en")

```
* Enable batch processing in T5Transformer and MarianTransformer annotators
* Add Schema to `readDataset` in CoNLL() class
* Welcoming 6x new Databricks runtimes to our Spark NLP family:
  * Databricks 10.0
  * Databricks 10.0 ML GPU
  * Databricks 10.1
  * Databricks 10.1 ML GPU
  * Databricks 10.2
  * Databricks 10.2 ML GPU
* Welcoming 3x new EMR 6.x series to our Spark NLP family:
  * EMR 5.33.1 (Apache Spark 2.4.7 / Hadoop 2.10.1)
  * EMR 6.3.1 (Apache Spark 3.1.1 / Hadoop 3.2.1)
  * EMR 6.4.0 (Apache Spark 3.1.2 / Hadoop 3.2.1)

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.4.0)**

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_3_4">Version 3.3.4</a>
    </li>
    <li>
        <strong>Version 3.4.0</strong>
    </li>
</ul>

<ul class="pagination pagination_big">
  <li class="active"><a href="release_notes_3_4_0">3.4.0</a></li>
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
  <li><a href="release_notes_2_4_0">2.4.0</a></li>
</ul>