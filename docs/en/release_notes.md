---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes
permalink: /docs/en/release_notes
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

### 3.3.4

#### John Snow Labs Spark-NLP 3.3.4: Patch release

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.3.4)**

### 3.3.3

#### John Snow Labs Spark-NLP 3.3.3: New DistilBERT for Sequence Classification, new trainable and distributed Doc2Vec, BERT improvements on GPU, new state-of-the-art DistilBERT models for topic and sentiment detection, enhancements, and bug fixes!

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.3.3)**

### 3.3.2

#### John Snow Labs Spark-NLP 3.3.2: New BERT for Sequence Classification, Comet.ml logging integration, new state-of-the-art BERT topic and sentiment detection models, and bug fixes!

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.3.2)**

### 3.3.1

#### John Snow Labs Spark-NLP 3.3.1: New EntityRuler annotator, better integration with TokenClassification annotators, new state-of-the-art XLM-RoBERTa models in African Languages, and bug fixes!

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.3.1)**

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

### 3.2.3

#### John Snow Labs Spark-NLP 3.2.3: New Transformers and Training documentation, Improved GraphExtraction, new Japanese models, new multilingual Transformer models, enhancements, and bug fixes

Overview

We are pleased to release Spark NLP 🚀 3.2.3! This release comes with new and completed documentation for all Transformers and Trainable annotators in Spark NLP, new Japanese NER and Embeddings models, new multilingual Transformer models, code enhancements, and bug fixes.

As always, we would like to thank our community for their feedback, questions, and feature requests.

New Features

* Add delimiter feature to CoNLL() class to support other delimiters in CoNLL files https://github.com/JohnSnowLabs/spark-nlp/pull/5934
* Add support for IOB in addition to IOB2 format in GraphExtraction annotator https://github.com/JohnSnowLabs/spark-nlp/pull/6101
* Change YakeModel output type from KEYWORD to CHUNK to have more available features after the YakeModel annotator such as Chunk2Doc or ChunkEmbeddings https://github.com/JohnSnowLabs/spark-nlp/pull/6065
* Welcoming [Databricks Runtime 9.0](https://docs.databricks.com/release-notes/runtime/9.0.html), 9.0 ML, and 9.0 ML with GPU
* A new and completed [Transformer page](https://nlp.johnsnowlabs.com/docs/en/transformers)
    * description
    * default model's name
    * link to Models Hub
    * link to notebook on Spark NLP Workshop
    * link to Python APIs
    * link to Scala APIs
    * link to source code and unit test
    * Examples in Python and Scala for
        * Prediction
        * Training
        * Raw Embeddings
* A new and completed [Training page](https://nlp.johnsnowlabs.com/docs/en/training)
    * Training Datasets
    * Text Processing
    * Spell Checkers
    * Token Classification
    * Text Classification
    * External Trainable Models

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.2.3)**

### 3.2.2

#### John Snow Labs Spark-NLP 3.2.2: Models Hub for the community by the community, new RoBERTa and XLM-RoBERTa Sentence Embeddings, 40 new models in 20 languages, bug fixes, and more!

Overview

We are pleased to release Spark NLP 🚀 3.2.2! This release comes with accessible Models Hub to our community to host their models and pipelines for free, new RoBERTa and XLM-RoBERTa Sentence Embeddings, over 40 new models and pipelines in 20+ languages, bug fixes, and more

As always, we would like to thank our community for their feedback, questions, and feature requests.

New Features

* A new RoBertaSentenceEmbeddings annotator for sentence embeddings used in SentimentDL, ClassifierDL, and MultiClassifierDL annotators
* A new XlmRoBertaSentenceEmbeddings annotator for sentence embeddings used in SentimentDL, ClassifierDL, and MultiClassifierDL annotators
* Add support for AWS MFA via Spark NLP configuration
* Add new AWS configs to Spark NLP configuration when using a private S3 bucket to store logs for training models or access TF graphs needed in NerDLApproach
  * spark.jsl.settings.aws.credentials.access_key_id
  * spark.jsl.settings.aws.credentials.secret_access_key
  * spark.jsl.settings.aws.credentials.session_token
  * spark.jsl.settings.aws.s3_bucket
  * spark.jsl.settings.aws.region

Models Hub for the community, by the community

Serve Your Spark NLP Models for Free! You can host and share your Spark NLP models & pipelines publicly with everyone to reuse them with one line of code!

We are opening Models Hub to everyone to upload their models and pipelines, showcase their work, and share them with others.

Please visit the following page for more information: [https://modelshub.johnsnowlabs.com/](https://modelshub.johnsnowlabs.com/)

![image](https://user-images.githubusercontent.com/5762953/131699383-96fe7637-3a1b-460e-bf4a-43b44c815951.png)

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.2.2)**
### 3.2.1

#### John Snow Labs Spark-NLP 3.2.1: Patch release

Patch release

* Fix `unsupported model` error in pretrained function for **LongformerEmbeddings**, **BertForTokenClassification**, and **DistilBertForTokenClassification** https://github.com/JohnSnowLabs/spark-nlp/issues/5947

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.2.3)**
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
### 3.1.3

#### John Snow Labs Spark-NLP 3.1.3: TF Hub support, new multilingual NER models for 40 languages, state-of-the-art multilingual sentence embeddings for 100+ languages, and bug fixes!

Overview

We are pleased to release Spark NLP 🚀  3.1.3! In this release, we bring notebooks to easily import models for BERT and ALBERT models from TF Hub into Spark NLP, new multilingual NER models for 40 languages with a fine-tuned XLM-RoBERTa model, and new state-of-the-art document/sentence embeddings models for English and 100+ languages!

As always, we would like to thank our community for their feedback, questions, and feature requests.

New Features

* Support BERT models from TF Hub to Spark NLP
* Support BERT for sentence embeddings from TF Hub to Spark NLP
* Support ALBERT models from TF Hub to Spark NLP
* Welcoming new Databricks 8.4 / 8.4 ML/GPU runtimes to Spark NLP platforms

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.1.3)**
### 3.1.2

#### John Snow Labs Spark-NLP 3.1.2: New and improved XLNet with support for external Transformers, better documentation, bug fixes, and other improvements!

Overview

We are pleased to release Spark NLP 🚀  3.1.2! We have a new and much-improved XLNet annotator with support for HuggingFace 🤗  models in Spark NLP. We managed to make XlnetEmbeddings almost 5x times faster on GPU compare to prior releases!

As always, we would like to thank our community for their feedback, questions, and feature requests.

New Features

* Migrate XlnetEmbeddings to TensorFlow v2. This allows the importing of HuggingFace XLNet models to Spark NLP
* Migrate XlnetEmbeddings to BatchAnnotate to allow better performance on accelerated hardware such as GPU
* Dynamically extract special tokens from SentencePiece model in XlmRoBertaEmbeddings
* Add setIncludeAllConfidenceScores param in NerDLModel to merge confidence scores per label to only predicted label
* Fully updated [Annotators page](https://nlp.johnsnowlabs.com/docs/en/annotators) with full examples in Python and Scala
* Fully update [Transformers page](https://nlp.johnsnowlabs.com/docs/en/transformers) for all the transformers in Spark NLP

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.1.3)**
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

### 3.1.0

#### John Snow Labs Spark-NLP 3.1.0: Over 2600+ new models and pipelines in 200+ languages, new DistilBERT, RoBERTa, and XLM-RoBERTa transformers, support for external Transformers, and lots more!

Overview

We are very excited to release Spark NLP 🚀  3.1.0! This is one of our biggest releases with lots of models, pipelines, and groundworks for future features that we are so proud to share it with our community.

Spark NLP 3.1.0 comes with over 2600+ new pretrained models and pipelines in over 200+ languages, new DistilBERT, RoBERTa, and XLM-RoBERTa annotators, support for HuggingFace 🤗 (Autoencoding) models in Spark NLP, and extends support for new Databricks and EMR instances.

As always, we would like to thank our community for their feedback, questions, and feature requests.

Major features and improvements

* **NEW:** Introducing DistiBertEmbeddings annotator. DistilBERT is a small, fast, cheap, and light Transformer model trained by distilling BERT base. It has 40% fewer parameters than `bert-base-uncased`, runs 60% faster while preserving over 95% of BERT’s performances
* **NEW:** Introducing RoBERTaEmbeddings annotator. RoBERTa (Robustly Optimized BERT-Pretraining Approach) models deliver state-of-the-art performance on NLP/NLU tasks and a sizable performance improvement on the GLUE benchmark. With a score of 88.5, RoBERTa reached the top position on the GLUE leaderboard
* **NEW:** Introducing XlmRoBERTaEmbeddings annotator. XLM-RoBERTa (Unsupervised Cross-lingual Representation Learning at Scale) is a large multi-lingual language model, trained on 2.5TB of filtered CommonCrawl data with 100 different languages. It also outperforms multilingual BERT (mBERT) on a variety of cross-lingual benchmarks, including +13.8% average accuracy on XNLI, +12.3% average F1 score on MLQA, and +2.1% average F1 score on NER. XLM-R performs particularly well on low-resource languages, improving 11.8% in XNLI accuracy for Swahili and 9.2% for Urdu over the previous XLM model
* **NEW:** Introducing support for HuggingFace exported models in equivalent Spark NLP annotators. Starting this release, you can easily use the `saved_model` feature in HuggingFace within a few lines of codes and import any BERT, DistilBERT, RoBERTa, and XLM-RoBERTa models to Spark NLP. We will work on the remaining annotators and extend this support to the rest with each release - For more information please visit [this discussion](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669)
* **NEW:** Migrate MarianTransformer to BatchAnnotate to control the throughput when you are on accelerated hardware such as GPU to fully utilize it
* Upgrade to TensorFlow v2.4.1 with native support for Java to take advantage of many optimizations for CPU/GPU and new features/models introduced in TF v2.x
* Update to CUDA11 and cuDNN 8.0.2 for GPU support
* Implement ModelSignatureManager to automatically detect inputs, outputs, save and restore tensors from SavedModel in TF v2. This allows Spark NLP 3.1.x to extend support for external Encoders such as HuggingFace and TF Hub (coming soon!)
* Implement a new BPE tokenizer for RoBERTa and XLM models. This tokenizer will use the custom tokens from `Tokenizer` or `RegexTokenizer` and generates token pieces, encodes, and decodes the results
* Welcoming new Databricks runtimes to our Spark NLP family:
  * Databricks 8.1 ML & GPU
  * Databricks 8.2 ML & GPU
  * Databricks 8.3 ML & GPU
* Welcoming a new EMR 6.x series to our Spark NLP family:
  * EMR 6.3.0 (Apache Spark 3.1.1 / Hadoop 3.2.1)
 * Added examples to Spark NLP Scaladoc

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.1.0)**
### 3.0.3

#### John Snow Labs Spark-NLP 3.0.3: New T5 features for longer and more accurate text generation, new multi-lingual models & pipelines, bug fixes, and other improvements!

Overview

We are glad to release Spark NLP 3.0.3! We have added some new features to our T5 Transformer annotator to help with longer and more accurate text generation, trained some new multi-lingual models and pipelines in `Farsi`, `Hebrew`, `Korean`, and `Turkish`, and fixed some bugs in this release.

As always, we would like to thank our community for their feedback, questions, and feature requests.

New Features

* Add 6 new features to T5Transformer for longer and better text generation
  - doSample: Whether or not to use sampling; use greedy decoding otherwise
  - temperature: The value used to module the next token probabilities
  - topK: The number of highest probability vocabulary tokens to keep for top-k-filtering
  - topP: If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation
  - repetitionPenalty: The parameter for repetition penalty. 1.0 means no penalty. See [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858) paper for more details
  - noRepeatNgramSize: If set to int > 0, all ngrams of that size can only occur once
* Spark NLP 3.0.3 is compatible with the new Databricks 8.2 (ML) runtime
* Spark NLP 3.0.3 is compatible with the new EMR 5.33.0 (with Zeppelin 0.9.0) release

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.0.3)**
### 3.0.2

#### John Snow Labs Spark-NLP 3.0.2: New multilingual models, confidence scores for entities and all NER tags, first support for community models, bug fixes, and other improvements!

Overview

We are glad to release Spark NLP 3.0.2! We have added some new features, improvements, trained some new multi-lingual models, and fixed some bugs in this release.

As always, we would like to thank our community for their feedback, questions, and feature requests.

New Features

* Experimental support for community models and pipelines (uploaded by users) https://github.com/JohnSnowLabs/spark-nlp/pull/2743
* Provide confidence scores for all available tags in NerDLModel and NerCrfModel https://github.com/JohnSnowLabs/spark-nlp/pull/2760

```python
# NerDLModel and NerCrfModel before 3.0.2
[[named_entity, 0, 4, B-LOC, [word -> Japan, confidence -> 0.9998], []]

# Now in Spark NLP 3.0.2
[[named_entity, 0, 4, B-LOC, [B-LOC -> 0.9998, I-ORG -> 0.0, I-MISC -> 0.0, I-LOC -> 0.0, I-PER -> 0.0, B-MISC -> 0.0, B-ORG -> 1.0E-4, word -> Japan, O -> 0.0, B-PER -> 0.0], []]
```
* Calculate confidence score for entities in NerConverter https://github.com/JohnSnowLabs/spark-nlp/pull/2784
```python
[chunk, 30, 41, Barack Obama, [entity -> PERSON, sentence -> 0, chunk -> 0, confidence -> 0.94035]
```

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.0.2)**

### 3.0.1

#### John Snow Labs Spark-NLP 3.0.1: New parameters in Normalizer, bug fixes and other improvements!

Overview

We are glad to release Spark NLP 3.0.1! We have made some improvements, added 1 line bash script to set up Google Colab and Kaggle kernel for Spark NLP 3.x, and improved our Models Hub filtering to help our community to have easier access to over 1300 pretrained models and pipelines in over 200+ languages.

As always, we would like to thank our community for their feedback, questions, and feature requests.

New Features

* Add minLength and maxLength parameters to Normalizer annotator https://github.com/JohnSnowLabs/spark-nlp/pull/2614
* 1 line to setup [Google Colab](https://github.com/JohnSnowLabs/spark-nlp#google-colab-notebook)
* 1 line to setup [Kaggle Kernel](https://github.com/JohnSnowLabs/spark-nlp#kaggle-kernel)

Enhancements

* Adjust shading rule for amazon AWS to support sub-projects from Spark NLP Fat JAR https://github.com/JohnSnowLabs/spark-nlp/pull/2613
* Fix the missing variables in BertSentenceEmbeddings https://github.com/JohnSnowLabs/spark-nlp/pull/2615
* Restrict loading Sentencepiece ops only to supported models https://github.com/JohnSnowLabs/spark-nlp/pull/2623
* improve dependency management and resolvers https://github.com/JohnSnowLabs/spark-nlp/pull/2479

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.0.1)**

### 3.0.0

#### John Snow Labs Spark-NLP 3.0.0: Supporting Spark 3.x, Scala 2.12, more Databricks runtimes, more EMR versions, performance improvements & lots more

Overview

We are very excited to release Spark NLP 3.0.0! This has been one of the biggest releases we have ever done and we are so proud to share this with our community.

Spark NLP 3.0.0 extends the support for Apache Spark 3.0.x and 3.1.x major releases on Scala 2.12 with both Hadoop 2.7. and 3.2. We will support all 4 major Apache Spark and PySpark releases of 2.3.x, 2.4.x, 3.0.x, and 3.1.x helping the community to migrate from earlier Apache Spark versions to newer releases without being worried about Spark NLP support.

As always, we would like to thank our community for their feedback, questions, and feature requests.

New Features

* Support for Apache Spark and PySpark 3.0.x on Scala 2.12
* Support for Apache Spark and PySpark 3.1.x on Scala 2.12
* Migrate to TensorFlow v2.3.1 with native support for Java to take advantage of many optimizations for CPU/GPU and new features/models introduced in TF v2.x
* Welcoming 9x new Databricks runtimes to our Spark NLP family:
  * Databricks 7.3
  * Databricks 7.3 ML GPU
  * Databricks 7.4
  * Databricks 7.4 ML GPU
  * Databricks 7.5
  * Databricks 7.5 ML GPU
  * Databricks 7.6
  * Databricks 7.6 ML GPU
  * Databricks 8.0
  * Databricks 8.0 ML (there is no GPU in 8.0)
  * Databricks 8.1 Beta
* Welcoming 2x new EMR 6.x series to our Spark NLP family:
  * EMR 6.1.0 (Apache Spark 3.0.0 / Hadoop 3.2.1)
  * EMR 6.2.0 (Apache Spark 3.0.1 / Hadoop 3.2.1)
* Starting Spark NLP 3.0.0 the default packages  for CPU and GPU will be based on Apache Spark 3.x and Scala 2.12 (`spark-nlp` and `spark-nlp-gpu` will be compatible only with Apache Spark 3.x and Scala 2.12)
* Starting Spark NLP 3.0.0 we have two new packages to support Apache Spark 2.4.x and Scala 2.11 (`spark-nlp-spark24` and `spark-nlp-gpu-spark24`)
* Spark NLP 3.0.0 still is and will be compatible with Apache Spark 2.3.x and Scala 2.11 (`spark-nlp-spark23` and `spark-nlp-gpu-spark23`)
* Adding a new param to sparknlp.start() function in Python for Apache Spark 2.4.x (`spark24=True`)
* Adding a new param to adjust Driver memory in sparknlp.start() function (`memory="16G"`)

Performance Improvements

Introducing a new batch annotation technique implemented in Spark NLP 3.0.0 for `NerDLModel`, `BertEmbeddings`, and `BertSentenceEmbeddings` annotators to radically improve prediction/inferencing performance. From now on the `batchSize` for these annotators means the number of rows that can be fed into the models for prediction instead of sentences per row. You can control the throughput when you are on accelerated hardware such as GPU to fully utilize it.

**Performance achievements by using Spark NLP 3.0.0 vs. Spark NLP 2.7.x on CPU and GPU:**

(Performed on a Databricks cluster)

| Spark NLP 3.0.0 vs. 2.7.x  |  PySpark 3.x on CPU   |  PySpark 3.x on GPU  |
|--------------------------|-----------------------|-----------------------|
|BertEmbeddings (bert-base)                         | +10%   | +550% (6.6x)
|BertEmbeddings (bert-large)                        | +12%.   | +690% (7.9x)
|NerDLModel                                                     | +185% | +327% (4.2x)

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.0.0)**

### 2.7.5

#### John Snow Labs Spark-NLP 2.7.5: Supporting more EMR versions and other improvements!

Overview

We are glad to release Spark NLP 2.7.5 release! Starting this release we no longer ship Hadoop AWS and AWS Java SDK dependencies. This change allows users to avoid any conflicts in AWS environments and also results in more EMR 5.x versions support.

As always, we would like to thank our community for their feedback, questions, and feature requests.

New Features

* Support more EMR 5.x versions
  * emr-5.20.0
  * emr-5.21.0
  * emr-5.21.1
  * emr-5.22.0
  * emr-5.23.0
  * emr-5.24.0
  * emr-5.24.1
  * emr-5.25.0
  * emr-5.26.0
  * emr-5.27.0
  * emr-5.28.0
  * emr-5.29.0
  * emr-5.30.0
  * emr-5.30.1
  * emr-5.31.0
  * emr-5.32.0

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.7.5)**

### 2.7.4

#### John Snow Labs Spark-NLP 2.7.4: New Bengali NER and Word Embeddings models, new Intent Prediction models, bug fixes, and other improvements!

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.7.4)**

### 2.7.3

#### John Snow Labs Spark-NLP 2.7.3: 18 new state-of-the-art transformer-based OntoNotes models and pipelines, new support for Bengali NER and Hindi Word Embeddings, and other improvements!

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.7.3)**

### 2.7.2

#### John Snow Labs Spark-NLP 2.7.2: New multilingual models, GPU support to train a Spell Checker, bug fixes, and other improvements!

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.7.2)**

### 2.7.1

#### John Snow Labs Spark-NLP 2.7.1: New T5 models, new TREC pipelines, bug fixes, and other improvements!

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.7.1)**

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

### 2.6.5

#### John Snow Labs Spark-NLP 2.6.5: A few bug fixes and other improvements!

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.6.5)**

### 2.6.4

#### John Snow Labs Spark-NLP 2.6.4: A few bug fixes and other improvements!

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.6.4)**

### 2.6.3

#### John Snow Labs Spark-NLP 2.6.3: New refactored NerDL with memory optimization, bug fixes, and other improvements!

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.6.3)**

### 2.6.2

#### John Snow Labs Spark-NLP 2.6.2: New SentenceDetectorDL, improved BioBERT models, new Models Hub, and other improvements!

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.6.2)**

### 2.6.1

#### John Snow Labs Spark-NLP 2.6.1: New Portuguese BERT models, import any BERT models to Spark NLP, and a bug-fix for ClassifierDL

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.6.1)**

### 2.6.0

#### John Snow Labs Spark-NLP 2.6.0: New multi-label classifier, BERT sentence embeddings, unsupervised keyword extractions, over 110 pretrained pipelines, models, Transformers, and more!

Overview

We are very excited to finally release Spark NLP 2.6.0! This has been one of the biggest releases we have ever made and we are so proud to share it with our community!

This release comes with a brand new MultiClassifierDL for multi-label text classification, BertSentenceEmbeddings with 42 models, unsupervised keyword extractions annotator, and adding 28 new pretrained Transformers such as Small BERT, CovidBERT, ELECTRA, and the state-of-the-art language-agnostic BERT Sentence Embedding model(LaBSE).

The 2.6.0 release has over 110 new pretrained models, pipelines, and Transformers with extending full support for Danish, Finnish, and Swedish languages.

Major features and improvements

* **NEW:** A new MultiClassifierDL annotator for multi-label text classification built by using Bidirectional GRU and CNN inside TensorFlow that supports up to 100 classes
* **NEW:** A new BertSentenceEmbeddings annotator with 42 available pre-trained models for sentence embeddings used in SentimentDL, ClassifierDL, and MultiClassifierDL annotators
* **NEW:** A new YakeModel annotator for an unsupervised, corpus-independent, domain, and language-independent and single-document keyword extraction algorithm
* **NEW:** Integrate 24 new Small BERT models where the smallest model is 24x times smaller and 28x times faster compare to BERT base models
* **NEW:** Add 3 new ELECTRA small, base, and large models
* **NEW:** Add 4 new Finnish BERT models for BertEmbeddings and BertSentenceEmbeddings
* Improve BertEmbeddings memory consumption by 30%
* Improve BertEmbeddings performance by more than 70% with a new built-in dynamic shape inputs
* Remove the poolingLayer parameter in BertEmbeddings in favor of sequence_output that is provided by TF Hub models for new BERT models
* Add validation loss, validation accuracy, validation F1, and validation True Positive Rate during the training in MultiClassifierDL
* Add parameter to enable/disable list detection in SentenceDetector
* Unify the loggings in ClassifierDL and SentimentDL during training

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.6.0)**

### 2.5.5

#### John Snow Labs Spark-NLP 2.5.5: 28 new Lemma and POS models in 14 languages, bug fixes, and lots of new notebooks!

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.5.5)**

### 2.5.4

#### John Snow Labs Spark-NLP 2.5.4: Supporting Apache Spark 2.3, 43 new models and 26 new languages, new RegexTokenizer, lots of new notebooks, and more

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.5.4)**

### 2.5.3

#### John Snow Labs Spark-NLP 2.5.3: Detect Fake news, emotions, spams, and more classification models, enhancements, and bug fixes

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.5.3)**

### 2.5.2

#### John Snow Labs Spark-NLP 2.5.2: New Language Detection annotator, enhancements, and bug fixes

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.5.2)**

### 2.5.1

#### John Snow Labs Spark-NLP 2.5.1: Adding support for 6 new BioBERT and ClinicalBERT models

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.5.1)**

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

### 2.4.5

#### John Snow Labs Spark-NLP 2.4.5: Supporting more Databricks runtimes and YARN in cluster mode

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.4.5)**

### 2.4.4

#### John Snow Labs Spark-NLP 2.4.4: The very first native multi-class text classifier and pre-trained models and pipelines in Russian

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.4.4)**

### 2.4.3

#### John Snow Labs Spark-NLP 2.4.3: Minor bug fix in Python

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.4.3)**

### 2.4.2

#### John Snow Labs Spark-NLP 2.4.2: Minor bug fixes and improvements

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.4.2)**

### 2.4.1

#### John Snow Labs Spark-NLP 2.4.1: Bug fixes and the very first Spanish models & pipelines

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.4.1)**

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