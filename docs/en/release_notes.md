---
layout: docs
header: true
title: Spark NLP release notes
permalink: /docs/en/release_notes
key: docs-release-notes
modify_date: "2021-08-10"
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

Models and Pipelines

Spark NLP 3.2.0 comes with new LongformerEmbeddings, BertForTokenClassification, and DistilBertForTokenClassification annotators. 

#### New Longformer Models

| Model                | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| LongformerEmbeddings | [longformer_base_4096](https://nlp.johnsnowlabs.com/2021/08/04/longformer_base_4096_en.html) | 3.2.0 |      `en`
| LongformerEmbeddings | [longformer_large_4096](https://nlp.johnsnowlabs.com/2021/08/04/longformer_large_4096_en.html) | 3.2.0 |      `en`

#### Featured NerDL Models

New NER models for **CoNLL** (4 entities) and **OntoNotes** (18 entities) trained by using **BERT**, **RoBERTa**, **DistilBERT**, **XLM-RoBERTa**, and **Longformer** Embeddings:

| Model                | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| NerDLModel     | [ner_ontonotes_roberta_base](https://nlp.johnsnowlabs.com/2021/08/04/ner_ontonotes_roberta_base_en.html) | 3.2.0 |      `en`
| NerDLModel     | [ner_ontonotes_roberta_large](https://nlp.johnsnowlabs.com/2021/08/04/ner_ontonotes_roberta_large_en.html) | 3.2.0 |      `en`
| NerDLModel     | [ner_ontonotes_distilbert_base_cased](https://nlp.johnsnowlabs.com/2021/08/04/ner_ontonotes_distilbert_base_cased_en.html) | 3.2.0 |      `en`
| NerDLModel     | [ner_conll_bert_base_cased](https://nlp.johnsnowlabs.com/2021/08/04/ner_conll_bert_base_cased_en.html) | 3.2.0 |      `en`
| NerDLModel     | [ner_conll_distilbert_base_cased](https://nlp.johnsnowlabs.com/2021/08/04/ner_conll_distilbert_base_cased_en.html) | 3.2.0 |      `en`
| NerDLModel     | [ner_conll_roberta_base](https://nlp.johnsnowlabs.com/2021/08/04/ner_conll_roberta_base_en.html) | 3.2.0 |      `en`
| NerDLModel     | [ner_conll_roberta_large](https://nlp.johnsnowlabs.com/2021/08/04/ner_conll_roberta_large_en.html) | 3.2.0 |      `en`
| NerDLModel     | [ner_conll_xlm_roberta_base](https://nlp.johnsnowlabs.com/2021/08/04/ner_conll_xlm_roberta_base_en.html) | 3.2.0 |      `en`
| NerDLModel     | [ner_conll_longformer_base_4096](https://nlp.johnsnowlabs.com/2021/08/04/ner_conll_longformer_base_4096_en.html) | 3.2.0 |      `en`
| NerDLModel     | [ner_conll_longformer_large_4096](https://nlp.johnsnowlabs.com/2021/08/04/ner_conll_longformer_large_4096_en.html) | 3.2.0 |      `en`

#### BERT and DistilBERT for Token Classification

New BERT and DistilBERT fine-tuned for the Named Entity Recognition (NER) in **English**, **Persian**, **Spanish**, **Swedish**, and **Turkish**:


| Model                | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| BertForTokenClassification | [bert_base_token_classifier_conll03](https://nlp.johnsnowlabs.com/2021/08/05/bert_base_token_classifier_conll03_en.html) | 3.2.0 |      `en`
| BertForTokenClassification | [bert_large_token_classifier_conll03](https://nlp.johnsnowlabs.com/2021/08/05/bert_large_token_classifier_conll03_en.html) | 3.2.0 |      `en`
| BertForTokenClassification | [bert_base_token_classifier_ontonote](https://nlp.johnsnowlabs.com/2021/08/05/bert_base_token_classifier_ontonote_en.html) | 3.2.0 |      `en`
| BertForTokenClassification | [bert_large_token_classifier_ontonote](https://nlp.johnsnowlabs.com/2021/08/05/bert_large_token_classifier_ontonote_en.html) | 3.2.0 |      `en`
| BertForTokenClassification | [bert_token_classifier_parsbert_armanner](https://nlp.johnsnowlabs.com/2021/08/05/bert_token_classifier_parsbert_armanner_fa.html) | 3.2.0 |      `fa`
| BertForTokenClassification | [bert_token_classifier_parsbert_ner](https://nlp.johnsnowlabs.com/2021/08/05/bert_token_classifier_parsbert_ner_fa.html) | 3.2.0 |      `fa`
| BertForTokenClassification | [bert_token_classifier_parsbert_peymaner](https://nlp.johnsnowlabs.com/2021/08/05/bert_token_classifier_parsbert_peymaner_fa.html) | 3.2.0 |      `fa`
| BertForTokenClassification | [bert_token_classifier_turkish_ner](https://nlp.johnsnowlabs.com/2021/08/05/bert_token_classifier_turkish_ner_tr.html) | 3.2.0 |      `tr`
| BertForTokenClassification | [bert_token_classifier_spanish_ner](https://nlp.johnsnowlabs.com/2021/08/05/bert_token_classifier_spanish_ner_es.html) | 3.2.0 |      `es`
| BertForTokenClassification | [bert_token_classifier_swedish_ner](https://nlp.johnsnowlabs.com/2021/08/05/bert_token_classifier_swedish_ner_sv.html) | 3.2.0 |      `sv`
| BertForTokenClassification | [bert_base_token_classifier_few_nerd](https://nlp.johnsnowlabs.com/2021/08/08/bert_base_token_classifier_few_nerd_en.html) | 3.2.0 |      `en`
| DistilBertForTokenClassification | [distilbert_base_token_classifier_few_nerd](https://nlp.johnsnowlabs.com/2021/08/08/distilbert_base_token_classifier_few_nerd_en.html) | 3.2.0 |     `en`
| DistilBertForTokenClassification | [distilbert_base_token_classifier_conll03](https://nlp.johnsnowlabs.com/2021/08/05/distilbert_base_token_classifier_conll03_en.html) | 3.2.0 |      `en`
| DistilBertForTokenClassification | [distilbert_base_token_classifier_ontonotes](https://nlp.johnsnowlabs.com/2021/08/05/distilbert_base_token_classifier_ontonotes_en.html) | 3.2.0 |    `en`
| DistilBertForTokenClassification | [distilbert_token_classifier_persian_ner](https://nlp.johnsnowlabs.com/2021/08/05/distilbert_token_classifier_persian_ner_fa.html) | 3.2.0 |       `fa`

The complete list of all 3700+ models & pipelines in 200+ languages is available on [Models Hub](https://nlp.johnsnowlabs.com/models).

#### New Notebooks

Import hundreds of models in different languages to Spark NLP

Spark NLP | HuggingFace Notebooks | Colab
:------------ | :-------------| :----------|
LongformerEmbeddings|[HuggingFace in Spark NLP - Longformer](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20Longformer.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20Longformer.ipynb)
BertForTokenClassification|[HuggingFace in Spark NLP - BertForTokenClassification](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BertForTokenClassification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BertForTokenClassification.ipynb)
DistilBertForTokenClassification|[HuggingFace in Spark NLP - DistilBertForTokenClassification](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DistilBertForTokenClassification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DistilBertForTokenClassification.ipynb)

You can visit [Import Transformers in Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669) for more info

New Multilingual DateMatcher and MultiDateMatcher

Spark NLP | Jupyter Notebooks
:------------ | :-------------|
MultiDateMatcher |  [Date Matcehr in English](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/MultiDateMatcherMultiLanguage_en.ipynb)
MultiDateMatcher |  [Date Matcehr in French](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/french/MultiDateMatcherMultiLanguage_fr.ipynb)
MultiDateMatcher |  [Date Matcehr in German](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/german/MultiDateMatcherMultiLanguage_de.ipynb)
MultiDateMatcher |  [Date Matcehr in Italian](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/italian/MultiDateMatcherMultiLanguage_it.ipynb)
MultiDateMatcher |  [Date Matcehr in Portuguese](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/portuguese/MultiDateMatcherMultiLanguage_pt.ipynb)
MultiDateMatcher |  [Date Matcehr in Spanish](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/spanish/MultiDateMatcherMultiLanguage_es.ipynb)
GraphExtraction | [Graph Extraction Intro](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/graph-extraction/graph_extraction_intro.ipynb)
GraphExtraction | [Graph Extraction](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/graph-extraction/graph_extraction.ipynb)
GraphExtraction | [Graph Extraction Explode Entities](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/graph-extraction/graph_extraction_explode_entities.ipynb)

Documentation

* [TF Hub & HuggingFace to Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669) 
* [Models Hub](https://nlp.johnsnowlabs.com/models) with new models
* [Spark NLP publications](https://medium.com/spark-nlp)
* [Spark NLP in Action](https://nlp.johnsnowlabs.com/demo)
* [Spark NLP documentation](https://nlp.johnsnowlabs.com/docs/en/quickstart)
* [Spark NLP Scala APIs](https://nlp.johnsnowlabs.com/api)
* [Spark NLP Python APIs](https://nlp.johnsnowlabs.com/api/python)
* [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks
* [Spark NLP training certification notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public) for Google Colab and Databricks
* [Spark NLP Display](https://github.com/JohnSnowLabs/spark-nlp-display) for visualization of different types of annotations
* [Discussions](https://github.com/JohnSnowLabs/spark-nlp/discussions) Engage with other community members, share ideas, and show off how you use Spark NLP!

Installation

**Python**

```shell
#PyPI

pip install spark-nlp==3.2.0
```

**Spark Packages**

**spark-nlp** on Apache Spark 3.0.x and 3.1.x (Scala 2.12 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.2.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.2.0
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:3.2.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:3.2.0
```

**spark-nlp** on Apache Spark 2.4.x (Scala 2.11 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:3.2.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:3.2.0
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark24_2.11:3.2.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark24_2.11:3.2.0
```

**spark-nlp** on Apache Spark 2.3.x (Scala 2.11 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:3.2.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:3.2.0
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:3.2.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark23_2.11:3.2.0
```

**Maven**

**spark-nlp** on Apache Spark 3.0.x and 3.1.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.12</artifactId>
    <version>3.2.0</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.12</artifactId>
    <version>3.2.0</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark24_2.11</artifactId>
    <version>3.2.0</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark24_2.11</artifactId>
    <version>3.2.0</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>3.2.0</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>3.2.0</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-assembly-3.2.0.jar

* GPU on Apache Spark 3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-assembly-3.2.0.jar

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark24-assembly-3.2.0.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-spark24-assembly-3.2.0.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark23-assembly-3.2.0.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-spark23-assembly-3.2.0.jar

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

New Models

We have trained multilingual NER models by using the entire `XTREME` (40 languages) and `WIKINER` (8 languages). 

**Multilingual Named Entity Recognition:**

| Model                | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| NerDLModel      | [ner_xtreme_xlm_roberta_xtreme_base](https://nlp.johnsnowlabs.com/2021/07/19/ner_xtreme_xlm_roberta_xtreme_base_xx.html) | 3.1.3 |      `xx`
| NerDLModel      | [ner_xtreme_glove_840B_300](https://nlp.johnsnowlabs.com/2021/07/19/ner_xtreme_glove_840B_300_xx.html) | 3.1.3 |      `xx`
| NerDLModel      | [ner_wikiner_xlm_roberta_base](https://nlp.johnsnowlabs.com/2021/07/19/ner_wikiner_xlm_roberta_base_xx.html) | 3.1.3 |      `xx`
| NerDLModel      | [ner_wikiner_glove_840B_300](https://nlp.johnsnowlabs.com/2021/07/19/ner_wikiner_glove_840B_300_xx.html) | 3.1.3 |      `xx`
| NerDLModel      | [ner_mit_movie_simple_distilbert_base_cased](https://nlp.johnsnowlabs.com/2021/07/20/ner_mit_movie_simple_distilbert_base_cased_en.html) | 3.1.3 |      `en`
| NerDLModel      | [ner_mit_movie_complex_distilbert_base_cased](https://nlp.johnsnowlabs.com/2021/07/20/ner_mit_movie_complex_distilbert_base_cased_en.html) | 3.1.3 |      `en`
| NerDLModel      | [ner_mit_movie_complex_bert_base_cased](https://nlp.johnsnowlabs.com/2021/07/20/ner_mit_movie_complex_bert_base_cased_en.html) | 3.1.3 |      `en`

**Fine-tuned XLM-RoBERTa base model by randomly masking 15% of XTREME dataset:**

| Model                | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| XlmRoBertaEmbeddings      | [xlm_roberta_xtreme_base](https://nlp.johnsnowlabs.com/2021/07/19/xlm_roberta_xtreme_base_xx.html) | 3.1.3 |      `xx`

**New Universal Sentence Encoder trained with CMLM (English & 100+ languages):**

The models extend the BERT transformer architecture and that is why we use them with BertSentenceEmbeddings.

| Model                | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| BertSentenceEmbeddings      | [sent_bert_use_cmlm_en_base](https://nlp.johnsnowlabs.com/2021/07/20/sent_bert_use_cmlm_en_base_en.html) | 3.1.3 |      `en`
| BertSentenceEmbeddings      | [sent_bert_use_cmlm_en_large](https://nlp.johnsnowlabs.com/2021/07/20/sent_bert_use_cmlm_en_large_en.html) | 3.1.3 |      `en`
| BertSentenceEmbeddings      | [sent_bert_use_cmlm_multi_base](https://nlp.johnsnowlabs.com/2021/07/20/sent_bert_use_cmlm_multi_base_xx.html) | 3.1.3 |      `xx`
| BertSentenceEmbeddings      | [sent_bert_use_cmlm_multi_base_br](https://nlp.johnsnowlabs.com/2021/07/20/sent_bert_use_cmlm_multi_base_br_xx.html) | 3.1.3 |      `xx`

Benchmark

We used BERT base, large, and the new Universal Sentence Encoder trained with CMLM extending the BERT transformer architecture to train ClassifierDL with News dataset:

(120k training examples - 10 Epochs - 512 max sequence - Nvidia Tesla P100)

| Model | Accuracy | F1 | Duration
|:-----------------------------|:-------------------|:-----------------|:------|
|tfhub_use | 0.90 | 0.89 | 10 min
|tfhub_use_lg | 0.91 | 0.90 | 24 min
|sent_bert_base_cased | 0.92 | 0.90 | 35 min
|sent_bert_large_cased | 0.93 | 0.91 | 75 min
|sent_bert_use_cmlm_en_base | 0.934 | 0.91 | 36 min
|sent_bert_use_cmlm_en_large | 0.945 | 0.92|  72 min

The complete list of all 3700+ models & pipelines in 200+ languages is available on [Models Hub](https://nlp.johnsnowlabs.com/models).

Bug Fixes

* Fix serialization issue in NorvigSweetingModel
* Fix the issue with BertSentenceEmbeddings model in TF v2
* Update ArrayType structure to fix Finisher failing to clean up some annotators

New Notebooks

Spark NLP | TF Hub Notebooks
:------------ | :-------------|
BertEmbeddings |  [TF Hub in Spark NLP - BERT](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/TF%20Hub%20in%20Spark%20NLP%20-%20BERT.ipynb)
BertSentenceEmbeddings |  [TF Hub in Spark NLP - BERT Sentence](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/TF%20Hub%20in%20Spark%20NLP%20-%20BERT%20Sentence.ipynb)
AlbertEmbeddings |  [TF Hub in Spark NLP - ALBERT](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/TF%20Hub%20in%20Spark%20NLP%20-%20ALBERT.ipynb)

Documentation

* [HuggingFace & TF Hub to Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669) 
* [Models Hub](https://nlp.johnsnowlabs.com/models) with new models
* [Spark NLP publications](https://medium.com/spark-nlp)
* [Spark NLP in Action](https://nlp.johnsnowlabs.com/demo)
* [Spark NLP documentation](https://nlp.johnsnowlabs.com/docs/en/quickstart)
* [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks
* [Spark NLP training certification notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public) for Google Colab and Databricks
* [Spark NLP Display](https://github.com/JohnSnowLabs/spark-nlp-display) for visualization of different types of annotations
* [Discussions](https://github.com/JohnSnowLabs/spark-nlp/discussions) Engage with other community members, share ideas, and show off how you use Spark NLP!

Installation

**Python**

```shell
#PyPI

pip install spark-nlp==3.1.3
```

**Spark Packages**

**spark-nlp** on Apache Spark 3.0.x and 3.1.x (Scala 2.12 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.1.3

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.1.3
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:3.1.3

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:3.1.3
```

**spark-nlp** on Apache Spark 2.4.x (Scala 2.11 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:3.1.3

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:3.1.3
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark24_2.11:3.1.3

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark24_2.11:3.1.3
```

**spark-nlp** on Apache Spark 2.3.x (Scala 2.11 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:3.1.3

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:3.1.3
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:3.1.3

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark23_2.11:3.1.3
```

**Maven**

**spark-nlp** on Apache Spark 3.0.x and 3.1.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.12</artifactId>
    <version>3.1.3</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.12</artifactId>
    <version>3.1.3</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark24_2.11</artifactId>
    <version>3.1.3</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark24_2.11</artifactId>
    <version>3.1.3</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>3.1.3</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>3.1.3</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-assembly-3.1.3.jar

* GPU on Apache Spark 3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-assembly-3.1.3.jar

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark24-assembly-3.1.3.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-spark24-assembly-3.1.3.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark23-assembly-3.1.3.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-spark23-assembly-3.1.3.jar

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

Bug Fixes & Enhancements

* Fix issue with SymmetricDeleteModel
* Fix issue with encoding unknown bytes in RoBertaEmbeddings
* Fix issue with multi-lingual UniversalSentenceEncoder models 
* Sync params between Python and Scala for `ContextSpellChecker`
  * change setWordMaxDist to setWordMaxDistance in Scala
  * change setLMClasses to setLanguageModelClasses in Scala
  * change setWordMaxDist to setWordMaxDistance in Scala
  * change setBlackListMinFreq to setCompoundCount in Scala
  * change setClassThreshold to setClassCount in Scala
  * change setWeights to setWeightedDistPath in Scala
  * change setInitialBatchSize to setBatchSize in Python
* Sync params between Python and Scala for `ViveknSentimentApproach`
  * change setCorpusPrune to setPruneCorpus in Scala
* Sync params between Python and Scala for `RegexMatcher`
  * change setRules to setExternalRules in Scala
* Sync params between Python and Scala for `WordSegmenterApproach`
  * change setPosCol to setPosColumn
  * change setIterations to setNIterations
* Sync params between Python and Scala for `ViveknSentimentApproach`
  * change setCorpusPrune to setPruneCorpus
* Sync params between Python and Scala for `PerceptronApproach `
  * change setPosCol to setPosColumn
* Fix typos in docs: https://github.com/JohnSnowLabs/spark-nlp/pull/5766 and https://github.com/JohnSnowLabs/spark-nlp/pull/5775 thanks to @brollb

Performance Improvements

Introducing a new batch annotation technique implemented in Spark NLP 3.1.2 for XlnetEmbeddings annotator to radically improve prediction/inferencing performance. From now on the `batchSize` for these annotators means the number of rows that can be fed into the models for prediction instead of sentences per row. You can control the throughput when you are on accelerated hardware such as GPU to fully utilize it. 

Backward compatibility

We have migrated XlnetEmbeddings to TensorFlow v2, the earlier models prior to 3.1.2 won't work after this release. 
We have already updated the models and uploaded them on Models Hub. You can use `pretrained()` that takes care of it automatically or please make sure you download the new models manually.

Documentation

* [HuggingFace to Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669) 
* [Models Hub](https://nlp.johnsnowlabs.com/models) with new models
* [Spark NLP publications](https://medium.com/spark-nlp)
* [Spark NLP in Action](https://nlp.johnsnowlabs.com/demo)
* [Spark NLP documentation](https://nlp.johnsnowlabs.com/docs/en/quickstart)
* [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks
* [Spark NLP training certification notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public) for Google Colab and Databricks
* [Spark NLP Display](https://github.com/JohnSnowLabs/spark-nlp-display) for visualization of different types of annotations
* [Discussions](https://github.com/JohnSnowLabs/spark-nlp/discussions) Engage with other community members, share ideas, and show off how you use Spark NLP!

Installation

**Python**

```shell
#PyPI

pip install spark-nlp==3.1.2
```

**Spark Packages**

**spark-nlp** on Apache Spark 3.0.x and 3.1.x (Scala 2.12 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.1.2

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.1.2
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:3.1.2

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:3.1.2
```

**spark-nlp** on Apache Spark 2.4.x (Scala 2.11 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:3.1.2

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:3.1.2
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark24_2.11:3.1.2

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark24_2.11:3.1.2
```

**spark-nlp** on Apache Spark 2.3.x (Scala 2.11 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:3.1.2

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:3.1.2
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:3.1.2

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark23_2.11:3.1.2
```

**Maven**

**spark-nlp** on Apache Spark 3.0.x and 3.1.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.12</artifactId>
    <version>3.1.2</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.12</artifactId>
    <version>3.1.2</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark24_2.11</artifactId>
    <version>3.1.2</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark24_2.11</artifactId>
    <version>3.1.2</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>3.1.2</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>3.1.2</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-assembly-3.1.2.jar

* GPU on Apache Spark 3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-assembly-3.1.2.jar

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark24-assembly-3.1.2.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-spark24-assembly-3.1.2.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark23-assembly-3.1.2.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-spark23-assembly-3.1.2.jar

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

Bug Fixes & Enhancements

* Fix YakeModel issue with empty token https://github.com/JohnSnowLabs/spark-nlp/pull/5683 thanks to @shaddoxac
* Fix getAnchorDateMonth method in DateMatcher and MultiDateMatcher https://github.com/JohnSnowLabs/spark-nlp/pull/5693
* Fix the broken PubTutor class in Python https://github.com/JohnSnowLabs/spark-nlp/pull/5702
* Fix relative dates in DateMatcher and MultiDateMatcher such as `day after tomorrow` or `day before yesterday` https://github.com/JohnSnowLabs/spark-nlp/pull/5706
* Add isPaddedToken param to PubTutor https://github.com/JohnSnowLabs/spark-nlp/pull/5702
* Fix issue with `logger` inside session on some setup https://github.com/JohnSnowLabs/spark-nlp/pull/5715
* Add signatures to TF session to handle inputs/outputs more dynamically in BertEmbeddings, DistilBertEmbeddings, RoBertaEmbeddings, and XlmRoBertaEmbeddings https://github.com/JohnSnowLabs/spark-nlp/pull/5715
* Fix XlmRoBertaEmbeddings issue with `init_all_tables` https://github.com/JohnSnowLabs/spark-nlp/pull/5715
* Add missing YakeModel from annotators 
* Add missing random seed param to ClassifierDLApproach, MultiClassifierDLApproach, and SentimentDLApproach https://github.com/JohnSnowLabs/spark-nlp/pull/5697
* Make the Java Exceptions appear before Py4J exceptions for ease of debugging in Python https://github.com/JohnSnowLabs/spark-nlp/pull/5709 
* Make sure batchSize set in NerDLModel is the same internally to feed TensorFlow https://github.com/JohnSnowLabs/spark-nlp/pull/5716
* Fix a typo in documentation https://github.com/JohnSnowLabs/spark-nlp/pull/5664 thanks to @roger-yu-ds

Performance Improvements

Introducing a new batch annotation technique implemented in Spark NLP 3.1.1 for AlbertEmbeddings annotator to radically improve prediction/inferencing performance. From now on the `batchSize` for these annotators means the number of rows that can be fed into the models for prediction instead of sentences per row. You can control the throughput when you are on accelerated hardware such as GPU to fully utilize it. 

#### Performance achievements by using Spark NLP 2.x/3.0.x vs. Spark NLP 3.1.1

(Performed on a Databricks cluster)

| Spark NLP 2.x/3.0.x vs. 3.1.1  |  CPU   |  GPU  | 
|------------------|-------------------------|------------------------
|ALBERT Base     | 22% | 340% |  
|Albert Large       | 20% | 770% |  

We will update this benchmark table in future pre-releases.

Backward compatibility

We have migrated AlbertEmbeddings to TensorFlow v2, the earlier models prior to 3.1.1 won't work after this release. We have already updated the models and uploaded them on Models Hub. You can use `pretrained()` that takes care of it automatically or please make sure you download the new models manually.

Documentation

* [HuggingFace to Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669) 
* [Models Hub](https://nlp.johnsnowlabs.com/models) with new models
* [Spark NLP publications](https://medium.com/spark-nlp)
* [Spark NLP in Action](https://nlp.johnsnowlabs.com/demo)
* [Spark NLP documentation](https://nlp.johnsnowlabs.com/docs/en/quickstart)
* [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks
* [Spark NLP training certification notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public) for Google Colab and Databricks
* [Spark NLP Display](https://github.com/JohnSnowLabs/spark-nlp-display) for visualization of different types of annotations
* [Discussions](https://github.com/JohnSnowLabs/spark-nlp/discussions) Engage with other community members, share ideas, and show off how you use Spark NLP!


Installation

**Python**

```shell
#PyPI

pip install spark-nlp==3.1.1
```

**Spark Packages**

**spark-nlp** on Apache Spark 3.0.x and 3.1.x (Scala 2.12 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.1.1

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.1.1
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:3.1.1

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:3.1.1
```

**spark-nlp** on Apache Spark 2.4.x (Scala 2.11 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:3.1.1

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:3.1.1
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark24_2.11:3.1.1

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark24_2.11:3.1.1
```

**spark-nlp** on Apache Spark 2.3.x (Scala 2.11 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:3.1.1

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:3.1.1
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:3.1.1

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark23_2.11:3.1.1
```

**Maven**

**spark-nlp** on Apache Spark 3.0.x and 3.1.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.12</artifactId>
    <version>3.1.1</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.12</artifactId>
    <version>3.1.1</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark24_2.11</artifactId>
    <version>3.1.1</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark24_2.11</artifactId>
    <version>3.1.1</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>3.1.1</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>3.1.1</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-assembly-3.1.1.jar

* GPU on Apache Spark 3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-assembly-3.1.1.jar

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark24-assembly-3.1.1.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-spark24-assembly-3.1.1.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark23-assembly-3.1.1.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-spark23-assembly-3.1.1.jar


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

Models and Pipelines

Spark NLP 3.1.0 comes with over 2600+ new pretrained models and pipelines in over 200 languages available for Windows, Linux, and macOS users. 

**Featured Transformers:**

| Model                | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| BertEmbeddings           | [bert_base_dutch_cased](https://nlp.johnsnowlabs.com/2021/05/20/bert_base_dutch_cased_nl.html) | 3.1.0 |      `nl`
| BertEmbeddings           | [bert_base_german_cased](https://nlp.johnsnowlabs.com/2021/05/20/bert_base_german_cased_de.html) | 3.1.0 |      `de`
| BertEmbeddings           | [bert_base_german_uncased](https://nlp.johnsnowlabs.com/2021/05/20/bert_base_german_uncased_de.html) | 3.1.0 |      `de`
| BertEmbeddings           | [bert_base_italian_cased](https://nlp.johnsnowlabs.com/2021/05/20/bert_base_italian_cased_it.html) | 3.1.0 |      `it`
| BertEmbeddings           | [bert_base_italian_uncased](https://nlp.johnsnowlabs.com/2021/05/20/bert_base_italian_uncased_it.html) | 3.1.0 | `it`
| BertEmbeddings           | [bert_base_turkish_cased](https://nlp.johnsnowlabs.com/2021/05/20/bert_base_turkish_cased_tr.html) | 3.1.0 |      `tr`
| BertEmbeddings           | [bert_base_turkish_uncased](https://nlp.johnsnowlabs.com/2021/05/20/bert_base_turkish_uncased_tr.html) | 3.1.0 |      `tr`
| BertEmbeddings           | [chinese_bert_wwm](https://nlp.johnsnowlabs.com/2021/05/20/chinese_bert_wwm_zh.html) | 3.1.0 |      `zh`
| BertEmbeddings           | [bert_base_chinese](https://nlp.johnsnowlabs.com/2021/05/20/bert_base_chinese_zh.html) | 3.1.0 |      `zh`
| DistilBertEmbeddings           | [distilbert_base_cased](https://nlp.johnsnowlabs.com/2021/05/20/distilbert_base_cased_en.html) | 3.1.0 |      `en`
| DistilBertEmbeddings           | [distilbert_base_uncased](https://nlp.johnsnowlabs.com/2021/05/20/distilbert_base_uncased_en.html) | 3.1.0 |      `en`
| DistilBertEmbeddings           | [distilbert_base_multilingual_cased](https://nlp.johnsnowlabs.com/2021/05/20/distilbert_base_multilingual_cased_xx.html) | 3.1.0 |      `xx`
| RoBertaEmbeddings           | [roberta_base](https://nlp.johnsnowlabs.com/2021/05/20/roberta_base_en.html) | 3.1.0 |      `en`
| RoBertaEmbeddings           | [roberta_large](https://nlp.johnsnowlabs.com/2021/05/20/roberta_large_en.html) | 3.1.0 |      `en`
| RoBertaEmbeddings           | [distilroberta_base](https://nlp.johnsnowlabs.com/2021/05/20/distilroberta_base_en.html) | 3.1.0 |      `en`
| XlmRoBertaEmbeddings     | [xlm_roberta_base](https://nlp.johnsnowlabs.com/2021/05/25/xlm_roberta_base_xx.html) | 3.1.0 |      `xx`
| XlmRoBertaEmbeddings     | [twitter_xlm_roberta_base](https://nlp.johnsnowlabs.com/2021/05/25/twitter_xlm_roberta_base_xx.html) | 3.1.0 |      `xx`

**Featured Translation Models:**

| Model                | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
|  MarianTransformer    | [Chinese to Vietnamese](https://nlp.johnsnowlabs.com/2021/06/01/opus_mt_zh_vi_xx.html) | 3.1.0 |      `xx`
|  MarianTransformer    | [Chinese to Ukrainian](https://nlp.johnsnowlabs.com/2021/06/01/opus_mt_zh_uk_xx.html) | 3.1.0 |      `xx`
|  MarianTransformer    | [Chinese to Dutch](https://nlp.johnsnowlabs.com/2021/06/01/opus_mt_zh_nl_xx.html) | 3.1.0 |      `xx`
|  MarianTransformer    | [Chinese to English](https://nlp.johnsnowlabs.com/2021/06/01/opus_mt_zh_en_xx.html) | 3.1.0 |      `xx`
|  MarianTransformer    | [Chinese to Finnish](https://nlp.johnsnowlabs.com/2021/06/01/opus_mt_zh_fi_xx.html) | 3.1.0 |      `xx`
|  MarianTransformer    | [Chinese to Italian](https://nlp.johnsnowlabs.com/2021/06/01/opus_mt_zh_it_xx.html) | 3.1.0 |      `xx`
|  MarianTransformer    | [Yoruba to English](https://nlp.johnsnowlabs.com/2021/06/01/opus_mt_yo_en_xx.html) | 3.1.0 |      `xx`
|  MarianTransformer    | [Yapese to French](https://nlp.johnsnowlabs.com/2021/06/01/opus_mt_yap_fr_xx.html) | 3.1.0 |      `xx`
|  MarianTransformer    | [Waray to Spanish](https://nlp.johnsnowlabs.com/2021/06/01/opus_mt_war_es_xx.html) | 3.1.0 |      `xx`
|  MarianTransformer    | [Ukrainian to English](https://nlp.johnsnowlabs.com/2021/06/01/opus_mt_uk_en_xx.html) | 3.1.0 |      `xx`
|  MarianTransformer    | [Hindi to Urdu](https://nlp.johnsnowlabs.com/2021/06/01/opus_mt_hi_ur_xx.html) | 3.1.0 |      `xx`
|  MarianTransformer    | [Italian to Ukrainian](https://nlp.johnsnowlabs.com/2021/06/01/opus_mt_it_uk_xx.html) | 3.1.0 |      `xx`
|  MarianTransformer    | [Italian to Icelandic](https://nlp.johnsnowlabs.com/2021/06/01/opus_mt_it_is_xx.html) | 3.1.0 |      `xx`

**Transformers in Spark NLP:**

Import hundreds of models in different languages to Spark NLP

Spark NLP | HuggingFace Notebooks
:------------ | :-------------|
BertEmbeddings |  [HuggingFace in Spark NLP - BERT](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BERT.ipynb)
BertSentenceEmbeddings | [HuggingFace in Spark NLP - BERT Sentence](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BERT%20Sentence.ipynb)
DistilBertEmbeddings| [HuggingFace in Spark NLP - DistilBERT](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DistilBERT.ipynb)  
RoBertaEmbeddings | [HuggingFace in Spark NLP - RoBERTa](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20RoBERTa.ipynb)  
XlmRoBertaEmbeddings | [HuggingFace in Spark NLP - XLM-RoBERTa](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20XLM-RoBERTa.ipynb)

The complete list of all 3700+ models & pipelines in 200+ languages is available on [Models Hub](https://nlp.johnsnowlabs.com/models).

Backward compatibility

* We have updated our MarianTransformer annotator to be compatible with TF v2 models. This change is not compatible with previous models/pipelines. However, we have updated and uploaded all the models and pipelines for `3.1.x` release. You can either use `MarianTransformer.pretrained(MODEL_NAME)` and it will automatically download the compatible model or you can visit [Models Hub](https://nlp.johnsnowlabs.com/models) to download the compatible models for offline use via `MarianTransformer.load(PATH)`

Documentation

* [HuggingFace to Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669) 
* [Models Hub](https://nlp.johnsnowlabs.com/models) with new models
* [Spark NLP publications](https://medium.com/spark-nlp)
* [Spark NLP in Action](https://nlp.johnsnowlabs.com/demo)
* [Spark NLP documentation](https://nlp.johnsnowlabs.com/docs/en/quickstart)
* [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks
* [Spark NLP training certification notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public) for Google Colab and Databricks
* [Spark NLP Display](https://github.com/JohnSnowLabs/spark-nlp-display) for visualization of different types of annotations
* [Discussions](https://github.com/JohnSnowLabs/spark-nlp/discussions) Engage with other community members, share ideas, and show off how you use Spark NLP!

Installation

**Python**

```shell
#PyPI

pip install spark-nlp==3.1.0
```

**Spark Packages**

**spark-nlp** on Apache Spark 3.0.x and 3.1.x (Scala 2.12 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.1.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.1.0
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:3.1.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:3.1.0
```

**spark-nlp** on Apache Spark 2.4.x (Scala 2.11 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:3.1.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:3.1.0
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark24_2.11:3.1.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark24_2.11:3.1.0
```

**spark-nlp** on Apache Spark 2.3.x (Scala 2.11 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:3.1.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:3.1.0
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:3.1.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark23_2.11:3.1.0
```

**Maven**

**spark-nlp** on Apache Spark 3.0.x and 3.1.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.12</artifactId>
    <version>3.1.0</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.12</artifactId>
    <version>3.1.0</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark24_2.11</artifactId>
    <version>3.1.0</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark24_2.11</artifactId>
    <version>3.1.0</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>3.1.0</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>3.1.0</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-assembly-3.1.0.jar

* GPU on Apache Spark 3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-assembly-3.1.0.jar

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark24-assembly-3.1.0.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-spark24-assembly-3.1.0.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark23-assembly-3.1.0.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-spark23-assembly-3.1.0.jar

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

Bug Fixes

* Fix ChunkEmbeddings Array out of bounds exception https://github.com/JohnSnowLabs/spark-nlp/pull/2796
* Fix pretrained tfhub_use_multi and tfhub_use_multi_lg models in UniversalSentenceEncoder https://github.com/JohnSnowLabs/spark-nlp/pull/2827
* Fix anchorDateMonth in Python that resulted in 1 additional month and case sensitivity to some relative dates like `next friday` or `next Friday` https://github.com/JohnSnowLabs/spark-nlp/pull/2848

Models and Pipelines

New multilingual models and pipelines for `Farsi`, `Hebrew`, `Korean`, and `Turkish`

| Model                | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| ClassifierDLModel | [classifierdl_bert_news](https://nlp.johnsnowlabs.com/2021/05/03/classifierdl_bert_news_tr.html) | 3.0.2 |  `tr`
| UniversalSentenceEncoder | [tfhub_use_multi](https://nlp.johnsnowlabs.com/2021/05/06/tfhub_use_multi_xx.html) | 3.0.0 |  `xx`
| UniversalSentenceEncoder | [tfhub_use_multi_lg](https://nlp.johnsnowlabs.com/2021/05/06/tfhub_use_multi_lg_xx.html) | 3.0.0 |  `xx`

| Pipeline                | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| PretrainedPipeline | [recognize_entities_dl](https://nlp.johnsnowlabs.com/2021/04/26/recognize_entities_dl_fa.html) | 3.0.0 |  `fa`
| PretrainedPipeline | [explain_document_lg](https://nlp.johnsnowlabs.com/2021/04/30/explain_document_lg_he.html) | 3.0.2 |  `he`
| PretrainedPipeline | [explain_document_lg](https://nlp.johnsnowlabs.com/2021/04/30/explain_document_lg_ko.html) | 3.0.2 |  `ko`

The complete list of all 1100+ models & pipelines in 192+ languages is available on [Models Hub](https://nlp.johnsnowlabs.com/models).

Documentation and Notebooks

* Add a new [Offline section](https://github.com/JohnSnowLabs/spark-nlp#offline) to docs
* [Installing Spark NLP and Spark OCR in air-gapped networks (offline mode)](https://medium.com/spark-nlp/installing-spark-nlp-and-spark-ocr-in-air-gapped-networks-offline-mode-f42a1ee6b7a8)
* [Models Hub](https://nlp.johnsnowlabs.com/models) with new models
* [Spark NLP publications](https://medium.com/spark-nlp)
* [Spark NLP in Action](https://nlp.johnsnowlabs.com/demo)
* [Spark NLP documentation](https://nlp.johnsnowlabs.com/docs/en/quickstart)
* [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks
* [Spark NLP training certification notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public) for Google Colab and Databricks
* [Spark NLP Display](https://github.com/JohnSnowLabs/spark-nlp-display) for visualization of different types of annotations
* [Discussions](https://github.com/JohnSnowLabs/spark-nlp/discussions) Engage with other community members, share ideas, and show off how you use Spark NLP!

Installation

**Python**

```shell
#PyPI

pip install spark-nlp==3.0.3
```

**Spark Packages**

**spark-nlp** on Apache Spark 3.0.x and 3.1.x (Scala 2.12 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.0.3

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.0.3
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:3.0.3

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:3.0.3
```

**spark-nlp** on Apache Spark 2.4.x (Scala 2.11 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:3.0.3

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:3.0.3
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark24_2.11:3.0.3

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark24_2.11:3.0.3
```

**spark-nlp** on Apache Spark 2.3.x (Scala 2.11 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:3.0.3

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:3.0.3
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:3.0.3

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark23_2.11:3.0.3
```

**Maven**

**spark-nlp** on Apache Spark 3.0.x and 3.1.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.12</artifactId>
    <version>3.0.3</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.12</artifactId>
    <version>3.0.3</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark24_2.11</artifactId>
    <version>3.0.3</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark24_2.11</artifactId>
    <version>3.0.3</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>3.0.3</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>3.0.3</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-assembly-3.0.3.jar

* GPU on Apache Spark 3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-assembly-3.0.3.jar

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark24-assembly-3.0.3.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-spark24-assembly-3.0.3.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark23-assembly-3.0.3.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-spark23-assembly-3.0.3.jar

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

Enhancements

* Add proper conversions for Scala 2.11/2.12 in ContextSpellChecker to use models from Spark 2.x in Spark 3.x https://github.com/JohnSnowLabs/spark-nlp/pull/2758
* Refactoring SentencePiece encoding in AlbertEmbeddings and XlnetEmbeddings https://github.com/JohnSnowLabs/spark-nlp/pull/2777

Bug Fixes

* Fix an exception in NerConverter when the documents/sentences don't carry the used tokens in NerDLModel https://github.com/JohnSnowLabs/spark-nlp/pull/2784 thanks to @rahulraina7 
* Fix an exception in AlbertEmbeddings when the original tokens are longer than the piece tokens https://github.com/JohnSnowLabs/spark-nlp/pull/2777

Models and Pipelines

New multilingual models for `Afrikaans`, `Welsh`, `Maltese`, `Tamil`, and `Vietnamese`

| Model                | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| PerceptronModel | [pos_afribooms](https://nlp.johnsnowlabs.com/2021/04/06/pos_afribooms_af.html) | 3.0.0 |  `af`
| LemmatizerModel | [lemma](https://nlp.johnsnowlabs.com/2021/04/02/lemma_cy.html) |     3.0.0     |`cy`
| LemmatizerModel | [lemma](https://nlp.johnsnowlabs.com/2021/04/02/lemma_mt.html)| 3.0.0 | `mt`
| LemmatizerModel | [lemma](https://nlp.johnsnowlabs.com/2021/04/02/lemma_af.html)| 3.0.0 | `af`
| LemmatizerModel | [lemma](https://nlp.johnsnowlabs.com/2021/04/02/lemma_ta.html)| 3.0.0 | `ta`
| LemmatizerModel | [lemma](https://nlp.johnsnowlabs.com/2021/04/02/lemma_vi.html)| 3.0.0 | `vi`

The complete list of all 1100+ models & pipelines in 192+ languages is available on [Models Hub](https://nlp.johnsnowlabs.com/models).

Documentation and Notebooks

* Add a new [Offline section](https://github.com/JohnSnowLabs/spark-nlp#offline) to docs
* [Models Hub](https://nlp.johnsnowlabs.com/models) with new models
* [Spark NLP publications](https://medium.com/spark-nlp)
* [Spark NLP in Action](https://nlp.johnsnowlabs.com/demo)
* [Spark NLP documentation](https://nlp.johnsnowlabs.com/docs/en/quickstart)
* [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks
* [Spark NLP training certification notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public) for Google Colab and Databricks
* [Spark NLP Display](https://github.com/JohnSnowLabs/spark-nlp-display) for visualization of different types of annotations
* [Discussions](https://github.com/JohnSnowLabs/spark-nlp/discussions) Engage with other community members, share ideas, and show off how you use Spark NLP!

Installation

**Python**

```shell
#PyPI

pip install spark-nlp==3.0.2
```

**Spark Packages**

**spark-nlp** on Apache Spark 3.0.x and 3.1.x (Scala 2.12 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.0.2

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.0.2
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:3.0.2

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:3.0.2
```

**spark-nlp** on Apache Spark 2.4.x (Scala 2.11 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:3.0.2

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:3.0.2
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark24_2.11:3.0.2

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark24_2.11:3.0.2
```

**spark-nlp** on Apache Spark 2.3.x (Scala 2.11 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:3.0.2

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:3.0.2
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:3.0.2

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark23_2.11:3.0.2
```

**Maven**

**spark-nlp** on Apache Spark 3.0.x and 3.1.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.12</artifactId>
    <version>3.0.2</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.12</artifactId>
    <version>3.0.2</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark24_2.11</artifactId>
    <version>3.0.2</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark24_2.11</artifactId>
    <version>3.0.2</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>3.0.2</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>3.0.2</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-assembly-3.0.2.jar

* GPU on Apache Spark 3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-assembly-3.0.2.jar

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark24-assembly-3.0.2.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-spark24-assembly-3.0.2.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark23-assembly-3.0.2.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-spark23-assembly-3.0.2.jar

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

Documentation and Notebooks

* [Models Hub](https://nlp.johnsnowlabs.com/models) with new models
* [Spark NLP publications](https://medium.com/spark-nlp)
* [Spark NLP in Action](https://nlp.johnsnowlabs.com/demo)
* [Spark NLP documentation](https://nlp.johnsnowlabs.com/docs/en/quickstart)
* [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks
* [Spark NLP training certification notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public) for Google Colab and Databricks
* [Spark NLP Display](https://github.com/JohnSnowLabs/spark-nlp-display) for visualization of different types of annotations
* [Discussions](https://github.com/JohnSnowLabs/spark-nlp/discussions) Engage with other community members, share ideas, and show off how you use Spark NLP!

Installation

**Python**

```shell
#PyPI

pip install spark-nlp==3.0.1
```

**Spark Packages**

**spark-nlp** on Apache Spark 3.0.x and 3.1.x (Scala 2.12 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.0.1

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.0.1
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:3.0.1

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:3.0.1
```

**spark-nlp** on Apache Spark 2.4.x (Scala 2.11 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:3.0.1

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:3.0.1
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark24_2.11:3.0.1

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark24_2.11:3.0.1
```

**spark-nlp** on Apache Spark 2.3.x (Scala 2.11 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:3.0.1

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:3.0.1
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:3.0.1

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark23_2.11:3.0.1
```

**Maven**

**spark-nlp** on Apache Spark 3.0.x and 3.1.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.12</artifactId>
    <version>3.0.1</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.12</artifactId>
    <version>3.0.1</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark24_2.11</artifactId>
    <version>3.0.1</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark24_2.11</artifactId>
    <version>3.0.1</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>3.0.1</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>3.0.1</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-assembly-3.0.1.jar

* GPU on Apache Spark 3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-assembly-3.0.1.jar

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark24-assembly-3.0.1.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-spark24-assembly-3.0.1.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark23-assembly-3.0.1.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-spark23-assembly-3.0.1.jar

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

Breaking changes

There are only 6 annotators that are not compatible to be used with both Scala 2.11 (Apache Spark 2.3 and Apache Spark 2.4) and Scala 2.12 (Apache Spark 3.x) at the same time. You can either train and use them on Apache Spark 2.3.x/2.4.x or train and use them on Apache Spark 3.x. 

- TokenizerModel
- PerceptronApproach (POS Tagger)
- WordSegmenter
- DependencyParser
- TypedDependencyParser
- NerCrfModel

**The rest of our models/pipelines can be used on all Apache Spark and Scala major versions without any issue.**

We have already retrained and uploaded all the exiting pretrained for Part of Speech and WordSegmenter models in Apache Spark 3.x and Scala 2.12. We will continue doing this as we see existing models which are not compatible with Apache Spark 3.x and Scala 2.12.

NOTE: You can always use the `.pretrained()` function which seamlessly will find the compatible and most recent models to download for you. It will download and extract them in your HOME DIRECTORY `~/cached_pretrained/`.

More info: [https://github.com/JohnSnowLabs/spark-nlp/discussions/2562](https://github.com/JohnSnowLabs/spark-nlp/discussions/2562)

Deprecated

Starting Spark NLP 3.0.0 release we no longer publish any artifacts on [spark-packages](https://spark-packages.org/) and we continue to host all the artifacts only [Maven Repository](https://mvnrepository.com/artifact/com.johnsnowlabs.nlp).


Documentation

* [Apache Spark Migration Guide](https://spark.apache.org/docs/3.1.1/migration-guide.html)
* [PySpark Migration Guide](https://spark.apache.org/docs/3.1.1/api/python/migration_guide/index.html)
* "Spark NLP: Natural language understanding at scale" [published paper](https://www.sciencedirect.com/science/article/pii/S2665963821000063)
* [Spark NLP publications](https://medium.com/spark-nlp)
* [Spark NLP in Action](https://nlp.johnsnowlabs.com/demo)
* [Spark NLP training certification notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public) for Google Colab and Databricks
* [Spark NLP documentation](https://nlp.johnsnowlabs.com/docs/en/quickstart)
* [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks
* [Models Hub](https://nlp.johnsnowlabs.com/models) with new models
* [Spark NLP Display](https://github.com/JohnSnowLabs/spark-nlp-display) for visualization of different types of annotations
* [Discussions](https://github.com/JohnSnowLabs/spark-nlp/discussions) Engage with other community members, share ideas, and show off how you use Spark NLP!

Installation

**Python**

```shell
#PyPI

pip install spark-nlp==3.0.0
```

**Spark Packages**

**spark-nlp** on Apache Spark 3.0.x and 3.1.x (Scala 2.12 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.0.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.0.0
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:3.0.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:3.0.0
```

**spark-nlp** on Apache Spark 2.4.x (Scala 2.11 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:3.0.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark24_2.11:3.0.0
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark24_2.11:3.0.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark24_2.11:3.0.0
```

**spark-nlp** on Apache Spark 2.3.x (Scala 2.11 only):

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:3.0.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:3.0.0
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:3.0.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu-spark23_2.11:3.0.0
```

**Maven**

**spark-nlp** on Apache Spark 3.0.x and 3.1.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.12</artifactId>
    <version>3.0.0</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.12</artifactId>
    <version>3.0.0</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark24_2.11</artifactId>
    <version>3.0.0</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark24_2.11</artifactId>
    <version>3.0.0</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>3.0.0</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>3.0.0</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-assembly-3.0.0.jar

* GPU on Apache Spark 3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-assembly-3.0.0.jar

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-assembly-3.0.0.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-assembly-3.0.0.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark23-assembly-3.0.0.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark23-gpu-assembly-3.0.0.jar

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

Bugfixes

* Fix BigDecimal error in NerDL when includeConfidence is true

Enhancements

* Shade Hadoop AWS and AWS Java SDK dependencies

Documentation and Notebooks

* "Spark NLP: Natural language understanding at scale" [published paper](https://www.sciencedirect.com/science/article/pii/S2665963821000063)
* [Spark NLP publications](https://medium.com/spark-nlp)
* [Spark NLP in Action](https://nlp.johnsnowlabs.com/demo)
* [Spark NLP training certification notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public) for Google Colab and Databricks
* [Spark NLP documentation](https://nlp.johnsnowlabs.com/docs/en/quickstart)
* [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks
* [Models Hub](https://nlp.johnsnowlabs.com/models) with new models
* New [Spark NLP Display](https://github.com/JohnSnowLabs/spark-nlp-display) for visualization of different types of annotations
* [Discussions](https://github.com/JohnSnowLabs/spark-nlp/discussions) Engage with other community members, share ideas, and show off how you use Spark NLP!

Installation

**Python**

```shell
#PyPI

pip install spark-nlp==2.7.5

#Conda

conda install -c johnsnowlabs spark-nlp==2.7.5
```

**Spark**

**spark-nlp** on Apache Spark 2.4.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.5

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.5
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.7.5

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.7.5
```

**spark-nlp** on Apache Spark 2.3.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.7.5

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.7.5
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.7.5

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.7.5
```

**Maven**

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.7.5</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.11</artifactId>
    <version>2.7.5</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>2.7.5</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>2.7.5</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-assembly-2.7.5.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-assembly-2.7.5.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark23-assembly-2.7.5.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark23-gpu-assembly-2.7.5.jar

### 2.7.4

#### John Snow Labs Spark-NLP 2.7.4: New Bengali NER and Word Embeddings models, new Intent Prediction models, bug fixes, and other improvements!

Overview

We are glad to release Spark NLP 2.7.4 release! This release comes with a few bug fixes, enhancements, and 4 new pretrained models.

As always, we would like to thank our community for their feedback, questions, and feature requests.

Bugfixes

* Fix Tensors with a 0 dimension issue in ClassifierDL and SentimentDL thanks to @pradeepgowda21 https://github.com/JohnSnowLabs/spark-nlp/pull/2288
* Fix index error in TokenAssembler https://github.com/JohnSnowLabs/spark-nlp/pull/2289
* Fix MatchError in DateMatcher and MultiDateMatcher annotators https://github.com/JohnSnowLabs/spark-nlp/pull/2297
* Fix setOutputAsArray and its default value for valueSplitSymbol in Finisher annotator https://github.com/JohnSnowLabs/spark-nlp/pull/2290

Enhancements

* Implement missing frequencyThreshold and ambiguityThreshold params in WordSegmenterApproach annotator https://github.com/JohnSnowLabs/spark-nlp/pull/2308
* Downgrade Hadoop from 3.2 to 2.7 which caused an issue with S3 https://github.com/JohnSnowLabs/spark-nlp/pull/2310
* Update Apache HTTP Client https://github.com/JohnSnowLabs/spark-nlp/pull/2312

Models and Pipelines

| Model                | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| NerDLModel  | [bengali_cc_300d](https://nlp.johnsnowlabs.com/2021/02/10/bengaliner_cc_300d_bn.html) | 2.7.3 |  `bn`
| WordEmbeddingsModel  | [bengaliner_cc_300d](https://nlp.johnsnowlabs.com/2021/02/10/bengali_cc_300d_bn.html) | 2.7.3 |  `bn`
| NerDLModel | [nerdl_snips_100d](https://nlp.johnsnowlabs.com/2021/02/15/nerdl_snips_100d_en.html)| 2.7.3 | `en`
| ClassifierDLModel | [classifierdl_use_snips](https://nlp.johnsnowlabs.com/2021/02/15/classifierdl_use_snips_en.html) | 2.7.3 | `en`

The complete list of all 1100+ models & pipelines in 192+ languages is available on [Models Hub](https://nlp.johnsnowlabs.com/models).

Compatibility 

Starting today, we have moved all of the Fat JARs hosted on our S3 to the `auxdata.johnsnowlabs.com/public/jars/` location. We have also fixed the links in the previous releases.

Documentation and Notebooks

* "Spark NLP: Natural language understanding at scale" [published paper](https://www.sciencedirect.com/science/article/pii/S2665963821000063)
* [Spark NLP publications](https://medium.com/spark-nlp)
* [Spark NLP in Action](https://nlp.johnsnowlabs.com/demo)
* [Spark NLP training certification notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public) for Google Colab and Databricks
* [Spark NLP documentation](https://nlp.johnsnowlabs.com/docs/en/quickstart)
* [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks
* [Models Hub](https://nlp.johnsnowlabs.com/models) with new models
* New [Spark NLP Display](https://github.com/JohnSnowLabs/spark-nlp-display) for visualization of different types of annotations
* [Discussions](https://github.com/JohnSnowLabs/spark-nlp/discussions) Engage with other community members, share ideas, and show off how you use Spark NLP!

Installation

**Python**

```shell
#PyPI

pip install spark-nlp==2.7.4

#Conda

conda install -c johnsnowlabs spark-nlp==2.7.4
```

**Spark**

**spark-nlp** on Apache Spark 2.4.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.4

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.4
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.7.4

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.7.4
```

**spark-nlp** on Apache Spark 2.3.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.7.4

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.7.4
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.7.4

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.7.4
```

**Maven**

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.7.4</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.11</artifactId>
    <version>2.7.4</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>2.7.4</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>2.7.4</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-assembly-2.7.4.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-assembly-2.7.4.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark23-assembly-2.7.4.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark23-gpu-assembly-2.7.4.jar

### 2.7.3

#### John Snow Labs Spark-NLP 2.7.3: 18 new state-of-the-art transformer-based OntoNotes models and pipelines, new support for Bengali NER and Hindi Word Embeddings, and other improvements!

Overview

We are glad to release Spark NLP 2.7.3 release! This release comes with a couple of bug fixes, enhancements, and 20+ pretrained models and pipelines including support for Bengali Named Entity Recognition, Hindi Word Embeddings, and state-of-the-art transformer based OntoNotes models and pipelines!

As always, we would like to thank our community for their feedback, questions, and feature requests.

New Features

* Add anchorDateYear, anchorDateMonth, and anchorDateDay to DateMatcher and MultiDateMatcher to be used for relative dates extraction

Bugfixes

* Fix the default value for action parameter in Python wrapper for DocumentNormalizer annotator
* Fix Lemmatizer pretrained models published in 2021

Enhancements

* Improve T5Transformer performance on documents with many sentences

Models and Pipelines

This release comes with support for Bengali Named Entity Recognition and Hindi Word Embeddings. We are also announcing the release of 18 new state-of-the-art transformer based OntoNotes models and pipelines! These models are trained by using Transformers pretrained models such as `BERT Tiny`, `BERT Mini`, `BERT Small`, `BERT Medium`, `BERT Base`, `BERT Large`, `ELECTRA Small`, `ELECTRA Base`, and `ELECTRA Large`.

**New Bengali and Hindi Models:**

| Model                | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| NerDLModel  | [ner_jifs_glove_840B_300d](https://nlp.johnsnowlabs.com/2021/01/27/ner_jifs_glove_840B_300d_bn.html) | 2.7.0 |      `bn`
| WordEmbeddingsModel  | [hindi_cc_300d](https://nlp.johnsnowlabs.com/2021/02/03/hindi_cc_300d_hi.html) | 2.7.0 |      `hi`

**New Transformer-based OntoNotes Models & Pipelines:**

| Model                | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| NerDLModel  | [onto_small_bert_L2_128](https://nlp.johnsnowlabs.com/2020/12/05/onto_small_bert_L2_128_en.html) | 2.7.0 |      `en`
| NerDLModel  | [onto_small_bert_L4_256](https://nlp.johnsnowlabs.com/2020/12/05/onto_small_bert_L4_256_en.html) | 2.7.0 |      `en`
| NerDLModel  | [onto_small_bert_L4_512](https://nlp.johnsnowlabs.com/2020/12/05/onto_small_bert_L4_512_en.html) | 2.7.0 |      `en`
| NerDLModel  | [onto_small_bert_L8_512](https://nlp.johnsnowlabs.com/2020/12/05/onto_small_bert_L8_512_en.html) | 2.7.0 |      `en`
| NerDLModel  | [onto_bert_base_cased](https://nlp.johnsnowlabs.com/2020/12/05/onto_bert_base_cased_en.html) | 2.7.0 |      `en`
| NerDLModel  | [onto_bert_large_cased](https://nlp.johnsnowlabs.com/2020/12/05/onto_bert_large_cased_en.html) | 2.7.0 |      `en`
| NerDLModel  | [onto_electra_small_uncased](https://nlp.johnsnowlabs.com/2020/12/05/onto_electra_small_uncased_en.html) | 2.7.0 |      `en`
| NerDLModel  | [onto_electra_base_uncased](https://nlp.johnsnowlabs.com/2020/12/05/onto_electra_base_uncased_en.html) | 2.7.0 |      `en`
| NerDLModel  | [onto_electra_large_uncased](https://nlp.johnsnowlabs.com/2020/12/05/onto_electra_large_uncased_en.html) | 2.7.0 |      `en`

| Pipeline               | Build            | Lang |  
|:-----------------------------|:-----------------|:------|
[onto_recognize_entities_bert_tiny](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_tiny_en.html) | 2.7.0 |      `en`
[onto_recognize_entities_bert_mini](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_mini_en.html) | 2.7.0 |      `en`
[onto_recognize_entities_bert_small](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_small_en.html) | 2.7.0 |      `en`
[onto_recognize_entities_bert_medium](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_medium_en.html) | 2.7.0 |      `en`
[onto_recognize_entities_bert_base](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_base_en.html) | 2.7.0 |      `en`
[onto_recognize_entities_bert_large](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_large_en.html) | 2.7.0 |      `en`
[onto_recognize_entities_electra_small](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_electra_small_en.html) | 2.7.0 |      `en`
[onto_recognize_entities_electra_base](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_electra_base_en.html) | 2.7.0 |      `en`
[onto_recognize_entities_electra_large](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_electra_large_en.html) | 2.7.0 |      `en`

**OntoNotes Benchmark:**

| SYSTEM               | YEAR            | LANGUAGE |  ONTONOTES
|:-----------------------------|:-----------------|:------|:------|
Spark NLP v2.7 | 2021 | Python/Scala/Java/R | 90.0 (test F1) 92.5 (dev F1)
spaCy 3.0 (RoBERTa) | 2020 | Python | 89.7 (dev F1)
Stanza (StanfordNLP) | 2020 | Python | 88.8 (dev F1)
Flair | 2018 | Python | 89.7

Documentation and Notebooks

* "Spark NLP: Natural language understanding at scale" [published paper](https://www.sciencedirect.com/science/article/pii/S2665963821000063)
* [Spark NLP publications](https://medium.com/spark-nlp)
* [Spark NLP in Action](https://nlp.johnsnowlabs.com/demo)
* [Spark NLP training certification notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public) for Google Colab and Databricks
* [Spark NLP documentation](https://nlp.johnsnowlabs.com/docs/en/quickstart)
* [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks
* [Models Hub](https://nlp.johnsnowlabs.com/models) with new models
* New [Spark NLP Display](https://github.com/JohnSnowLabs/spark-nlp-display) for visualization of different types of annotations
* [Discussions](https://github.com/JohnSnowLabs/spark-nlp/discussions) Engage with other community members, share ideas, and show off how you use Spark NLP!

Installation

**Python**

```shell
#PyPI

pip install spark-nlp==2.7.3

#Conda

conda install -c johnsnowlabs spark-nlp==2.7.3
```

**Spark**

**spark-nlp** on Apache Spark 2.4.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.3

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.3
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.7.3

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.7.3
```

**spark-nlp** on Apache Spark 2.3.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.7.3

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.7.3
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.7.3

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.7.3
```

**Maven**

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.7.3</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.11</artifactId>
    <version>2.7.3</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>2.7.3</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>2.7.3</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-assembly-2.7.3.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-assembly-2.7.3.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark23-assembly-2.7.3.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark23-gpu-assembly-2.7.3.jar

### 2.7.2

#### John Snow Labs Spark-NLP 2.7.2: New multilingual models, GPU support to train a Spell Checker, bug fixes, and other improvements!

Overview

We are glad to release Spark NLP 2.7.2 release! This release comes with a couple of bug fixes, enhancements, and 25+ pretrained models and pipelines in Amharic, Bengali, Bhojpuri, Japanese, and Korean languages.

As always, we would like to thank our community for their feedback, questions, and feature requests.

Bugfixes

* Fix casual mask calculations resulting in bad translation in MarianTransformer https://github.com/JohnSnowLabs/spark-nlp/pull/2149
* Fix Serialization issue in the cluster while training ContextSpellChecker https://github.com/JohnSnowLabs/spark-nlp/pull/2167
* Fix calculating CHUNK spans based on the sentences' boundaries in RegexMatcher https://github.com/JohnSnowLabs/spark-nlp/pull/2150

Enhancements

* Add GPU support for training ContextSpellChecker https://github.com/JohnSnowLabs/spark-nlp/pull/2167
* Adding Scalatest ability to control tests by tags https://github.com/JohnSnowLabs/spark-nlp/pull/2156

Models and Pipelines

The 2.7.x release comes with over 720+ new pretrained models and pipelines available for Windows, Linux, and macOS users. 

**New Text Classifier models:**

| Model                | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| SentimentDLModel  | [sentimentdl_glove_imdb](https://nlp.johnsnowlabs.com/2021/01/09/sentimentdl_glove_imdb_en.html) | 2.7.1 |      `en`
| SentimentDLModel  | [sentimentdl_use_imdb](https://nlp.johnsnowlabs.com/2021/01/15/sentimentdl_use_imdb_en.html) | 2.7.1 |      `en`
| SentimentDLModel  | [sentimentdl_use_twitter](https://nlp.johnsnowlabs.com/2021/01/18/sentimentdl_use_twitter_en.html) | 2.7.1 |      `en`
| ClassifierDLMode| [classifierdl_use_spam](https://nlp.johnsnowlabs.com/2021/01/09/classifierdl_use_spam_en.html) | 2.7.1 |  `en`
| ClassifierDLModel           | [classifierdl_use_sarcasm](https://nlp.johnsnowlabs.com/2021/01/09/classifierdl_use_sarcasm_en.html) | 2.7.1 |      `en`
| ClassifierDLModel           | [classifierdl_use_fakenews](https://nlp.johnsnowlabs.com/2021/01/09/classifierdl_use_fakenews_en.html) | 2.7.1 |      `en`
| ClassifierDLModel           | [classifierdl_use_emotion](https://nlp.johnsnowlabs.com/2021/01/09/classifierdl_use_emotion_en.html) | 2.7.1 |      `en`
| ClassifierDLModel  | [classifierdl_use_cyberbullying](https://nlp.johnsnowlabs.com/2021/01/09/classifierdl_use_cyberbullying_en.html) | 2.7.1 |      `en`
| MultiClassifierDLModel           | [multiclassifierdl_use_toxic_sm](https://nlp.johnsnowlabs.com/2021/01/21/multiclassifierdl_use_toxic_sm_en.html) | 2.7.1 |      `en`
| MultiClassifierDLModel           | [multiclassifierdl_use_toxic](https://nlp.johnsnowlabs.com/2021/01/21/multiclassifierdl_use_toxic_en.html) | 2.7.1 |      `en`
| MultiClassifierDLModel           | [multiclassifierdl_use_e2e](https://nlp.johnsnowlabs.com/2021/01/21/multiclassifierdl_use_e2e_en.html) | 2.7.1 |      `en`

**New Multi-lingual models:**
Some of the new models for Amharic, Bengali, Bhojpuri, Japanese, and Korean languages

| Model                | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| SentimentDLModel  | [sentiment_jager_use](https://nlp.johnsnowlabs.com/2021/01/14/sentiment_jager_use_th.html) | 2.7.1 |      `th`
| SentimentDLModel | [sentimentdl_urduvec_imdb](https://nlp.johnsnowlabs.com/2021/01/09/sentimentdl_urduvec_imdb_ur.html) | 2.7.1 |      `ur`
| LemmatizerModel  | [lemma](https://nlp.johnsnowlabs.com/2021/01/20/lemma_am.html) | 2.7.0 |      `am`
| LemmatizerModel  | [lemma](https://nlp.johnsnowlabs.com/2021/01/20/lemma_bn.html) | 2.7.0 |      `bn`
| LemmatizerModel  | [lemma](https://nlp.johnsnowlabs.com/2021/01/18/lemma_bh.html) | 2.7.0 |      `bh `
| LemmatizerModel  | [lemma](https://nlp.johnsnowlabs.com/2021/01/15/lemma_ja.html) | 2.7.0 |      `ja`
| LemmatizerModel  | [lemma](https://nlp.johnsnowlabs.com/2021/01/15/lemma_ko.html) | 2.7.0 |      `ko`
| PerceptronModel  | [pos_ud_att](https://nlp.johnsnowlabs.com/2021/01/20/pos_ud_att_am.html) | 2.7.0 |      `am `
| PerceptronModel  | [pos_ud_bhtb](https://nlp.johnsnowlabs.com/2021/01/18/pos_ud_bhtb_bh.html) | 2.7.0 |      `bh `
| PerceptronModel  | [pos_msri](https://nlp.johnsnowlabs.com/2021/01/20/pos_msri_bn.html) | 2.7.0 |      `bn`
| PerceptronModel  | [pos_lst20](https://nlp.johnsnowlabs.com/2021/01/13/pos_lst20_th.html) | 2.7.0 |      `th`
| WordSegmenterModel  | [wordseg_best](https://nlp.johnsnowlabs.com/2021/01/13/wordseg_best_th.html) | 2.7.0 |      `th`
| NerDLModel  | [ner_lst20_glove_840B_300d](https://nlp.johnsnowlabs.com/2021/01/11/ner_lst20_glove_840B_300d_th.html) | 2.7.0 |      `th`

The complete list of all 1100+ models & pipelines in 192+ languages is available on [Models Hub](https://nlp.johnsnowlabs.com/models).

Documentation and Notebooks

* [Spark NLP publications](https://medium.com/spark-nlp)
* [Spark NLP in Action](https://nlp.johnsnowlabs.com/demo)
* [Spark NLP training certification notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public) for Google Colab and Databricks
* [Spark NLP documentation](https://nlp.johnsnowlabs.com/docs/en/quickstart)
* [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks
* [Models Hub](https://nlp.johnsnowlabs.com/models) with new models
* New [Spark NLP Display](https://github.com/JohnSnowLabs/spark-nlp-display) for visualization of different types of annotations
* [Discussions](https://github.com/JohnSnowLabs/spark-nlp/discussions) Engage with other community members, share ideas, and show off how you use Spark NLP!

Installation

**Python**

```shell
#PyPI

pip install spark-nlp==2.7.2

#Conda

conda install -c johnsnowlabs spark-nlp==2.7.2
```

**Spark**

**spark-nlp** on Apache Spark 2.4.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.2

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.2
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.7.2

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.7.2
```

**spark-nlp** on Apache Spark 2.3.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.7.2

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.7.2
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.7.2

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.7.2
```

**Maven**

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.7.2</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.11</artifactId>
    <version>2.7.2</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>2.7.2</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>2.7.2</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-assembly-2.7.2.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-assembly-2.7.2.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark23-assembly-2.7.2.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark23-gpu-assembly-2.7.2.jar

### 2.7.1

#### John Snow Labs Spark-NLP 2.7.1: New T5 models, new TREC pipelines, bug fixes, and other improvements!

Overview

We are glad to release Spark NLP 2.7.1 towards making 2.7 stable release! This release comes with 3 new optimized T5 models, 2 new TREC pipelines, a few bug fixes, and other improvements. We highly recommend all users to upgrade to 2.7.1 for more stability while paying attention to the backward compatibility notice.

As always, we would like to thank our community for their feedback, questions, and feature requests.

Bugfixes

* Fix default pretrained model T5Transformer https://github.com/JohnSnowLabs/spark-nlp/pull/2068
* Fix default pretrained model WordSegmenter https://github.com/JohnSnowLabs/spark-nlp/pull/2068
* Fix missing reference to WordSegmenter in ResourceDwonloader https://github.com/JohnSnowLabs/spark-nlp/pull/2068
* Fix T5Transformer models crashing due to unknown task https://github.com/JohnSnowLabs/spark-nlp/pull/2070
* Fix the issue of reading and writing ClassifierDL, SentimentDL, and MultiClassifierDL models introduced in the 2.7.0 release https://github.com/JohnSnowLabs/spark-nlp/pull/2081

Enhancements

* Export new T5 models with optimized Encoder/Decoder https://github.com/JohnSnowLabs/spark-nlp/pull/2074
* Add support for alternative tagging with the positional parser in RegexTokenizer https://github.com/JohnSnowLabs/spark-nlp/pull/2077
* Refactor AssertAnnotations https://github.com/JohnSnowLabs/spark-nlp/pull/2079

Backward compatibility

* In order to fix the issue of Classifiers in the clusters, we had to export new TF models and change the read/write functions of these annotators. This caused any model trained prior to the 2.7.0 release not to be compatible with 2.7.1 and require retraining including pre-trained models. (we are re-training all the existing text classification models with 2.7.1)

Models and Pipelines

The 2.7.x release comes with over 720+ new pretrained models and pipelines available for Windows, Linux, and macOS users. 

**New optimized T5 models:**

| Model                | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| T5Transformer           | [t5_small](https://nlp.johnsnowlabs.com/2021/01/08/t5_small_en.html) | 2.7.1 |      `en`
| T5Transformer           | [t5_base](https://nlp.johnsnowlabs.com/2021/01/08/t5_base_en.html) | 2.7.1 |      `en`
| T5Transformer           | [google_t5_small_ssm_nq](https://nlp.johnsnowlabs.com/2021/01/08/google_t5_small_ssm_nq_en.html) | 2.7.1 |      `en`

**Question classification of open-domain and fact-based questions:**

| Model                | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| ClassifierDL           | [classifierdl_use_trec6](https://nlp.johnsnowlabs.com/2021/01/08/classifierdl_use_trec6_en.html) | 2.7.1 |      `en`
| ClassifierDL           | [classifierdl_use_trec50](https://nlp.johnsnowlabs.com/2021/01/08/classifierdl_use_trec50_en.html) | 2.7.1 |      `en`
| ClassifierDL           | [classifierdl_use_trec6_pipeline](https://nlp.johnsnowlabs.com/2021/01/08/classifierdl_use_trec6_pipeline_en.html) | 2.7.1 |      `en`
| ClassifierDL           | [classifierdl_use_trec50_pipeline](https://nlp.johnsnowlabs.com/2021/01/08/classifierdl_use_trec50_pipeline_en.html) | 2.7.1 |      `en`

The complete list of all 1100+ models & pipelines in 192+ languages is available on [Models Hub](https://nlp.johnsnowlabs.com/models).

Documentation and Notebooks

* New [RegexTokenizer Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/regex-tokenizer/regex_tokenizer_examples.ipynb)
* New [T5 Summarization & Question Answering Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/T5TRANSFORMER.ipynb)
* New [T5 Translation Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/T5TRANSFORMER_TRANSLATION.ipynb)
* New [Marian Translation Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/TRANSLATION_MARIAN.ipynb)
* [Spark NLP in Action](https://nlp.johnsnowlabs.com/demo)
* [Spark NLP training certification notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public) for Google Colab and Databricks
* [Spark NLP documentation](https://nlp.johnsnowlabs.com/docs/en/quickstart)
* Update the entire [spark-nlp-workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks
* Update [Models Hub](https://nlp.johnsnowlabs.com/models) with new models
* New [Spark NLP Display](https://github.com/JohnSnowLabs/spark-nlp-display) for visualization of different types of annotations
* [Discussions](https://github.com/JohnSnowLabs/spark-nlp/discussions) Engage with other community members, share ideas, and show off how you use Spark NLP!

Installation

**Python**

```shell
#PyPI

pip install spark-nlp==2.7.1

#Conda

conda install -c johnsnowlabs spark-nlp==2.7.1
```

**Spark**

**spark-nlp** on Apache Spark 2.4.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.1

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.1
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.7.1

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.7.1
```

**spark-nlp** on Apache Spark 2.3.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.7.1

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.7.1
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.7.1

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.7.1
```

**Maven**

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.7.1</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.11</artifactId>
    <version>2.7.1</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>2.7.1</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>2.7.1</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-assembly-2.7.1.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-assembly-2.7.1.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark23-assembly-2.7.1.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-spark23-gpu-assembly-2.7.1.jar

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


Bugfixes and Enhancements

* Fix all the known issues in ClassifierDL, SentimentDL, and MultiClassifierDL annotators in a Cluster
* NerDL enhancements for memory optimization and logging during the training with the test dataset
* SentenceEmbeddings annotator now reuses the storageRef of any embeddings used in prior
* Fix dropout in SentenceDetectorDL models for more deterministic results. Both English and Multi-lingual models are retrained for the 2.7.0 release
* Fix Python dataType Annotation
* Upgrade to Apache Spark 2.4.7

Models and Pipelines

The 2.7.0 release comes with over 720+ new pretrained models and pipelines available for Windows, Linux, and macOS users. 

Selected T5 and Marian models

| Model                | Name               | Build            | Lang |  
|:---------------------|:-------------------|:-----------------|:------|
| T5Transformer           | `google_t5_small_ssm_nq` | 2.7.0 |      `en`
| T5Transformer           | `t5_small`         | 2.7.0 |      `en`
| MarianTransformer       | `opus-mt-en-aav`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-af`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-afa`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-alv`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-ar`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-az`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-bat`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-bcl`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-bem`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-ber`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-bg`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-bi`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-bnt`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-bzs`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-ca`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-ceb`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-cel`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-chk`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-cpf`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-cpp`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-crs`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-cs`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-cus`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-cy`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-da`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-de`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-dra`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-ee`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-efi`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-el`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-eo`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-es`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-et`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-eu`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-euq`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-fi`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-fiu`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-fj`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-fr`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-ga`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-gaa`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-gem`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-gil`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-gl`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-gmq`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-gmw`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-grk`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-guw`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-gv`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-ha`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-he`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-hi`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-hil`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-ho`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-ht`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-hu`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-hy`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-id`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-ig`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-iir`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-ilo`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-inc`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-ine`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-is`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-iso`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-it`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-itc`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-jap`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-kg`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-kj`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-kqn`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-kwn`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-kwy`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-lg`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-ln`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-loz`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-lu`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-lua`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-lue`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-lun`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-luo`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-lus`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-map`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-mfe`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-mg`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-mh`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-mk`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-mkh`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-ml`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-mos`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-mr`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-mt`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-mul`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-ng`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-nic`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-niu`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-nl`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-nso`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-ny`    | 2.7.0 |      `xx`
| MarianTransformer       | `opus-mt-en-nyk`    | 2.7.0 |      `xx`

Chinese models

| Models                        | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
|   WordSegmenterModel  | `wordseg_weibo`  | 2.7.0 |   `zh` 
|   WordSegmenterModel  | `wordseg_pku`  | 2.7.0 |   `zh` 
|   WordSegmenterModel  | `wordseg_msra`  | 2.7.0 |   `zh` 
|   WordSegmenterModel  | `wordseg_large`  | 2.7.0 |   `zh` 
|   WordSegmenterModel  | `wordseg_ctb9`  | 2.7.0 |   `zh` 
|   PerceptronModel  | `pos_ud_gsd`  | 2.7.0 |   `zh` 
|   PerceptronModel  | `pos_ctb9`  | 2.7.0 |   `zh` 
|   NerDLModel  | `ner_msra_bert_768d`  | 2.7.0 |   `zh` 
|   NerDLModel  | `ner_weibo_bert_768d`  | 2.7.0 |   `zh` 

Arabic models

| Models                        | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
|   StopWordsCleaner  | `stopwords_ar`  | 2.7.0 |   `ar` 
|   LemmatizerModel  | `lemma`  | 2.7.0 |   `ar` 
|   PerceptronModel  | `pos_ud_padt`  | 2.7.0 |   `ar` 
|   WordEmbeddingsModel  | `arabic_w2v_cc_300d`  | 2.7.0 |   `ar` 
|   NerDLModel  | `aner_cc_300d`  | 2.7.0 |   `ar` 

Persian models

| Models                        | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
|   StopWordsCleaner  | `stopwords_fa`  | 2.7.0 |   `fa` 
|   LemmatizerModel  | `lemma`  | 2.7.0 |   `fa` 
|   PerceptronModel  | `pos_ud_perdt`  | 2.7.0 |   `fa` 
|   WordEmbeddingsModel  | `persian_w2v_cc_300d`  | 2.7.0 |   `fa` 
|   NerDLModel  | `personer_cc_300d`  | 2.7.0 |   `fa` 

The complete list of all 1100+ models & pipelines in 192+ languages is available on [Models Hub](https://nlp.johnsnowlabs.com/models).

Documentation and Notebooks

* [Spark NLP in Action](https://nlp.johnsnowlabs.com/demo)
* [Spark NLP training certification notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public) for Google Colab and Databricks
* [Spark NLP documentation](https://nlp.johnsnowlabs.com/docs/en/quickstart)
* Update the entire [spark-nlp-workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks
* Update [Models Hub](https://nlp.johnsnowlabs.com/models) with new models
* New [Spark NLP Display](https://github.com/JohnSnowLabs/spark-nlp-display) for visualization of different types of annotations
* [Discussions](https://github.com/JohnSnowLabs/spark-nlp/discussions) Engage with other community members, share ideas, and show off how you use Spark NLP!

Installation

**Python**

```shell
#PyPI

pip install spark-nlp==2.7.0

#Conda

conda install -c johnsnowlabs spark-nlp==2.7.0
```

**Spark**

**spark-nlp** on Apache Spark 2.4.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.0
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.7.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.7.0
```

**spark-nlp** on Apache Spark 2.3.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.7.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.7.0
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.7.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.7.0
```

**Maven**

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.7.0</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.11</artifactId>
    <version>2.7.0</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>2.7.0</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>2.7.0</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-assembly-2.7.0.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-gpu-assembly-2.7.0.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-spark23-assembly-2.7.0.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-spark23-gpu-assembly-2.7.0.jar

### 2.6.5

#### John Snow Labs Spark-NLP 2.6.5: A few bug fixes and other improvements!

Overview

We are glad to release Spark NLP 2.6.5! This release comes with a few bug fixes before we move to a new major version.

As always, we would like to thank our community for their feedback, questions, and feature requests.

Bugfixes

* Fix a bug in batching sentences in BertSentenceEmbeddings
* Fix AttributeError when trying to load a saved EmbeddingsFinisher in Python

Enhancements

* Improve handling exceptions in DocumentAssmbler when the user uses a corrupted DataFrame

Documentation and Notebooks

* [Spark NLP in Action](https://nlp.johnsnowlabs.com/demo)
* [Spark NLP training certification notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public) for Google Colab and Databricks
* [Spark NLP documentation](https://nlp.johnsnowlabs.com/docs/en/quickstart)
* Update the entire [spark-nlp-workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks
* A brand new [1-hour Spark NLP workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/1hr_workshop)
* Update [Model Hubs](https://nlp.johnsnowlabs.com/models) with new models

Installation

**Python**

```shell
#PyPI

pip install spark-nlp==2.6.5

#Conda

conda install -c johnsnowlabs spark-nlp==2.6.5
```

**Spark**

**spark-nlp** on Apache Spark 2.4.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.6.5

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.6.5
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.6.5

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.6.5
```

**spark-nlp** on Apache Spark 2.3.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.6.5

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.6.5
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.6.5

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.6.5
```

**Maven**

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.6.5</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.11</artifactId>
    <version>2.6.5</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>2.6.5</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>2.6.5</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-assembly-2.6.5.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-gpu-assembly-2.6.5.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-spark23-assembly-2.6.5.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-spark23-gpu-assembly-2.6.5.jar

### 2.6.4

#### John Snow Labs Spark-NLP 2.6.4: A few bug fixes and other improvements!

Overview

We are glad to release Spark NLP 2.6.4! This release comes with a few bug fixes before we move to a new major version.

As always, we would like to thank our community for their feedback, questions, and feature requests.

Bugfixes

* Fix loading from a local folder with no access to the cache folder https://github.com/JohnSnowLabs/spark-nlp/pull/1141
* Fix NullPointerException in DocumentAssembler when there are null in the rows https://github.com/JohnSnowLabs/spark-nlp/pull/1145
* Fix dynamic padding in BertSentenceEmbeddings https://github.com/JohnSnowLabs/spark-nlp/pull/1162

Documentation and Notebooks

* [Spark NLP in Action](https://nlp.johnsnowlabs.com/demo)
* [Spark NLP training certification notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public) for Google Colab and Databricks
* [Spark NLP documentation](https://nlp.johnsnowlabs.com/docs/en/quickstart)
* Update the entire [spark-nlp-workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks
* A brand new [1-hour Spark NLP workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/1hr_workshop)
* Update [Model Hubs](https://nlp.johnsnowlabs.com/models) with new models

Installation

**Python**

```shell
#PyPI

pip install spark-nlp==2.6.4

#Conda

conda install -c johnsnowlabs spark-nlp==2.6.4
```

**Spark**

**spark-nlp** on Apache Spark 2.4.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.6.4

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.6.4
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.6.4

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.6.4
```

**spark-nlp** on Apache Spark 2.3.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.6.4

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.6.4
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.6.4

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.6.4
```

**Maven**

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.6.4</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.11</artifactId>
    <version>2.6.4</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>2.6.4</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>2.6.4</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-assembly-2.6.4.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-gpu-assembly-2.6.4.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-spark23-assembly-2.6.4.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-spark23-gpu-assembly-2.6.4.jar

### 2.6.3

#### John Snow Labs Spark-NLP 2.6.3: New refactored NerDL with memory optimization, bug fixes, and other improvements!

Overview

We are glad to release Spark NLP 2.6.3! This release comes with a refactored NerDLApproach that allows users to train their NerDL on any size of the CoNLL file regardless of the memory limitations. We also have some bug fixes and improvements in the 2.6.3 release.

Spark NLP has a new and improved [Website](https://nlp.johnsnowlabs.com/) for its documentation and models. We have been moving our 330+ pretrained models and pipelines into [Models Hubs](https://nlp.johnsnowlabs.com/models) and we would appreciate your feedback! :) 

As always, we would like to thank our community for their feedback, questions, and feature requests.

New Features

* Add enableMemoryOptimizer to allow training NerDLApproach on a dataset larger than the memory
* Add option to explode sentences in SentenceDetectorDL

Enhancements

* Improve POS (AveragedPerceptron) performance
* Improve Norvig Spell Checker performance

Bugfixes

* Fix SentenceDetectorDL unsupported model error in pretrained function
* Fix a race condition in LRU algorithm that can cause NullPointerException during a LightPipeline operation with embeddings
* Fix max sequence length calculation in BertEmbeddings and BertSentenceEmbeddings
* Fix threshold in YakeModel on Python side

Documentation and Notebooks

* [Spark NLP training certification notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public) for Google Colab and Databricks
* A brand new [1-hour Spark NLP workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/1hr_workshop)
* Update [Model Hubs](https://nlp.johnsnowlabs.com/models) with new models in Spark NLP 2.6.3
* Update documentation for release of Spark NLP 2.6.3
* Update the entire [spark-nlp-workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks for Spark NLP 2.6.3

Installation

**Python**

```shell
#PyPI

pip install spark-nlp==2.6.3

#Conda

conda install -c johnsnowlabs spark-nlp==2.6.3
```

**Spark**

**spark-nlp** on Apache Spark 2.4.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.6.3

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.6.3
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.6.3

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.6.3
```

**spark-nlp** on Apache Spark 2.3.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.6.3

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.6.3
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.6.3

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.6.3
```

**Maven**

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.6.3</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.11</artifactId>
    <version>2.6.3</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>2.6.3</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>2.6.3</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-assembly-2.6.3.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-gpu-assembly-2.6.3.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-spark23-assembly-2.6.3.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-spark23-gpu-assembly-2.6.3.jar

### 2.6.2

#### John Snow Labs Spark-NLP 2.6.2: New SentenceDetectorDL, improved BioBERT models, new Models Hub, and other improvements!

Overview

We are glad to release Spark NLP 2.6.2! This release comes with a brand new SentenceDetectorDL (SDDL) that is based on a general-purpose neural network model for sentence boundary detection with higher accuracy. In addition, we are releasing 12 new and improved BioBERT models for BertEmbeddings and BertSentenceEembeddings used for sequence and text classifications.

Spark NLP has a new and improved [Website](https://nlp.johnsnowlabs.com/) for its documentation and models. We have been moving our 330+ pretrained models and pipelines into [Models Hubs](https://nlp.johnsnowlabs.com/models) and we would appreciate your feedback! :) 

As always, we would like to thank our community for their feedback, questions, and feature requests.

New Features

* Introducing a new SentenceDetectorDL (trainable) for sentence boundary detection
* Dedicated [Models Hub](https://nlp.johnsnowlabs.com/models) for all pretrained models & pipelines

Enhancements

* Improved BioBERT models quality for BertEmbeddings (it achieves higher accuracy in sequence classification)
* Improved Sentence BioBERT models quality for BertSentenceEmbeddings (it achieves higher accuracy in text classification)
* Improve loadSavedModel in BertEmbeddings and BertSentenceEmbeddings
* Add unit test to MultiClassifierDL annotator
* Better error handling in SentimentDLApproach

Bugfixes

* Fix BERT LaBSE model for BertSentenceEmbeddings
* Fix loadSavedModel for BertSentenceEmbeddings in Python

Deprecations

* DeepSentenceDetector is deprecated in favor of SentenceDetectorDL

Models

| Model                        | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| BertEmbeddings                    | `biobert_pubmed_base_cased`        | 2.6.2 |      `en`         | 
| BertEmbeddings                    | `biobert_pubmed_large_cased`        | 2.6.2 |      `en`        |
| BertEmbeddings                    | `biobert_pmc_base_cased`        | 2.6.2 |      `en`            | 
| BertEmbeddings                    | `biobert_pubmed_pmc_base_cased`        | 2.6.2 |      `en`     |
| BertEmbeddings                    | `biobert_clinical_base_cased`        | 2.6.2 |      `en`       | 
| BertEmbeddings                    | `biobert_discharge_base_cased`        | 2.6.2 |      `en`      |
| BertSentenceEmbeddings   | `sent_biobert_pubmed_base_cased`        | 2.6.2 |      `en`         | 
| BertSentenceEmbeddings   | `sent_biobert_pubmed_large_cased`        | 2.6.2 |      `en`        | 
| BertSentenceEmbeddings   | `sent_biobert_pmc_base_cased`        | 2.6.2 |      `en`            |
| BertSentenceEmbeddings   | `sent_biobert_pubmed_pmc_base_cased`        | 2.6.0 |      `en`     |
| BertSentenceEmbeddings   | `sent_biobert_clinical_base_cased`        | 2.6.2 |      `en`       |
| BertSentenceEmbeddings   | `sent_biobert_discharge_base_cased`        | 2.6.2 |      `en`      |

The complete list of all 330+ models & pipelines in 46+ languages is [available here](https://github.com/JohnSnowLabs/spark-nlp-models/).

Documentation and Notebooks

* New notebook to [use SentenceDetectorDL](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/9.SentenceDetectorDL.ipynb)
* Update [Model Hubs](https://nlp.johnsnowlabs.com/models) with new models in Spark NLP 2.6.2
* Update documentation for release of Spark NLP 2.6.2
* Update the entire [spark-nlp-workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks for Spark NLP 2.6.2

Installation

**Python**

```shell
#PyPI

pip install spark-nlp==2.6.2

#Conda

conda install -c johnsnowlabs spark-nlp==2.6.2
```

**Spark**

**spark-nlp** on Apache Spark 2.4.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.6.2

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.6.2
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.6.2

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.6.2
```

**spark-nlp** on Apache Spark 2.3.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.6.2

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.6.2
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.6.2

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.6.2
```

**Maven**

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.6.2</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.11</artifactId>
    <version>2.6.2</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>2.6.2</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>2.6.2</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-assembly-2.6.2.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-gpu-assembly-2.6.2.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-spark23-assembly-2.6.2.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-spark23-gpu-assembly-2.6.2.jar

### 2.6.1

#### John Snow Labs Spark-NLP 2.6.1: New Portuguese BERT models, import any BERT models to Spark NLP, and a bug-fix for ClassifierDL

Overview

We are glad to release Spark NLP 2.6.1! This release comes with new Portuguese BERT models, a notebook to demonstrate how to import any BERT models to Spark NLP, and a fix for ClassifierDL which was introduced in the 2.6.0 release that resulted in low accuracy during training.

As always, we would like to thank our community for their feedback, questions, and feature requests.

Bugfixes

* Fix lower accuracy in ClassifierDL introduced in 2.6.0 release

Models and Pipelines

| Model                        | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| BertEmbeddings                    | `bert_portuguese_base_cased`       | 2.6.0 |      `pt`
| BertEmbeddings                    | `bert_portuguese_large_cased`       | 2.6.0 |      `pt` 

The complete list of all 330+ models & pipelines in 46+ languages is [available here](https://github.com/JohnSnowLabs/spark-nlp-models/).

Documentation and Notebooks

* New notebook to import [BERT checkpoints into Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/portuguese/Export_BERT_model_to_Spark_NLP_BertEmbeddings.ipynb)
* New notebook to [extract keywords](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/KEYPHRASE_EXTRACTION.ipynb)
* Update documentation for release of Spark NLP 2.6.1
* Update the entire [spark-nlp-models](https://github.com/JohnSnowLabs/spark-nlp-models) repository with new pre-trained models and pipelines
* Update the entire [spark-nlp-workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks for Spark NLP 2.6.1

Installation
**Python**

```shell
#PyPI

pip install spark-nlp==2.6.1

#Conda

conda install -c johnsnowlabs spark-nlp==2.6.1
```

**Spark**

**spark-nlp** on Apache Spark 2.4.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.6.1

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.6.1
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.6.1

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.6.1
```

**spark-nlp** on Apache Spark 2.3.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.6.1

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.6.1
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.6.1

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.6.1
```

**Maven**

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.6.1</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.11</artifactId>
    <version>2.6.1</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>2.6.1</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>2.6.1</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-assembly-2.6.1.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-gpu-assembly-2.6.1.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-spark23-assembly-2.6.1.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-spark23-gpu-assembly-2.6.1.jar

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

Bugfixes

* Fix Tokenization bug with Bigrams in the exception list
* Fix the versioning error in second SBT projects causing models not being found via pretrained function
* Fix logging to file in NerDLApproach, ClassifierDL, SentimentDL, and MultiClassifierDL on HDFS
* Fix ignored modified tokens in BertEmbeddings, now it will consider modified tokens instead of originals

Models and Pipelines

This release comes with over 100+ new pretrained models and pipelines available for Windows, Linux, and macOS users. 

The complete list of all 330+ models & pipelines in 46+ languages is [available here](https://github.com/JohnSnowLabs/spark-nlp-models/).

#### Some selected Transformers:

{:.table-model-big}
| Model                        | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| BertEmbeddings                    | `electra_small_uncased`       | 2.6.0 |      `en`
| BertEmbeddings                    | `electra_base_uncased`       | 2.6.0 |      `en` 
| BertEmbeddings                    | `electra_large_uncased`       | 2.6.0 |      `en`
| BertEmbeddings                    | `covidbert_large_uncased`        | 2.6.0 |      `en`
| BertEmbeddings                    | `small_bert_L2_128`        | 2.6.0 |      `en`      
| BertEmbeddings                    | `small_bert_L4_128`        | 2.6.0 |      `en`      
| BertEmbeddings                    | `small_bert_L6_128`        | 2.6.0 |      `en`      
| BertEmbeddings                    | `small_bert_L8_128`        | 2.6.0 |      `en`      
| BertEmbeddings                    | `small_bert_L10_128`        | 2.6.0 |      `en`     
| BertEmbeddings                    | `small_bert_L12_128`        | 2.6.0 |      `en`     
| BertEmbeddings                    | `small_bert_L2_256`        | 2.6.0 |      `en`      
| BertEmbeddings                    | `small_bert_L4_256`        | 2.6.0 |      `en`      
| BertEmbeddings                    | `small_bert_L6_256`        | 2.6.0 |      `en`      
| BertEmbeddings                    | `small_bert_L8_256`        | 2.6.0 |      `en`      
| BertEmbeddings                    | `small_bert_L10_256`        | 2.6.0 |      `en`     
| BertEmbeddings                    | `small_bert_L12_256`        | 2.6.0 |      `en`     
| BertEmbeddings                    | `small_bert_L2_512`        | 2.6.0 |      `en`      
| BertEmbeddings                    | `small_bert_L4_512`        | 2.6.0 |      `en`      
| BertEmbeddings                    | `small_bert_L6_512`        | 2.6.0 |      `en`      
| BertEmbeddings                    | `small_bert_L8_512`        | 2.6.0 |      `en`      
| BertEmbeddings                    | `small_bert_L10_512`        | 2.6.0 |      `en`     
| BertEmbeddings                    | `small_bert_L12_512`        | 2.6.0 |      `en`     
| BertEmbeddings                    | `small_bert_L2_768`        | 2.6.0 |      `en`      
| BertEmbeddings                    | `small_bert_L4_768`        | 2.6.0 |      `en`      
| BertEmbeddings                    | `small_bert_L6_768`        | 2.6.0 |      `en`      
| BertEmbeddings                    | `small_bert_L8_768`        | 2.6.0 |      `en`      
| BertEmbeddings                    | `small_bert_L10_768`        | 2.6.0 |      `en`     
| BertEmbeddings                    | `small_bert_L12_768`        | 2.6.0 |      `en` 
| BertEmbeddings   | `bert_finnish_cased`       | 2.6.0 |      `fi`      
| BertEmbeddings   | `bert_finnish_uncased`       | 2.6.0 |      `fi`    
| BertSentenceEmbeddings   | `sent_bert_finnish_cased`       | 2.6.0 |   `fi`
| BertSentenceEmbeddings   | `sent_bert_finnish_uncased`       | 2.6.0 | `fi`
| BertSentenceEmbeddings   | `sent_electra_small_uncased`       | 2.6.0 |      `en`    
| BertSentenceEmbeddings   | `sent_electra_base_uncased`         | 2.6.0 |      `en`        
| BertSentenceEmbeddings   | `sent_electra_large_uncased`      | 2.6.0 |      `en`          
| BertSentenceEmbeddings   | `sent_bert_base_uncased`       | 2.6.0 |      `en`             
| BertSentenceEmbeddings   | `sent_bert_base_cased`         | 2.6.0 |      `en`             
| BertSentenceEmbeddings   | `sent_bert_large_uncased`      | 2.6.0 |      `en`             
| BertSentenceEmbeddings   | `sent_bert_large_cased`        | 2.6.0 |      `en`             
| BertSentenceEmbeddings   | `sent_biobert_pubmed_base_cased`        | 2.6.0 |      `en`    
| BertSentenceEmbeddings   | `sent_biobert_pubmed_large_cased`        | 2.6.0 |      `en`   
| BertSentenceEmbeddings   | `sent_biobert_pmc_base_cased`        | 2.6.0 |      `en`       
| BertSentenceEmbeddings   | `sent_biobert_pubmed_pmc_base_cased`        | 2.6.0 |      `en`
| BertSentenceEmbeddings   | `sent_biobert_clinical_base_cased`        | 2.6.0 |      `en`  
| BertSentenceEmbeddings   | `sent_biobert_discharge_base_cased`        | 2.6.0 |      `en` 
| BertSentenceEmbeddings   | `sent_covidbert_large_uncased`        | 2.6.0 |      `en`      
| BertSentenceEmbeddings   | `sent_small_bert_L2_128`        | 2.6.0 |      `en`            
| BertSentenceEmbeddings   | `sent_small_bert_L4_128`        | 2.6.0 |      `en`            
| BertSentenceEmbeddings   | `sent_small_bert_L6_128`        | 2.6.0 |      `en`            
| BertSentenceEmbeddings   | `sent_small_bert_L8_128`        | 2.6.0 |      `en`            
| BertSentenceEmbeddings   | `sent_small_bert_L10_128`        | 2.6.0 |      `en`           
| BertSentenceEmbeddings   | `sent_small_bert_L12_128`        | 2.6.0 |      `en`           
| BertSentenceEmbeddings   | `sent_small_bert_L2_256`        | 2.6.0 |      `en`            
| BertSentenceEmbeddings   | `sent_small_bert_L4_256`        | 2.6.0 |      `en`            
| BertSentenceEmbeddings   | `sent_small_bert_L6_256`        | 2.6.0 |      `en`            
| BertSentenceEmbeddings   | `sent_small_bert_L8_256`        | 2.6.0 |      `en`            
| BertSentenceEmbeddings   | `sent_small_bert_L10_256`        | 2.6.0 |      `en`           
| BertSentenceEmbeddings   | `sent_small_bert_L12_256`        | 2.6.0 |      `en`           
| BertSentenceEmbeddings   | `sent_small_bert_L2_512`        | 2.6.0 |      `en`            
| BertSentenceEmbeddings   | `sent_small_bert_L4_512`        | 2.6.0 |      `en`            
| BertSentenceEmbeddings   | `sent_small_bert_L6_512`        | 2.6.0 |      `en`            
| BertSentenceEmbeddings   | `sent_small_bert_L8_512`        | 2.6.0 |      `en`            
| BertSentenceEmbeddings   | `sent_small_bert_L10_512`        | 2.6.0 |      `en`           
| BertSentenceEmbeddings   | `sent_small_bert_L12_512`        | 2.6.0 |      `en`           
| BertSentenceEmbeddings   | `sent_small_bert_L2_768`        | 2.6.0 |      `en`            
| BertSentenceEmbeddings   | `sent_small_bert_L4_768`        | 2.6.0 |      `en`            
| BertSentenceEmbeddings   | `sent_small_bert_L6_768`        | 2.6.0 |      `en`            
| BertSentenceEmbeddings   | `sent_small_bert_L8_768`        | 2.6.0 |      `en`            
| BertSentenceEmbeddings   | `sent_small_bert_L10_768`        | 2.6.0 |      `en`           
| BertSentenceEmbeddings   | `sent_small_bert_L12_768`        | 2.6.0 |      `en`    
| BertSentenceEmbeddings | `sent_bert_multi_cased` | 2.6.0 |   `xx`   
| BertSentenceEmbeddings | `labse` | 2.6.0 |   `xx`  

#### Danish pipelines

{:.table-model-big}
| Pipeline                        | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| Explain Document Small    | `explain_document_sm`  | 2.6.0 |   `da` 
| Explain Document Medium   | `explain_document_md`  | 2.6.0 |   `da` 
| Explain Document Large    | `explain_document_lg`  | 2.6.0 |   `da` 
| Entity Recognizer Small   | `entity_recognizer_sm`  | 2.6.0 |   `da`
| Entity Recognizer Medium  | `entity_recognizer_md`  | 2.6.0 |   `da`
| Entity Recognizer Large   | `entity_recognizer_lg`  | 2.6.0 |   `da`

#### Finnish pipelines

{:.table-model-big}
| Pipeline                        | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| Explain Document Small    | `explain_document_sm`  | 2.6.0 |   `fi` 
| Explain Document Medium   | `explain_document_md`  | 2.6.0 |   `fi` 
| Explain Document Large    | `explain_document_lg`  | 2.6.0 |   `fi` 
| Entity Recognizer Small   | `entity_recognizer_sm`  | 2.6.0 |   `fi`
| Entity Recognizer Medium  | `entity_recognizer_md`  | 2.6.0 |   `fi`
| Entity Recognizer Large   | `entity_recognizer_lg`  | 2.6.0 |   `fi`

#### Swedish pipelines

{:.table-model-big}
| Pipeline                        | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| Explain Document Small    | `explain_document_sm`  | 2.6.0 |   `sv` 
| Explain Document Medium   | `explain_document_md`  | 2.6.0 |   `sv` 
| Explain Document Large    | `explain_document_lg`  | 2.6.0 |   `sv`
| Entity Recognizer Small   | `entity_recognizer_sm`  | 2.6.0 |   `sv`
| Entity Recognizer Medium  | `entity_recognizer_md`  | 2.6.0 |   `sv`
| Entity Recognizer Large   | `entity_recognizer_lg`  | 2.6.0 |   `sv`

Documentation and Notebooks

* New notebook for training multi-label [Toxic comments](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/classification/MultiClassifierDL_train_multi_label_toxic_classifier.ipynb)
* New notebook for training multi-label [E2E Challenge](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/classification/MultiClassifierDL_train_multi_label_E2E_challenge_classifier.ipynb)
* Update documentation for release of Spark NLP 2.6.0
* Update the entire [spark-nlp-models](https://github.com/JohnSnowLabs/spark-nlp-models) repository with new pre-trained models and pipelines
* Update the entire [spark-nlp-workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks for Spark NLP 2.6.0

Installation

**Python**

```shell
#PyPI

pip install spark-nlp==2.6.0

#Conda

conda install -c johnsnowlabs spark-nlp==2.6.0
```

**Spark**

**spark-nlp** on Apache Spark 2.4.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.6.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.6.0
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.6.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.6.0
```

**spark-nlp** on Apache Spark 2.3.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.6.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.6.0
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.6.0

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.6.0
```

**Maven**

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.6.0</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.11</artifactId>
    <version>2.6.0</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>2.6.0</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>2.6.0</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-assembly-2.6.0.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-gpu-assembly-2.6.0.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-spark23-assembly-2.6.0.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-spark23-gpu-assembly-2.6.0.jar

### 2.5.5

#### John Snow Labs Spark-NLP 2.5.5: 28 new Lemma and POS models in 14 languages, bug fixes, and lots of new notebooks!

Overview

We are excited to release Spark NLP 2.5.5 with 28 new pretrained models for Lemma and POS in 14 languages, bug fixes, new notebooks, and more!

As always, we would like to thank our community for their feedback, questions, and feature requests.

New Features

- Add getClasses() function to NerDLModel
- Add getClasses() function to ClassifierDLModel
- Add  getClasses() function to SentimentDLModel

Example:
```python
ner_model = NerDLModel.pretrained('onto_100')
print(ner_model.getClasses())
#['O', 'B-CARDINAL', 'B-EVENT', 'I-EVENT', 'B-WORK_OF_ART', 'I-WORK_OF_ART', 'B-ORG', 'B-DATE', 'I-DATE', 'I-ORG', 'B-GPE', 'B-PERSON', 'B-PRODUCT', 'B-NORP', 'B-ORDINAL', 'I-PERSON', 'B-MONEY', 'I-MONEY', 'I-GPE', 'B-LOC', 'I-LOC', 'I-CARDINAL', 'B-FAC', 'I-FAC', 'B-LAW', 'I-LAW', 'B-TIME', 'I-TIME', 'B-PERCENT', 'I-PERCENT', 'I-NORP', 'I-PRODUCT', 'B-QUANTITY', 'I-QUANTITY', 'B-LANGUAGE', 'I-ORDINAL', 'I-LANGUAGE', 'X']
```

Enhancements

- Improve max sequence length calculation in BertEmbeddings and XlnetEmbeddings

Bugfixes

- Fix a bug in RegexTokenizer in Python
- Fix StopWordsCleaner exception in Python when pretrained() is used
- Fix max sequence length issue in AlbertEmbeddings and SentencePiece generation
- Fix HDFS support for setGaphFolder param in NerDLApproach

Models

* We have added 28 new pretrained models for Lemma and POS in 14 languages:

{:.table-model-big}
| Model                        | Name               | Build            | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.5 |   `br`    
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.5 |   `ca`    
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.5 |   `da`    
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.5 |   `ga`    
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.5 |   `hi`    
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.5 |   `hy`    
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.5 |   `eu`    
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.5 |   `mr`    
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.5 |   `yo`    
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.5 |   `la`    
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.5 |   `lv`    
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.5 |   `sl`    
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.5 |   `gl`    
| LemmatizerModel (Lemmatizer) | `lemma`            | 2.5.5 |   `id`    
| PerceptronModel (POS UD) | `pos_ud_keb`            | 2.5.5 |   `br`    
| PerceptronModel (POS UD) | `pos_ud_ancora`            | 2.5.5 |   `ca`    
| PerceptronModel (POS UD) | `pos_ud_ddt`            | 2.5.5 |   `da`    
| PerceptronModel (POS UD) | `pos_ud_idt`            | 2.5.5 |   `ga`    
| PerceptronModel (POS UD) | `pos_ud_hdtb`            | 2.5.5 |   `hi`    
| PerceptronModel (POS UD) | `pos_ud_armtdp`            | 2.5.5 |   `hy`    
| PerceptronModel (POS UD) | `pos_ud_bdt`            | 2.5.5 |   `eu`    
| PerceptronModel (POS UD) | `pos_ud_ufal`            | 2.5.5 |   `mr`    
| PerceptronModel (POS UD) | `pos_ud_ytb`            | 2.5.5 |   `yo`    
| PerceptronModel (POS UD) | `pos_ud_llct`            | 2.5.5 |   `la`    
| PerceptronModel (POS UD) | `pos_ud_lvtb`            | 2.5.5 |   `lv`    
| PerceptronModel (POS UD) | `pos_ud_ssj`            | 2.5.5 |   `sl`    
| PerceptronModel (POS UD) | `pos_ud_treegal`            | 2.5.5 |   `gl`    
| PerceptronModel (POS UD) | `pos_ud_gsd`            | 2.5.5 |   `id`    

Languages: Armenian, Basque, Breton, Catalan, Danish, Galician, Hindi, Indonesian, Irish, Latin, Latvian, Marathi, Slovenian, Yoruba

Documentation and Notebooks

* New notebook for pretrained [StopWordsCleaner](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/stop-words/StopWordsCleaner.ipynb)
* New notebook to [Detect entities in German language](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_DE.ipynb)
* New notebook to [Detect entities in English language](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb)
* New notebook to [Detect entities in Spanish language](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_ES.ipynb)
* New notebook to [Detect entities in French language](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_FR.ipynb)
* New notebook to [Detect entities in Italian language](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_IT.ipynb)
* New notebook to [Detect entities in Norwegian language](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_NO.ipynb)
* New notebook to [Detect entities in Polish language](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_PL.ipynb)
* New notebook to [Detect entities in Portugese language](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_PT.ipynb)
* New notebook to [Detect entities in Russian language](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_RU.ipynb)
* Update documentation for release of Spark NLP 2.5.x
* Update the entire [spark-nlp-models](https://github.com/JohnSnowLabs/spark-nlp-models) repository with new pre-trained models and pipelines
* Update the entire [spark-nlp-workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks for Spark NLP 2.5.x

Installation

**Python**
```shell
#PyPI

pip install spark-nlp==2.5.5

#Conda

conda install -c johnsnowlabs spark-nlp==2.5.5
```


**Spark**

**spark-nlp** on Apache Spark 2.4.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.5

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.5
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.5.5

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.5.5
```

**spark-nlp** on Apache Spark 2.3.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.5.5

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.5.5
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.5.5

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.5.5
```

**Maven**

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.5.5</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.11</artifactId>
    <version>2.5.5</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>2.5.5</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>2.5.5</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-assembly-2.5.5.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-gpu-assembly-2.5.5.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-spark23-assembly-2.5.5.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-spark23-gpu-assembly-2.5.5.jar

### 2.5.4

#### John Snow Labs Spark-NLP 2.5.4: Supporting Apache Spark 2.3, 43 new models and 26 new languages, new RegexTokenizer, lots of new notebooks, and more

Overview

We are excited to release Spark NLP 2.5.4 with the full support of Apache Spark 2.3.x, adding 43 new pre-trained models for stop words cleaning, supporting 26 new languages, a new RegexTokenizer annotator and more!

As always, we would like to thank our community for their feedback, questions, and feature requests.

New Features

* Add support for Apache Spark 2.3.x including new Maven artifacts and full support of all pre-trained models/pipelines
* Add 43 new pre-trained models in 43 languages to StopWordsCleaner annotator
* Introduce a new RegexTokenizer to split text by regex pattern

Enhancements

* Retrained 6 new BioBERT and ClinicalBERT models 
* Add a new param `spark23` to `start()` function to start the session for Apache Spark 2.3.x

Bugfixes

* Add missing library for SentencePiece used by AlbertEmbeddings and XlnetEmbeddings on Windows
* Fix ModuleNotFoundError in LanguageDetectorDL pipelines in Python

Models

* We have added 43 new pre-trained models in 43 languages for StopWordsCleaner. Some selected models:

#### Afrikaans - Models

| Model                        | Name               | Build            | Lang |  Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| StopWordsCleaner  | `stopwords_af`            | 2.5.4 |   `af`   |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stopwords_af_af_2.5.4_2.4_1594742440083.zip) |

#### Arabic - Models

| Model                        | Name               | Build            | Lang |  Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| StopWordsCleaner  | `stopwords_ar`            | 2.5.4 |   `ar`   |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stopwords_ar_ar_2.5.4_2.4_1594742440256.zip) |

#### Armenian - Models

| Model                        | Name               | Build            | Lang |  Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| StopWordsCleaner  | `stopwords_hy`            | 2.5.4 |   `hy`   |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stopwords_hy_hy_2.5.4_2.4_1594742439626.zip) |

#### Basque - Models

| Model                        | Name               | Build            | Lang |  Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| StopWordsCleaner  | `stopwords_eu`            | 2.5.4 |   `eu`   |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stopwords_eu_eu_2.5.4_2.4_1594742441951.zip) |

#### Bengali - Models

| Model                        | Name               | Build            | Lang |  Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| StopWordsCleaner  | `stopwords_bn`            | 2.5.4 |   `bn`   |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stopwords_bn_bn_2.5.4_2.4_1594742440339.zip) |

#### Breton - Models

| Model                        | Name               | Build            | Lang |  Offline|
|:-----------------------------|:-------------------|:-----------------|:------|:------------|
| StopWordsCleaner  | `stopwords_br`            | 2.5.4 |   `br`   |[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stopwords_br_br_2.5.4_2.4_1594742440778.zip) |

Documentation and Notebooks

* New notebook for [Language detection and identification](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/language-detection/Language_Detection_and_Indentification.ipynb)
* New notebook for [Classify text according to TREC classes](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_EN_TREC.ipynb)
* New notebook for [Detect Spam messages](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_EN_SPAM.ipynb)
* New notebook for [Detect fake news](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_EN_FAKENEWS.ipynb)
* New notebook for [Find sentiment in text](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN.ipynb)
* New notebook for [Detect bullying in tweets](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN_CYBERBULLYING.ipynb)
* New notebook for [Detect Emotions in text](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN_EMOTION.ipynb)
* New notebook for [Detect Sarcasm in text](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN_SARCASM.ipynb)
* Update the entire [spark-nlp-models](https://github.com/JohnSnowLabs/spark-nlp-models) repository with new pre-trained models and pipelines
* Update the entire [spark-nlp-workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks for Spark NLP 2.5.x
* Update documentation for release of Spark NLP 2.5.x

Installation

**Python**
```shell
#PyPI

pip install spark-nlp==2.5.4

#Conda

conda install -c johnsnowlabs spark-nlp==2.5.4
```


**Spark**

**spark-nlp** on Apache Spark 2.4.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.4

pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.4
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.5.4

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.5.4
```

**spark-nlp** on Apache Spark 2.3.x:

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.5.4

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.11:2.5.4
```

**GPU**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.5.4

pyspark --packages com.johnsnowlabs.nlp:spark-nlp-spark23-gpu_2.11:2.5.4
```

**Maven**

**spark-nlp** on Apache Spark 2.4.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.5.4</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu_2.11</artifactId>
    <version>2.5.4</version>
</dependency>
```

**spark-nlp** on Apache Spark 2.3.x:

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-spark23_2.11</artifactId>
    <version>2.5.4</version>
</dependency>
```

**spark-nlp-gpu:**

```xml
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp-gpu-spark23_2.11</artifactId>
    <version>2.5.4</version>
</dependency>
```

**FAT JARs**

* CPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-assembly-2.5.4.jar

* GPU on Apache Spark 2.4.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-gpu-assembly-2.5.4.jar

* CPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-spark23-assembly-2.5.4.jar

* GPU on Apache Spark 2.3.x: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-spark23-gpu-assembly-2.5.4.jar

### 2.5.3

<div class="h3-box" markdown="1">

#### John Snow Labs Spark-NLP 2.5.3: Detect Fake news, emotions, spams, and more classification models, enhancements, and bug fixes

Overview

We are very happy to release Spark NLP 2.5.3 with 5 new pre-trained ClassifierDL models for multi-class text classification. There are also bug-fixes and other enhancements introduced in this release which were reported and requested by Spark NLP users.

As always, we thank our community for their feedback, questions, and feature requests.

New Features

* TextMatcher now can construct the chunks from tokens instead of the original documents via buildFromTokens param
* CoNLLGenerator now is accessible in Python

Bugfixes

* Fix a bug in ContextSpellChecker resulting in IllegalArgumentException

Enhancements

* Improve RocksDB connection to support different storage capabilities
* Improve parameters naming convention in ContextSpellChecker
* Add NerConverter to documentation
* Fix multi-language tabs in documentation

Models

We have added 5 new pre-trained ClassifierDL models for multi-class text classification.

{:.table-model-big}
| Model    | Name                      | Build            | Lang | Description | Offline
|:--------------|:--------------------------|:-----------------|:-----|:----------|:------|
| ClassifierDLModel    | `classifierdl_use_spam`        | 2.5.3 |      `en` |  Detect if a message is spam or not    | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_spam_en_2.5.3_2.4_1593783318934.zip) |
| ClassifierDLModel    | `classifierdl_use_fakenews`        | 2.5.3 |      `en` | Classify if a news is fake or real        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_fakenews_en_2.5.3_2.4_1593783319296.zip) |
| ClassifierDLModel    | `classifierdl_use_emotion`        | 2.5.3 |      `en`  | Detect Emotions in TweetsDetect Emotions in Tweets       | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_emotion_en_2.5.3_2.4_1593783319297.zip) |
| ClassifierDLModel    | `classifierdl_use_cyberbullying`        | 2.5.3 |      `en`  | Classify if a tweet is bullying      | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_cyberbullying_en_2.5.3_2.4_1593783319298.zip) |
| ClassifierDLModel    | `classifierdl_use_sarcasm`        | 2.5.3 |      `en` | Identify sarcastic tweets        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_sarcasm_en_2.5.3_2.4_1593783319298.zip) |

Documentation

* Update documentation for release of Spark NLP 2.5.x
* Update the entire [spark-nlp-workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks for Spark NLP 2.5.x
* Update the entire [spark-nlp-models](https://github.com/JohnSnowLabs/spark-nlp-models) repository with new pre-trained models and pipelines

Installation

**Python**
```shell
#PyPI

pip install spark-nlp==2.5.3

#Conda

conda install -c johnsnowlabs spark-nlp==2.5.3
```

**Spark**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.3
```

**PySpark**
```shell
pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.3
```

**Maven**
```shell
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.5.3</version>
</dependency>
```

**FAT JARs**

* CPU: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-assembly-2.5.3.jar

* GPU: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-gpu-assembly-2.5.3.jar

</div><div class="h3-box" markdown="1">

### 2.5.2

#### John Snow Labs Spark-NLP 2.5.2: New Language Detection annotator, enhancements, and bug fixes

Overview

We are very happy to release Spark NLP 2.5.2 with a new state-of-the-art LanguageDetectorDL annotator to detect and identify up to 20 languages. There are also bug-fixes and other enhancements introduced in this release which were reported and requested by Spark NLP users.

As always, we thank our community for their feedback, questions, and feature requests.

New Features

* Introducing a new LanguageDetectorDL state-of-the-art annotator to detect and identify languages in documents and sentences
* Add a new param entityValue to TextMatcher to add custom value inside metadata. Useful in post-processing when there are multiple TextMatcher annotators with multiple dictionaries [https://github.com/JohnSnowLabs/spark-nlp/issues/920](https://github.com/JohnSnowLabs/spark-nlp/issues/920)

Bugfixes

* Add missing TensorFlow graphs to train ContextSpellChecker annotator [https://github.com/JohnSnowLabs/spark-nlp/issues/912](https://github.com/JohnSnowLabs/spark-nlp/issues/912)
* Fix misspelled param in classThreshold param in  ContextSpellChecker annotator [https://github.com/JohnSnowLabs/spark-nlp/issues/911](https://github.com/JohnSnowLabs/spark-nlp/issues/911)
* Fix a bug where setGraphFolder in NerDLApproach annotator couldn't find a graph on Databricks (DBFS) [https://github.com/JohnSnowLabs/spark-nlp/issues/739](https://github.com/JohnSnowLabs/spark-nlp/issues/739)
* Fix a bug in NerDLApproach when includeConfidence was set to true [https://github.com/JohnSnowLabs/spark-nlp/issues/917](https://github.com/JohnSnowLabs/spark-nlp/issues/917)
* Fix a bug in BertEmbeddings [https://github.com/JohnSnowLabs/spark-nlp/issues/906](https://github.com/JohnSnowLabs/spark-nlp/issues/906) [https://github.com/JohnSnowLabs/spark-nlp/issues/918](https://github.com/JohnSnowLabs/spark-nlp/issues/918)

Enhancements

* Improve TF backend in ContextSpellChecker annotator

Pipelines and Models

We have added 4 new LanguageDetectorDL models and pipelines to detect and identify up to 20 languages:

* The model with 7 languages: Czech, German, English, Spanish, French, Italy, and Slovak
* The model with 20 languages: Bulgarian, Czech, German, Greek, English, Spanish, Finnish, French, Croatian, Hungarian, Italy, Norwegian, Polish, Portuguese, Romanian, Russian, Slovak, Swedish, Turkish, and Ukrainian

{:.table-model-big}
| Model    | Name                      | Build            | Lang | Offline
|:--------------|:--------------------------|:-----------------|:------------|:------|
| LanguageDetectorDL    | `ld_wiki_7`        | 2.5.2 |      `xx`         | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ld_wiki_7_xx_2.5.0_2.4_1591875673486.zip) |
| LanguageDetectorDL    | `ld_wiki_20`        | 2.5.2 |      `xx`         | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ld_wiki_20_xx_2.5.0_2.4_1591875680011.zip) |

{:.table-model-big}
| Pipeline    | Name                      | Build            | Lang | Offline
|:--------------|:--------------------------|:-----------------|:------------|:------|
| LanguageDetectorDL    | `detect_language_7`        | 2.5.2 |      `xx`         | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/detect_language_7_xx_2.5.0_2.4_1591875676774.zip) |
| LanguageDetectorDL    | `detect_language_20`        | 2.5.2 |      `xx`         | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/detect_language_20_xx_2.5.0_2.4_1591875683182.zip) |

Documentation

* Update documentation for release of Spark NLP 2.5.x
* Update the entire [spark-nlp-workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks for Spark NLP 2.5.x
* Update the entire [spark-nlp-models](https://github.com/JohnSnowLabs/spark-nlp-models) repository with new pre-trained models and pipelines

Installation

**Python**
```shell
#PyPI

pip install spark-nlp==2.5.2

#Conda

conda install -c johnsnowlabs spark-nlp==2.5.2
```

**Spark**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.2
```

**PySpark**
```shell
pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.2
```

**Maven**
```shell
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.5.2</version>
</dependency>
```

**FAT JARs**

* CPU: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-assembly-2.5.2.jar

* GPU: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-gpu-assembly-2.5.2.jar

</div><div class="h3-box" markdown="1">

### 2.5.1

#### John Snow Labs Spark-NLP 2.5.1: Adding support for 6 new BioBERT and ClinicalBERT models

Overview

We are very excited to extend Spark NLP support to 6 new BERT models for medical and clinical documents. We have also updated our documentation for 2.5.x releases, notebooks in our workshop, and made some enhancements in this release.

As always, we thank our community for their feedback and questions in our Slack channel.

New Features

* Add Python support for PubTator reader to convert automatic annotations of the biomedical datasets into DataFrame 
* Add 6 new pre-trained BERT models from BioBERT and ClinicalBERT

Models

We have added 6 new BERT models for medical and clinical purposes. The 4 BERT pre-trained models are from BioBERT and the other 2 are coming from ClinicalBERT models:

{:.table-model-big}
| Model    | Name                      | Build            | Lang | Offline
|:--------------|:--------------------------|:-----------------|:------------|:------|
| BertEmbeddings                    | `biobert_pubmed_base_cased`        | 2.5.0 |      `en`         | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/biobert_pubmed_base_cased_en_2.5.0_2.4_1590487367971.zip) |
| BertEmbeddings                    | `biobert_pubmed_large_cased`        | 2.5.0 |      `en`        | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/biobert_pubmed_large_cased_en_2.5.0_2.4_1590487739645.zip) |
| BertEmbeddings                    | `biobert_pmc_base_cased`        | 2.5.0 |      `en`            | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/biobert_pmc_base_cased_en_2.5.0_2.4_1590489029151.zip) |
| BertEmbeddings                    | `biobert_pubmed_pmc_base_cased`        | 2.5.0 |      `en`     | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/biobert_pubmed_pmc_base_cased_en_2.5.0_2.4_1590489367180.zip) |
| BertEmbeddings                    | `biobert_clinical_base_cased`        | 2.5.0 |      `en`       | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/biobert_clinical_base_cased_en_2.5.0_2.4_1590489819943.zip) |
| BertEmbeddings                    | `biobert_discharge_base_cased`        | 2.5.0 |      `en`      | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/biobert_discharge_base_cased_en_2.5.0_2.4_1590490193605.zip) |

Enhancements

* Add unit tests for XlnetEmbeddings
* Add unit tests for AlbertEmbeddings
* Add unit tests for ContextSpellChecker

Documentation

* Update documentation for release of Spark NLP 2.5.x
* Update the entire [spark-nlp-workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks for Spark NLP 2.5.x
* Update the entire [spark-nlp-models](https://github.com/JohnSnowLabs/spark-nlp-models) repository with new pre-trained models and pipelines

Installation

**Python**
```shell
#PyPI

pip install spark-nlp==2.5.1

#Conda

conda install -c johnsnowlabs spark-nlp==2.5.1
```

**Spark**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.1
```

**PySpark**
```shell
pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.1
```

**Maven**
```shell
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.5.1</version>
</dependency>
```

**FAT JARs**

* CPU: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-assembly-2.5.1.jar

* GPU: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-gpu-assembly-2.5.1.jar

</div><div class="h3-box" markdown="1">

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

Bugfixes

* Fix position of tokens in Normalizer
* Fix Lemmatizer exception on a bad input
* Fix annotator logs failing on object storage file systems like DBFS

Models and Pipelines

Spark NLP `2.5.0` comes with 87 new pretrained models and pipelines in 14 new languages available for all Windows, Linux, and macOS users. We added new languages such as Dutch, Norwegian. Polish, Portuguese, Bulgarian, Czech, Greek, Finnish, Hungarian, Romanian, Slovak, Swedish, Turkish, and Ukrainian. 

The complete list of 160+ models & pipelines in 22+ languages is [available here](https://github.com/JohnSnowLabs/spark-nlp-models/).

**Featured Pretrained Pipelines**

**Dutch - Pipelines**

{:.table-model-big}
| Pipeline                 | Name                   | Build  | lang | Description | Offline   |
|:-------------------------|:-----------------------|:-------|:-------|:----------|:----------|
| Explain Document Small    | `explain_document_sm`  | 2.5.0 |   `nl` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_sm_nl_2.5.0_2.4_1588546621618.zip)  |
| Explain Document Medium   | `explain_document_md`  | 2.5.0 |   `nl` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_nl_2.5.0_2.4_1588546605329.zip)  |
| Explain Document Large    | `explain_document_lg`  | 2.5.0 |   `nl` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_nl_2.5.0_2.4_1588612556770.zip)  |
| Entity Recognizer Small   | `entity_recognizer_sm`  | 2.5.0 |   `nl` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_sm_nl_2.5.0_2.4_1588546655907.zip)  |
| Entity Recognizer Medium  | `entity_recognizer_md`  | 2.5.0 |   `nl` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_md_nl_2.5.0_2.4_1588546645304.zip)  |
| Entity Recognizer Large   | `entity_recognizer_lg`  | 2.5.0 |   `nl` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_nl_2.5.0_2.4_1588612569958.zip)  |  

**Norwegian - Pipelines**

{:.table-model-big}
| Pipeline                 | Name                   | Build  | lang | Description | Offline   |
|:-------------------------|:-----------------------|:-------|:-------|:----------|:----------|
| Explain Document Small    | `explain_document_sm`  | 2.5.0 |   `no` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_sm_no_2.5.0_2.4_1588784132955.zip)  |
| Explain Document Medium   | `explain_document_md`  | 2.5.0 |   `no` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_no_2.5.0_2.4_1588783879809.zip)  |
| Explain Document Large    | `explain_document_lg`  | 2.5.0 |   `no` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_no_2.5.0_2.4_1588782610672.zip)  |
| Entity Recognizer Small   | `entity_recognizer_sm`  | 2.5.0 |   `no` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_sm_no_2.5.0_2.4_1588794567766.zip)  |
| Entity Recognizer Medium  | `entity_recognizer_md`  | 2.5.0 |   `no` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_md_no_2.5.0_2.4_1588794357614.zip)  |
| Entity Recognizer Large   | `entity_recognizer_lg`  | 2.5.0 |   `no` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_no_2.5.0_2.4_1588793261642.zip)  |  

**Polish - Pipelines**

{:.table-model-big}
| Pipeline                 | Name                   | Build  | lang | Description | Offline   |
|:-------------------------|:-----------------------|:-------|:-------|:----------|:----------|
| Explain Document Small    | `explain_document_sm`  | 2.5.0 |   `pl` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_sm_pl_2.5.0_2.4_1588531081173.zip)  |
| Explain Document Medium   | `explain_document_md`  | 2.5.0 |   `pl` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_pl_2.5.0_2.4_1588530841737.zip)  |
| Explain Document Large    | `explain_document_lg`  | 2.5.0 |   `pl` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_pl_2.5.0_2.4_1588529695577.zip)  |
| Entity Recognizer Small   | `entity_recognizer_sm`  | 2.5.0 |   `pl` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_sm_pl_2.5.0_2.4_1588532616080.zip)  |
| Entity Recognizer Medium  | `entity_recognizer_md`  | 2.5.0 |   `pl` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_md_pl_2.5.0_2.4_1588532376753.zip)  |
| Entity Recognizer Large   | `entity_recognizer_lg`  | 2.5.0 |   `pl` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_pl_2.5.0_2.4_1588531171903.zip)  |  

**Portuguese - Pipelines**

{:.table-model-big}
| Pipeline                 | Name                   | Build  | lang | Description | Offline   |
|:-------------------------|:-----------------------|:-------|:-------|:----------|:----------|
| Explain Document Small    | `explain_document_sm`  | 2.5.0 |   `pt` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_sm_pt_2.5.0_2.4_1588501423743.zip)  |
| Explain Document Medium   | `explain_document_md`  | 2.5.0 |   `pt` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_pt_2.5.0_2.4_1588501189804.zip)  |
| Explain Document Large    | `explain_document_lg`  | 2.5.0 |   `pt` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_pt_2.5.0_2.4_1588500056427.zip)  |
| Entity Recognizer Small   | `entity_recognizer_sm`  | 2.5.0 |   `pt` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_sm_pt_2.5.0_2.4_1588502815900.zip)  |
| Entity Recognizer Medium  | `entity_recognizer_md`  | 2.5.0 |   `pt` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_md_pt_2.5.0_2.4_1588502606198.zip)  |
| Entity Recognizer Large   | `entity_recognizer_lg`  | 2.5.0 |   `pt` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_pt_2.5.0_2.4_1588501526324.zip)  |  

Documentation

* Update documentation for release of Spark NLP 2.5.0
* Update the entire [spark-nlp-workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks for Spark NLP 2.5.0
* Update the entire [spark-nlp-models](https://github.com/JohnSnowLabs/spark-nlp-models) repository with new pre-trained models and pipelines

Installation

**Python**
```shell
#PyPI

pip install spark-nlp==2.5.0

#Conda

conda install -c johnsnowlabs spark-nlp==2.5.0
```

**Spark**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.0
```

**PySpark**
```shell
pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.0
```

**Maven**
```shell
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.5.0</version>
</dependency>
```

**FAT JARs**

* CPU: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-assembly-2.5.0.jar

* GPU: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-gpu-assembly-2.5.0.jar

</div><div class="h3-box" markdown="1">

### 2.4.5

#### John Snow Labs Spark-NLP 2.4.5: Supporting more Databricks runtimes and YARN in cluster mode

Overview

We are very excited to extend Spark NLP support to 6 new Databricks runtimes and add support to Cloudera and EMR YARN cluster-mode.
As always, we thank our community for their feedback and questions in our Slack channel.

New Features

* Extend Spark NLP support for Databricks runtimes:
  * 6.2
  * 6.2 ML
  * 6.3
  * 6.3 ML
  * 6.4
  * 6.4 ML
  * 6.5
  * 6.5 ML
* Add support for cluster-mode in Cloudera and EMR YARN clusters
* New splitPattern param in Tokenizer to split tokens by regex rules

Bugfixes

* Fix ClassifierDLModel save and load in Python
* Fix ClassifierDL TensorFlow session reuse
* Fix Normalizer positions of new tokens

Documentation

* Update documentation for release of Spark NLP 2.4.x
* Update the entire [spark-nlp-workshop](https://github.com/JohnSnowLabs/spark-nlp-models) notebooks for Spark NLP 2.4.x
* Update the entire [spark-nlp-models](https://github.com/JohnSnowLabs/spark-nlp-workshop) repository with new pre-trained models and pipelines

Installation

**Python**
```shell
#PyPI

pip install spark-nlp==2.4.5

#Conda

conda install -c johnsnowlabs spark-nlp==2.4.5
```

**Spark**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5
```

**PySpark**
```shell
pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5
```

**Maven**
```shell
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.4.5</version>
</dependency>
```

**FAT JARs**

* CPU: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-assembly-2.4.5.jar

* GPU: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-gpu-assembly-2.4.5.jar

</div><div class="h3-box" markdown="1">

### 2.4.4

#### John Snow Labs Spark-NLP 2.4.4: The very first native multi-class text classifier and pre-trained models and pipelines in Russian

Overview

* We are very excited to release the very first multi-class text classifier in Spark NLP v2.4.4! We have built a generic ClassifierDL annotator that uses the state-of-the-art Universal Sentence Encoder as an input for text classifications. The ClassifierDL annotator uses a deep learning model (DNNs) we have built inside TensorFlow and supports up to 50 classes.
* We are also happy to announce the support of yet another language: Russian! We have trained and prepared 5 pre-trained models and 6 pre-trained pipelines in Russian.

**NOTE**: ClassifierDL is an experimental feature in 2.4.4 before it becomes stable in 2.4.5 release. We have worked hard to aim for simplicity and we are looking forward to your feedback as always.
We will add more examples by the upcoming days: 

Examples: [Python](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/jupyter/training/english/classification) and [Scala](https://johnsnowlabs.github.io/spark-nlp-workshop/databricks/index.html#training/3-%20Train%20Multi-Class%20Text%20Classification%20on%20News%20Articles.html)

New Features

* Introducing a generic multi-class text classifier: ClassifierDL. The ClassifierDL annotator uses a deep learning model (DNNs) we have built inside TensorFlow and supports up to 50 classes.
* 5 new pretrained Russian models (Lemma, POS, 3x NER)
* 6 new pretrained Russian pipelines

**Models:**

{:.table-model-big}
| Model                                  |   name     |   language     |
|--------------------------|--------------|----------|
| LemmatizerModel (Lemmatizer) | `lemma `|`ru`|
| PerceptronModel (POS UD) | `pos_ud_gsd `|`ru`|
| NerDLModel | `wikiner_6B_100 `|`ru`|
| NerDLModel | `wikiner_6B_300 `|`ru`|
| NerDLModel | `wikiner_840B_300 `|`ru`|

**Pipelines:**

{:.table-model-big}
| Pipeline                                  |   name     |   language     |
|--------------------------|--------------|----------|
| Explain Document (Small) | `explain_document_sm`|`ru`|
| Explain Document (Medium) | `explain_document_md`|`ru`|
| Explain Document (Large) | `explain_document_lg`|`ru`|
| Entity Recognizer (Small) | `entity_recognizer_sm`|`ru`|
| Entity Recognizer (Medium) | `entity_recognizer_md`|`ru`|
| Entity Recognizer (Large) | `entity_recognizer_lg`|`ru`|

**Evaluation:**


wikiner_6B_100 with `conlleval.pl`

{:.table-model-big.w7}
|Accuracy         |Precision         |Recall |F1-Score   |
|-----------------|------------------|-------|-----------|
|97.76%|88.85%|  88.55%| 88.70

wikiner_6B_300 with `conlleval.pl`

{:.table-model-big.w7}
|Accuracy         |Precision         |Recall |F1-Score   |
|-----------------|------------------|-------|-----------|
|97.78%| 89.09% | 88.51%|  88.80

wikiner_840B_300 with `conlleval.pl`

{:.table-model-big.w7}
|Accuracy         |Precision         |Recall |F1-Score   |
|-----------------|------------------|-------|-----------|
|97.85%|  89.85%|  89.11%|  89.48

**Example:**

```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = PretrainedPipeline("explain_document_sm", lang="ru")

val testData = spark.createDataFrame(Seq(
(1, "Пик распространения коронавируса и вызываемой им болезни Covid-19 в Китае прошел, заявил в четверг агентству Синьхуа официальный представитель Госкомитета по гигиене и здравоохранению КНР Ми Фэн.")
)).toDF("id", "text")

val annotation = pipeline.transform(testData)

annotation.show()
```

Enhancements

* Add param to NerConverter to override modified tokens instead of original tokens
* UniversalSentenceEncoder and SentenceEmbeddings are now accepting storageRef

Bugfixes

* Fix TokenAssembler
* Fix NerConverter exception when NerDL is trained with different tagging style than IOB/IOB2
* Normalizer now recomputes the index of tokens when it removes characters from a text

Documentation

* Update documentation for release of Spark NLP 2.4.x
* Update the entire [spark-nlp-workshop](https://github.com/JohnSnowLabs/spark-nlp-models) notebooks for Spark NLP 2.4.x
* Update the entire [spark-nlp-models](https://github.com/JohnSnowLabs/spark-nlp-workshop) repository with new pre-trained models and pipelines

Installation

**Python**
```shell
#PyPI

pip install spark-nlp==2.4.4

#Conda

conda install -c johnsnowlabs spark-nlp==2.4.4
```

**Spark**
```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.4
```

**PySpark**
```shell
pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.4
```

**Maven**
```shell
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.4.4</version>
</dependency>
```

**FAT JARs**

* CPU: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-assembly-2.4.4.jar

* GPU: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-gpu-assembly-2.4.4.jar

</div><div class="h3-box" markdown="1">

### 2.4.3

#### John Snow Labs Spark-NLP 2.4.3: Minor bug fix in Python

Overview

This minor release fixes a bug on our Python side that was introduced in 2.4.2 release. As always, we thank our community for their feedback and questions in our Slack channel.

**NOTE**: We highly recommend our Python users to update to 2.4.3 release.

Bugfixes

* Fix Python imports which resulted in AttributeError: module 'sparknlp' has no attribute

Documentation

* Update [documentation](https://nlp.johnsnowlabs.com/) for release of Spark NLP 2.4.x
* Update the entire [spark-nlp-workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks for Spark NLP 2.4.x
* Update the entire [spark-nlp-models](https://github.com/JohnSnowLabs/spark-nlp-models) repository with new pre-trained models and pipelines

Installation

* PyPI

```
pip install spark-nlp==2.4.3
```

* Conda

```
conda install -c johnsnowlabs spark-nlp==2.4.3
```

* spark-shell

```
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.3
```

* PySpark

```
pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.3
```

* Maven

```
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.4.3</version>
</dependency>
```

* FAT JARs

**CPU**: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-assembly-2.4.3.jar
**GPU**: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-gpu-assembly-2.4.3.jar

</div><div class="h3-box" markdown="1">

### 2.4.2

#### John Snow Labs Spark-NLP 2.4.2: Minor bug fixes and improvements

Overview

This minor release fixes a few bugs in some of our annotators reported by our community. 
As always, we thank our community for their feedback and questions in our Slack channel.

Bugfixes

* Fix UniversalSentenceEncoder.pretrained() that failed in Python
* Fix ElmoEmbeddings.pretrained() that failed in Python
* Fix ElmoEmbeddings poolingLayer param to be a string as expected
* Fix ChunkEmbeddings to preserve chunk's index
* Fix NGramGenerator and missing chunk metadata

New Features

* Add GPU support param in Spark NLP start function: sparknlp.start(gpu=true)
* Improve create_model.py to create custom TF graph for NerDLApproach

Documentation

* Update [documentation](https://nlp.johnsnowlabs.com/) for release of Spark NLP 2.4.x
* Update the entire [spark-nlp-workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop) notebooks for Spark NLP 2.4.x
* Update the entire [spark-nlp-models](https://github.com/JohnSnowLabs/spark-nlp-models) repository with new pre-trained models and pipelines

Installation

* PyPI

```
pip install spark-nlp==2.4.2
```

* Conda

```
conda install -c johnsnowlabs spark-nlp==2.4.2
```

* spark-shell

```
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.2
```

* PySpark

```
pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.2
```

* Maven

```
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.4.2</version>
</dependency>
```

* FAT JARs

**CPU**: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-assembly-2.4.2.jar
**GPU**: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-gpu-assembly-2.4.2.jar

</div><div class="h3-box" markdown="1">

### 2.4.1

#### John Snow Labs Spark-NLP 2.4.1: Bug fixes and the very first Spanish models & pipelines

Overview

This minor release fixes a few bugs in some of the annotators reported by our community.
As always, we thank our community for their feedback on our Slack channel.

Models & Pipelines

* 5 new pretrained Spanish models (Lemma, POS, 3x NER)
* 6 new pretrained Spanish pipelines

**Models:**

{:.table-model-big}
| Model                                  |   name     |   language     |
|--------------------------|--------------|----------|
| LemmatizerModel (Lemmatizer) | `lemma `|`es`|
| PerceptronModel (POS UD) | `pos_ud_gsd `|`es`|
| NerDLModel | `wikiner_6B_100 `|`es`|
| NerDLModel | `wikiner_6B_300 `|`es`|
| NerDLModel | `wikiner_840B_300 `|`es`|

**Pipelines:**

{:.table-model-big}
| Pipeline                                  |   name     |   language     |
|--------------------------|--------------|----------|
| Explain Document (Small) | `explain_document_sm`|`es`|
| Explain Document (Medium) | `explain_document_md`|`es`|
| Explain Document (Large) | `explain_document_lg`|`es`|
| Entity Recognizer (Small) | `entity_recognizer_sm`|`es`|
| Entity Recognizer (Medium) | `entity_recognizer_md`|`es`|
| Entity Recognizer (Large) | `entity_recognizer_lg`|`es`|

**Evaluation:**

wikiner_6B_100 with `conlleval.pl`

{:.table-model-big.w7}
|Accuracy         |Precision         |Recall |F1-Score   |
|-----------------|------------------|-------|-----------|
| 98.35% | 88.97% | 88.64% | 88.80 |

wikiner_6B_300 with `conlleval.pl`

{:.table-model-big.w7}
|Accuracy         |Precision         |Recall |F1-Score   |
|-----------------|------------------|-------|-----------|
| 98.38% | 89.42% | 89.03% | 89.22 |

wikiner_840B_300 with `conlleval.pl`

{:.table-model-big.w7}
|Accuracy         |Precision         |Recall |F1-Score   |
|-----------------|------------------|-------|-----------|
| 98.46% | 89.74% | 89.43% | 89.58 |

</div><div class="h3-box" markdown="1">

#### Example

```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = PretrainedPipeline("explain_document_sm", lang="es")

val testData = spark.createDataFrame(Seq(
(1, "Ésta se convertiría en una amistad de por vida, y Peleo, conociendo la sabiduría de Quirón , más adelante le confiaría la educación de su hijo Aquiles."),
(2, "Durante algo más de 200 años el territorio de la actual Bolivia constituyó la Real Audiencia de Charcas, uno de los centros más prósperos y densamente poblados de los virreinatos españoles.")
)).toDF("id", "text")

val annotation = pipeline.transform(testData)

annotation.show()
```

More info on [pre-trained models and pipelines](https://github.com/JohnSnowLabs/spark-nlp-models)

Bugfixes

* Improve ChunkEmbeddings annotator and fix the empty chunk result
* Fix UniversalSentenceEncoder crashing on empty Tensor
* Fix NorvigSweetingModel missing sentenceId that results in NGramsGenerator crashing
* Fix missing storageRef in embeddings' column for ElmoEmbeddings annotator

Documentation

* Update documentation for release of Spark NLP 2.4.x
* Add new features such as ElmoEmbeddings and UniversalSentenceEncoder
* Add multiple programming languages for demos and examples
* Update the entire [spark-nlp-models](https://github.com/JohnSnowLabs/spark-nlp-models) repository with new pre-trained models and pipelines

</div>

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

Models and Pipelines

Spark NLP `2.4.0` comes with new models including Universal Sentence Encoder, BERT, and Elmo models from TF Hub. In addition, our multilingual pipelines are now available for Windows as same as Linux and macOS users.

{:.table-model-big}
| Models              |   Name        |
|------------------------|---------------|
|UniversalSentenceEncoder|`tf_use`
|UniversalSentenceEncoder|`tf_use_lg`
|BertEmbeddings|`bert_large_cased`
|BertEmbeddings|`bert_large_uncased`
|BertEmbeddings|`bert_base_cased`
|BertEmbeddings|`bert_base_uncased`
|BertEmbeddings|`bert_multi_cased`
|ElmoEmbeddings|`elmo`
|NerDLModel|`onto_100`
|NerDLModel|`onto_300`

{:.table-model-big}
| Pipelines               | Name                   | Language
| ----------------------- | ---------------------  | ---------|
| Explain Document Large  | `explain_document_lg`  | fr
| Explain Document Medium | `explain_document_md`  | fr
| Entity Recognizer Large | `entity_recognizer_lg` | fr
| Entity Recognizer Medium| `entity_recognizer_md` | fr
| Explain Document Large  | `explain_document_lg`  | de
| Explain Document Medium | `explain_document_md`  | de
| Entity Recognizer Large | `entity_recognizer_lg` | de
| Entity Recognizer Medium| `entity_recognizer_md` | de
| Explain Document Large  | `explain_document_lg`  | it
| Explain Document Medium | `explain_document_md`  | it
| Entity Recognizer Large | `entity_recognizer_lg` | it
| Entity Recognizer Medium| `entity_recognizer_md` | it

Example:

```python
# Import Spark NLP
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline
import sparknlp

# Start Spark Session with Spark NLP
# If you already have a SparkSession (Zeppelin, Databricks, etc.) 
# you can skip this
spark = sparknlp.start()

# Download a pre-trained pipeline
pipeline = PretrainedPipeline('explain_document_md', lang='fr')

# Your testing dataset
text = """
Emmanuel Jean-Michel Frédéric Macron est le fils de Jean-Michel Macron, né en 1950, médecin, professeur de neurologie au CHU d'Amiens4 et responsable d'enseignement à la faculté de médecine de cette même ville5, et de Françoise Noguès, médecin conseil à la Sécurité sociale.
"""

# Annotate your testing dataset
result = pipeline.annotate(text)
 
# What's in the pipeline
list(result.keys())
# result:
# ['entities', 'lemma', 'document', 'pos', 'token', 'ner', 'embeddings', 'sentence']

# Check the results
result['entities']
# entities:
# ['Emmanuel Jean-Michel Frédéric Macron', 'Jean-Michel Macron', "CHU d'Amiens4", 'Françoise Noguès', 'Sécurité sociale']
```

Backward incompatibilities

Please note that in `2.4.0` we have added `storageRef` parameter to our `WordEmbeddogs`. This means every `WordEmbeddingsModel` will now have `storageRef` which is also bound to `NerDLModel` trained by that embeddings.
This assures users won't use a `NerDLModel` with a wrong `WordEmbeddingsModel`.

Example:

```scala
val embeddings = new WordEmbeddings()
      .setStoragePath("/tmp/glove.6B.100d.txt", ReadAs.TEXT)
      .setDimension(100)
      .setStorageRef("glove_100d") // Use or save this WordEmbeddings with storageRef
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
```

If you save the`WordEmbeddings` model the `storageRef` will be `glove_100d`. If you ever train any `NerDLApproach` the `glove_100d` will bind to that `NerDLModel`.

If you have already `WordEmbeddingsModels` saved from earlier versions, you either need to re-save them with `storageRed` or you can manually add this param in their `metadata/`. The same advice works for the `NerDLModel` from earlier versions.

Installation

**Python**

```shell
#PyPI

pip install spark-nlp==2.4.0

#Conda

conda install -c johnsnowlabs spark-nlp==2.4.0
```

**Spark**

```shell
spark-shell --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.0
```

**PySpark**

```shell
pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.0
```

**Maven**

```shell
<dependency>
    <groupId>com.johnsnowlabs.nlp</groupId>
    <artifactId>spark-nlp_2.11</artifactId>
    <version>2.4.0</version>
</dependency>
```

**FAT JARs**

* CPU: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-assembly-2.4.0.jar

* GPU: https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/spark-nlp-gpu-assembly-2.4.0.jar

Bugfixes

* Fixed splitChars in Tokenizer
* Fixed PretrainedPipeline in Python to allow accessing the inner PipelineModel in the instance
* Fixes in Chunk and SentenceEmbeddings to better deal with empty cleaned-up Annotations

Documentation and examples

* We have a new Developer section for those who are interested in contributing to Spark NLP
[Developer](https://nlp.johnsnowlabs.com/docs/en/developers)
* We have updated our workshop repository with more notebooks
[Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop)