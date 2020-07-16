---
layout: article
title: Spark NLP release notes
permalink: /docs/en/release_notes
key: docs-release-notes
modify_date: "2020-07-16"
---

### 2.5.3

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

| Model    | Name                      | Build            | Lang | Offline
|:--------------|:--------------------------|:-----------------|:------------|:------|
| LanguageDetectorDL    | `ld_wiki_7`        | 2.5.2 |      `xx`         | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ld_wiki_7_xx_2.5.0_2.4_1591875673486.zip) |
| LanguageDetectorDL    | `ld_wiki_20`        | 2.5.2 |      `xx`         | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ld_wiki_20_xx_2.5.0_2.4_1591875680011.zip) |

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

| Pipeline                 | Name                   | Build  | lang | Description | Offline   |
|:-------------------------|:-----------------------|:-------|:-------|:----------|:----------|
| Explain Document Small    | `explain_document_sm`  | 2.5.0 |   `nl` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_sm_nl_2.5.0_2.4_1588546621618.zip)  |
| Explain Document Medium   | `explain_document_md`  | 2.5.0 |   `nl` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_nl_2.5.0_2.4_1588546605329.zip)  |
| Explain Document Large    | `explain_document_lg`  | 2.5.0 |   `nl` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_nl_2.5.0_2.4_1588612556770.zip)  |
| Entity Recognizer Small   | `entity_recognizer_sm`  | 2.5.0 |   `nl` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_sm_nl_2.5.0_2.4_1588546655907.zip)  |
| Entity Recognizer Medium  | `entity_recognizer_md`  | 2.5.0 |   `nl` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_md_nl_2.5.0_2.4_1588546645304.zip)  |
| Entity Recognizer Large   | `entity_recognizer_lg`  | 2.5.0 |   `nl` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_nl_2.5.0_2.4_1588612569958.zip)  |  

**Norwegian - Pipelines**

| Pipeline                 | Name                   | Build  | lang | Description | Offline   |
|:-------------------------|:-----------------------|:-------|:-------|:----------|:----------|
| Explain Document Small    | `explain_document_sm`  | 2.5.0 |   `no` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_sm_no_2.5.0_2.4_1588784132955.zip)  |
| Explain Document Medium   | `explain_document_md`  | 2.5.0 |   `no` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_no_2.5.0_2.4_1588783879809.zip)  |
| Explain Document Large    | `explain_document_lg`  | 2.5.0 |   `no` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_no_2.5.0_2.4_1588782610672.zip)  |
| Entity Recognizer Small   | `entity_recognizer_sm`  | 2.5.0 |   `no` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_sm_no_2.5.0_2.4_1588794567766.zip)  |
| Entity Recognizer Medium  | `entity_recognizer_md`  | 2.5.0 |   `no` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_md_no_2.5.0_2.4_1588794357614.zip)  |
| Entity Recognizer Large   | `entity_recognizer_lg`  | 2.5.0 |   `no` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_no_2.5.0_2.4_1588793261642.zip)  |  

**Polish - Pipelines**

| Pipeline                 | Name                   | Build  | lang | Description | Offline   |
|:-------------------------|:-----------------------|:-------|:-------|:----------|:----------|
| Explain Document Small    | `explain_document_sm`  | 2.5.0 |   `pl` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_sm_pl_2.5.0_2.4_1588531081173.zip)  |
| Explain Document Medium   | `explain_document_md`  | 2.5.0 |   `pl` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_pl_2.5.0_2.4_1588530841737.zip)  |
| Explain Document Large    | `explain_document_lg`  | 2.5.0 |   `pl` |           | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_pl_2.5.0_2.4_1588529695577.zip)  |
| Entity Recognizer Small   | `entity_recognizer_sm`  | 2.5.0 |   `pl` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_sm_pl_2.5.0_2.4_1588532616080.zip)  |
| Entity Recognizer Medium  | `entity_recognizer_md`  | 2.5.0 |   `pl` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_md_pl_2.5.0_2.4_1588532376753.zip)  |
| Entity Recognizer Large   | `entity_recognizer_lg`  | 2.5.0 |   `pl` |          | [Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_pl_2.5.0_2.4_1588531171903.zip)  |  

**Portuguese - Pipelines**

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

| Model                                  |   name     |   language     |
|--------------------------|--------------|----------|
| LemmatizerModel (Lemmatizer) | `lemma `|`ru`|
| PerceptronModel (POS UD) | `pos_ud_gsd `|`ru`|
| NerDLModel | `wikiner_6B_100 `|`ru`|
| NerDLModel | `wikiner_6B_300 `|`ru`|
| NerDLModel | `wikiner_840B_300 `|`ru`|

**Pipelines:**

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
|Accuracy         |Precision         |Recall |F1-Score   |
|-----------------|------------------|-------|-----------|
|97.76%|88.85%|  88.55%| 88.70

wikiner_6B_300 with `conlleval.pl`
|Accuracy         |Precision         |Recall |F1-Score   |
|-----------------|------------------|-------|-----------|
|97.78%| 89.09% | 88.51%|  88.80

wikiner_840B_300 with `conlleval.pl`
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

### 2.4.1

#### John Snow Labs Spark-NLP 2.4.1: Bug fixes and the very first Spanish models & pipelines

Overview

This minor release fixes a few bugs in some of the annotators reported by our community.
As always, we thank our community for their feedback on our Slack channel.

Models & Pipelines

* 5 new pretrained Spanish models (Lemma, POS, 3x NER)
* 6 new pretrained Spanish pipelines

**Models:**

| Model                                  |   name     |   language     |
|--------------------------|--------------|----------|
| LemmatizerModel (Lemmatizer) | `lemma `|`es`|
| PerceptronModel (POS UD) | `pos_ud_gsd `|`es`|
| NerDLModel | `wikiner_6B_100 `|`es`|
| NerDLModel | `wikiner_6B_300 `|`es`|
| NerDLModel | `wikiner_840B_300 `|`es`|

**Pipelines:**

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
|Accuracy         |Precision         |Recall |F1-Score   |
|-----------------|------------------|-------|-----------|
| 98.35% | 88.97% | 88.64% | 88.80 |

wikiner_6B_300 with `conlleval.pl`
|Accuracy         |Precision         |Recall |F1-Score   |
|-----------------|------------------|-------|-----------|
| 98.38% | 89.42% | 89.03% | 89.22 |

wikiner_840B_300 with `conlleval.pl`
|Accuracy         |Precision         |Recall |F1-Score   |
|-----------------|------------------|-------|-----------|
| 98.46% | 89.74% | 89.43% | 89.58 |

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