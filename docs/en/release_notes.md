---
layout: docs
header: true
title: Spark NLP release notes
permalink: /docs/en/release_notes
key: docs-release-notes
modify_date: "2020-11-27"
---

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