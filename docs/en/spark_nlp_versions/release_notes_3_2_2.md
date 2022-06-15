---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 3.2.2
permalink: /docs/en/spark_nlp_versions/release_notes_3_2_2
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

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

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_2_1">Version 3.2.1</a>
    </li>
    <li>
        <strong>Version 3.2.2</strong>
    </li>
    <li>
        <a href="release_notes_3_2_3">Version 3.2.3</a>
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
  <li class="active"><a href="release_notes_3_2_2">3.2.2</a></li>
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