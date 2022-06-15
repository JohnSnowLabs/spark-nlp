---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 3.0.0
permalink: /docs/en/spark_nlp_versions/release_notes_3_0_0
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

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

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_2_7_5">Version 2.7.5</a>
    </li>
    <li>
        <strong>Version 3.0.0</strong>
    </li>
    <li>
        <a href="release_notes_3_0_1">Version 3.0.1</a>
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
  <li class="active"><a href="release_notes_3_0_0">3.0.0</a></li>
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