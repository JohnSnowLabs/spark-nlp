---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Developers Guideline
permalink: /docs/en/benchmark
key: docs-benchmark
modify_date: "2021-10-04"
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

## Cluster Speed Benchmarks

### NER (BiLSTM-CNN-Char Architecture) Benchmark Experiment

- **Dataset :** 1000 Clinical Texts from MTSamples Oncology Dataset, approx. 500 tokens per text.
- **Driver :** Standard_D4s_v3 - 16 GB Memory - 4 Cores
- **Enable Autoscaling :** False
- **Cluster Mode :** Standart
- **Worker :**
  - Standard_D4s_v3 - 16 GB Memory - 4 Cores
  - Standard_D4s_v2 - 28 GB Memory - 8 Cores
- **Versions :**
  - **Databricks Runtime Version :** 8.3(Scala 2.12, Spark 3.1.1)
  - **spark-nlp Version:** v3.2.3
  - **spark-nlp-jsl Version :** v3.2.3
  - **Spark Version :** v3.1.1
- **Spark NLP Pipeline :**

  ```
  nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,  
        embeddings_clinical,  
        clinical_ner,  
        ner_converter
        ])

  ```
  
**NOTES :**

+ **The first experiment with 5 different cluster configurations :** `ner_chunk`  as a column in Spark NLP Pipeline (`ner_converter`) output data frame, exploded (lazy evaluation) as `ner_chunk` and `ner_label`. Then results were written as **parquet** and **delta** formats.

+ **A second experiment with 2 different cluster configuration :** Spark NLP Pipeline output data frame (except `word_embeddings` column) was written as **parquet** and **delta** formats.

+ In the first experiment with the most basic driver node and worker **(1 worker x 4 cores)** configuration selection, it took **4.64 mins** and **4.53 mins** to write **4 partitioned data** as parquet and delta formats respectively.

+ With basic driver node and **8 workers (x8 cores)** configuration selection, it took **40 seconds** and **22 seconds** to write **1000 partitioned data** as parquet and delta formats respectively.

+ In the second experiment with basic driver node and **4 workers (x 4 cores)** configuration selection, it took **1.41 mins** as parquet and **1.42 mins** as delta format to write **16 partitioned (exploded results) data**.  **Without explode it took 1.08 mins as parquet and 1.12 mins as delta format to write the data frame.**

+ Since given computation durations are highly dependent on different parameters including driver node and worker node configurations as well as partitions, **results show that explode method increases duration  %10-30  on chosen configurations.**

</div>
<div class="h3-box" markdown="1">

#### NER Benchmark Tables

{:.table-model-big.db}
| driver\_name      | driver\_memory | driver\_cores | worker\_name      | worker\_memory | worker\_cores | input\_data\_rows | output\_data\_rows | action           | total\_worker\_number | total\_cores | partition | NER timing|NER+RE timing|
| ----------------- | -------------- | ------------- | ----------------- | -------------- | ------------- | ----------------- | ------------------ | ---------------- | --------------------- | ------------ | --------- | --------  |-----------  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v2 | 28 GB          | 8             | 1000              | 78000              | write\_parquet   | 8                     | 64           | 64        | 36 sec    | 1.14 mins   |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v2 | 28 GB          | 8             | 1000              | 78000              | write\_deltalake | 8                     | 64           | 64        | 19 sec    | 1.13 mins   |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v2 | 28 GB          | 8             | 1000              | 78000              | write\_parquet   | 8                     | 64           | 100       | 21 sec    | 50 sec      |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v2 | 28 GB          | 8             | 1000              | 78000              | write\_deltalake | 8                     | 64           | 100       | 41 sec    | 51 sec      |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v2 | 28 GB          | 8             | 1000              | 78000              | write\_parquet   | 8                     | 64           | 1000      | 40 sec    | 54 sec      |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v2 | 28 GB          | 8             | 1000              | 78000              | write\_deltalake | 8                     | 64           | 1000      | 22 sec    | 46 sec      |


{:.table-model-big.db}
| driver\_name      | driver\_memory | driver\_cores | worker\_name      | worker\_memory | worker\_cores | input\_data\_rows | output\_data\_rows | action           | total\_worker\_number | total\_cores | partition | duration  |NER+RE timing|
| ----------------- | -------------- | ------------- | ----------------- | -------------- | ------------- | ----------------- | ------------------ | ---------------- | --------------------- | ------------ | --------- | --------- |-----------  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 8                     | 32           | 32        | 1.21 mins | 2.05 mins   |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 8                     | 32           | 32        | 55.8 sec  | 1.91 mins   |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 8                     | 32           | 100       | 41 sec    | 1.64 mins   |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 8                     | 32           | 100       | 48 sec    | 1.61 mins   |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 8                     | 32           | 1000      | 1.36 min  | 1.83 mins   |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 8                     | 32           | 1000      | 48 sec    | 1.70 mins   |


{:.table-model-big.db}
| driver\_name      | driver\_memory | driver\_cores | worker\_name      | worker\_memory | worker\_cores | input\_data\_rows | output\_data\_rows | action           | total\_worker\_number | total\_cores | partition | NER timing|NER+RE timing|
| ----------------- | -------------- | ------------- | ----------------- | -------------- | ------------- | ----------------- | ------------------ | ---------------- | --------------------- | ------------ | --------- | --------- |-----------  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 4                     | 16           | 10        | 1.4 mins  |  3.78 mins  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 4                     | 16           | 10        | 1.76 mins |  3.93 mins  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 4                     | 16           | 16        | 1.41 mins |  3.97 mins  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 4                     | 16           | 16        | 1.42 mins |  3.82 mins  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 4                     | 16           | 32        | 1.36 mins |  3.70 mins  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 4                     | 16           | 32        | 1.35 mins |  3.65 mins  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 4                     | 16           | 100       | 1.21 mins |  3.18 mins  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 4                     | 16           | 100       | 1.24 mins |  3.15 mins  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 4                     | 16           | 1000      | 1.42 mins |  3.51 mins  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 4                     | 16           | 1000      | 1.46 mins |  3.48 mins  |

{:.table-model-big.db}
| driver\_name      | driver\_memory | driver\_cores | worker\_name      | worker\_memory | worker\_cores | input\_data\_rows | output\_data\_rows | action           | total\_worker\_number | total\_cores | partition | NER timing|NER+RE timing|
| ----------------- | -------------- | ------------- | ----------------- | -------------- | ------------- | ----------------- | ------------------ | ---------------- | --------------------- | ------------ | --------- | --------- |------------ |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 2                     | 8            | 10        | 2.82 mins | 5.91 mins  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 2                     | 8            | 10        | 2.82 mins | 5.99 mins  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 2                     | 8            | 100       | 2.27 mins | 5.29 mins  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 2                     | 8            | 100       | 2.25 min  | 5.26 mins  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 2                     | 8            | 1000      | 2.65 mins | 5.78 mins  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 2                     | 8            | 1000      | 2.7 mins  | 5.81 mins  |


{:.table-model-big.db}
| driver\_name      | driver\_memory | driver\_cores | worker\_name      | worker\_memory | worker\_cores | input\_data\_rows | output\_data\_rows | action           | total\_worker\_number | total\_cores | partition | NER timing|NER+RE timing|
| ----------------- | -------------- | ------------- | ----------------- | -------------- | ------------- | ----------------- | ------------------ | ---------------- | --------------------- | ------------ | --------- | --------- |----------- |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 1                     | 4            | 4         | 4.64 mins | 13.97 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 1                     | 4            | 4         | 4.53 mins | 13.88 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 1                     | 4            | 10        | 4.42 mins | 14.13 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 1                     | 4            | 10        | 4.55 mins | 14.63 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 1                     | 4            | 100       | 4.19 mins | 14.68 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 1                     | 4            | 100       | 4.18 mins | 14.89 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 1                     | 4            | 1000      | 5.01 mins | 16.38 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 1                     | 4            | 1000      | 4.99 mins | 16.52 mins |



</div>
<div class="h3-box" markdown="1">

### Clinical Bert For Token Classification Benchmark Experiment

- **Dataset :** 7537 Clinical Texts from PubMed Dataset
- **Driver :** Standard_DS3_v2 - 14GB Memory - 4 Cores
- **Enable Autoscaling :** True
- **Cluster Mode :** Standart
- **Worker :**
  - Standard_DS3_v2 - 14GB Memory - 4 Cores
- **Versions :**
  - **Databricks Runtime Version :** 10.0 (Apache Spark 3.2.0, Scala 2.12)
  - **spark-nlp Version:** v3.4.0
  - **spark-nlp-jsl Version :** v3.4.0
  - **Spark Version :** v3.2.0
- **Spark NLP Pipeline :**

```
nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        ner_jsl_slim_tokenClassifier,
        ner_converter,
        finisher])
```

**NOTES :**

+ In this experiment, the `bert_token_classifier_ner_jsl_slim` model was used to measure the inference time of clinical bert for token classification models in the databricks environment.
+ In the first experiment, the data read from the parquet file is saved as parquet after processing.

+ In the second experiment, the data read from the delta table was written to the delta table after it was processed.

</div>

#### Bert For Token Classification Benchmark Table

<div class="h3-box" markdown="1">

<table class="table-model-big table3">
    <thead>
        <tr>
            <th></th>
            <th>Repartition</th>
            <th>Time</th>
        </tr>
    </thead>    
    <tbody>
        <tr>
            <td rowspan="4"><strong>Read data from parquet</strong></td>
            <td>2</td>
            <td>26.03 mins</td>
        </tr>
        <tr>
            <td>64</td>
            <td>10.84 mins</td>
        </tr>
        <tr>
            <td>128</td>
            <td>7.53 mins</td>
        </tr>
        <tr>
            <td>1000</td>
            <td>8.93 mins</td>
        </tr>
        <tr>
            <td rowspan="4"><strong>Read data from delta table</strong></td>
            <td>2</td>
            <td>40.50 mins</td>
        </tr>
        <tr>
            <td>64</td>
            <td>11.84 mins</td>
        </tr>
        <tr>
            <td>128</td>
            <td>6.79 mins</td>
        </tr>
        <tr>
            <td>1000</td>
            <td>6.92 mins</td>
        </tr>
    </tbody>
</table>

</div>



<div class="h3-box" markdown="1">

### NER speed benchmarks across various Spark NLP and PySpark versions

This experiment compares the ClinicalNER runtime for different versions of `PySpark` and `Spark NLP`. 
In this experiment, all reports went through the pipeline 10 times and repeated execution 5 times, so we ran each report 50 times and averaged it, `%timeit -r 5 -n 10 run_model(spark, model)`.

- **Driver :** Standard Google Colab environment
- **Spark NLP Pipeline :**
```
nlpPipeline = Pipeline(
      stages=[
          documentAssembler,
          sentenceDetector,
          tokenizer,
          word_embeddings,
          clinical_ner,
          ner_converter
          ])
```

- **Dataset :** File sizes:
  - report_1: ~5.34kb
  - report_2: ~8.51kb
  - report_3: ~11.05kb
  - report_4: ~15.67kb
  - report_5: ~35.23kb


|          |Spark NLP 4.0.0 (PySpark 3.1.2) |Spark NLP 4.2.1 (PySpark 3.3.1) |Spark NLP 4.2.1 (PySpark 3.1.2) |Spark NLP 4.2.2 (PySpark 3.1.2) |Spark NLP 4.2.2 (PySpark 3.3.1) |Spark NLP 4.2.3 (PySpark 3.3.1) |Spark NLP 4.2.3 (PySpark 3.1.2) |
|:---------|-------------------------------:|-------------------------------:|-------------------------------:|-------------------------------:|-------------------------------:|-------------------------------:|-------------------------------:|
| report_1 |                        2.36066 |                        3.33056 |                        2.23723 |                        2.27243 |                        2.11513 |                        2.19655 |                        2.23915 |
| report_2 |                        2.2179  |                        3.31328 |                        2.15578 |                        2.23432 |                        2.07259 |                        2.07567 |                        2.16776 |
| report_3 |                        2.77923 |                        2.6134  |                        2.69023 |                        2.76358 |                        2.55306 |                        2.4424  |                        2.72496 |
| report_4 |                        4.41064 |                        4.07398 |                        4.66656 |                        4.59879 |                        3.98586 |                        3.92184 |                        4.6145  |
| report_5 |                        9.54389 |                        7.79465 |                        9.25499 |                        9.42764 |                        8.02252 |                        8.11318 |                        9.46555 |


Results show that the different versions can have some variance in the execution time, but the difference is not too relevant. 

</div>