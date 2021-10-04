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

## NER BENCHMARK EXPERIMENT

### Explanation of Benchmark Experiment

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

**The first experiment with 5 different cluster configurations :** `ner_chunk`  as a column in Spark NLP Pipeline (`ner_converter`) output data frame, exploded (lazy evaluation) as `ner_chunk` and `ner_label`. Then results were written as **parquet** and **delta** formats.

**A second experiment with 2 different cluster configuration :** Spark NLP Pipeline output data frame (except `word_embeddings` column) was written as **parquet** and **delta** formats.

In the first experiment with the most basic driver node and worker **(1 worker x 4 cores)** configuration selection, it took **4.64 mins** and **4.53 mins** to write **4 partitioned data** as parquet and delta formats respectively.

With basic driver node and **8 workers (x8 cores)** configuration selection, it took **40 seconds** and **22 seconds** to write **1000 partitioned data** as parquet and delta formats respectively.

In the second experiment with basic driver node and **4 workers (x 4 cores)** configuration selection, it took **1.41 mins** as parquet and **1.42 mins** as delta format to write **16 partitioned (exploded results) data**.  **Without explode it took 1.08 mins as parquet and 1.12 mins as delta format to write the data frame.**

Since given computation durations are highly dependent on different parameters including driver node and worker node configurations as well as partitions, **results show that explode method increases duration  %10-30  on chosen configurations.**

</div>
<div class="h3-box" markdown="1">

### BENCHMARK TABLES


{:.table-model-big.db}
| driver\_name      | driver\_memory | driver\_cores | worker\_name      | worker\_memory | worker\_cores | input\_data\_rows | output\_data\_rows | action           | total\_worker\_number | total\_cores | partition | duration  |
| ----------------- | -------------- | ------------- | ----------------- | -------------- | ------------- | ----------------- | ------------------ | ---------------- | --------------------- | ------------ | --------- | --------- |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 1                     | 4            | 4         | 4.64 min  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 1                     | 4            | 4         | 4.53 min  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 1                     | 4            | 10        | 4.42 min  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 1                     | 4            | 10        | 4.55 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 1                     | 4            | 100       | 4.19 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 1                     | 4            | 100       | 4.18 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 1                     | 4            | 1000      | 5.01 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 1                     | 4            | 1000      | 4.99 mins |


{:.table-model-big.db}
| driver\_name      | driver\_memory | driver\_cores | worker\_name      | worker\_memory | worker\_cores | input\_data\_rows | output\_data\_rows | action           | total\_worker\_number | total\_cores | partition | duration  |
| ----------------- | -------------- | ------------- | ----------------- | -------------- | ------------- | ----------------- | ------------------ | ---------------- | --------------------- | ------------ | --------- | --------- |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 2                     | 8            | 10        | 2.82 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 2                     | 8            | 10        | 2.82 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 2                     | 8            | 100       | 2.27 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 2                     | 8            | 100       | 2.25 min  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 2                     | 8            | 1000      | 2.65 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 2                     | 8            | 1000      | 2.7 mins  |

{:.table-model-big.db}
| driver\_name      | driver\_memory | driver\_cores | worker\_name      | worker\_memory | worker\_cores | input\_data\_rows | output\_data\_rows | action           | total\_worker\_number | total\_cores | partition | duration  |
| ----------------- | -------------- | ------------- | ----------------- | -------------- | ------------- | ----------------- | ------------------ | ---------------- | --------------------- | ------------ | --------- | --------- |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 4                     | 16           | 10        | 1.4 mins  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 4                     | 16           | 10        | 1.76 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 4                     | 16           | 16        | 1.41 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 4                     | 16           | 16        | 1.42 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 4                     | 16           | 32        | 1.36 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 4                     | 16           | 32        | 1.35 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 4                     | 16           | 100       | 1.21 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 4                     | 16           | 100       | 1.24 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 4                     | 16           | 1000      | 1.42 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 4                     | 16           | 1000      | 1.46 mins |

{:.table-model-big.db}
| driver\_name      | driver\_memory | driver\_cores | worker\_name      | worker\_memory | worker\_cores | input\_data\_rows | output\_data\_rows | action           | total\_worker\_number | total\_cores | partition | duration  |
| ----------------- | -------------- | ------------- | ----------------- | -------------- | ------------- | ----------------- | ------------------ | ---------------- | --------------------- | ------------ | --------- | --------- |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 1000               | write\_parquet   | 4                     | 16           | 10        | 1.39 min  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 1000               | write\_deltalake | 4                     | 16           | 10        | 1.33 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 1000               | write\_parquet   | 4                     | 16           | 16        | 1.08 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 1000               | write\_deltalake | 4                     | 16           | 16        | 1,12 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 1000               | write\_parquet   | 4                     | 16           | 32        | 1.09 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 1000               | write\_deltalake | 4                     | 16           | 32        | 1.23 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 1000               | write\_parquet   | 4                     | 16           | 100       | 1.05 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 1000               | write\_deltalake | 4                     | 16           | 100       | 1.11 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 1000               | write\_parquet   | 4                     | 16           | 1000      | 1.37 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 1000               | write\_deltalake | 4                     | 16           | 1000      | 1.29 mins |

{:.table-model-big.db}
| driver\_name      | driver\_memory | driver\_cores | worker\_name      | worker\_memory | worker\_cores | input\_data\_rows | output\_data\_rows | action           | total\_worker\_number | total\_cores | partition | duration  |
| ----------------- | -------------- | ------------- | ----------------- | -------------- | ------------- | ----------------- | ------------------ | ---------------- | --------------------- | ------------ | --------- | --------- |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 8                     | 32           | 32        | 1.21 mins |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 8                     | 32           | 32        | 55.8 sec  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 8                     | 32           | 100       | 41 sec    |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 8                     | 32           | 100       | 48 sec    |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_parquet   | 8                     | 32           | 1000      | 1.36 min  |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v3 | 16 GB          | 4             | 1000              | 78000              | write\_deltalake | 8                     | 32           | 1000      | 48 sec    |

{:.table-model-big.db}
| driver\_name      | driver\_memory | driver\_cores | worker\_name      | worker\_memory | worker\_cores | input\_data\_rows | output\_data\_rows | action           | total\_worker\_number | total\_cores | partition | duration |
| ----------------- | -------------- | ------------- | ----------------- | -------------- | ------------- | ----------------- | ------------------ | ---------------- | --------------------- | ------------ | --------- | -------- |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v2 | 28 GB          | 8             | 1000              | 78000              | write\_parquet   | 8                     | 64           | 64        | 36 sec   |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v2 | 28 GB          | 8             | 1000              | 78000              | write\_deltalake | 8                     | 64           | 64        | 19 sec   |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v2 | 28 GB          | 8             | 1000              | 78000              | write\_parquet   | 8                     | 64           | 100       | 21 sec   |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v2 | 28 GB          | 8             | 1000              | 78000              | write\_deltalake | 8                     | 64           | 100       | 41 sec   |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v2 | 28 GB          | 8             | 1000              | 78000              | write\_parquet   | 8                     | 64           | 1000      | 40 sec   |
| Standard\_D4s\_v3 | 16 GB          | 4             | Standard\_D4s\_v2 | 28 GB          | 8             | 1000              | 78000              | write\_deltalake | 8                     | 64           | 1000      | 22 sec   |

</div>