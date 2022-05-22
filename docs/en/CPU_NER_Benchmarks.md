---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: CPU NER Benchmarks
permalink: /docs/en/cpu-ner-benchmark
key: docs-benchmark
modify_date: "2021-10-04"
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

## CPU NER Benchmarks

### NER (BiLSTM-CNN-Char Architecture) CPU Benchmark Experiment

- **Dataset :** 1000 Clinical Texts from MTSamples Oncology Dataset, approx. 500 tokens per text.
- **Versions :**
  - **spark-nlp Version:** v3.4.4
  - **spark-nlp-jsl Version :** v3.5.2
  - **Spark Version :** v3.1.2
- **Spark NLP Pipeline :**

  ```python
  nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,  
        embeddings_clinical,  
        clinical_ner,  
        ner_converter
        ])

  ```

**NOTE:**

- Spark NLP Pipeline output data frame (except `word_embeddings` column) was written as **parquet** format in `transform` benchmarks.

</div><div class="h3-box" markdown="1">

<table class="table-model-big table3">
    <thead>
        <tr>
            <th>Plarform</th>
            <th>Process</th>
            <th>Repartition</th>
            <th>Time</th>
        </tr>
    </thead>    
    <tbody>
        <tr>
            <td rowspan="4">2 CPU cores, 13 GB RAM (Google COLAB)</td>
            <td>LP (fullAnnotate)</td>
            <td>-</td>
            <td>16min 52s</td>
        </tr>
        <tr>
            <td rowspan="3">Transform (parquet)</td>
            <td>10</td>
            <td>4min 47s</td>
        </tr>
        <tr>
            <td>100</td>
            <td><strong>4min 16s</strong></td>
        </tr>
        <tr>
            <td>1000</td>
            <td>5min 4s</td>
        </tr>
        <tr>
            <td rowspan="4">16 CPU cores, 27 GB RAM (AWS EC2 machine)</td>
            <td>LP (fullAnnotate)</td>
            <td>-</td>
            <td>14min 28s</td>
        </tr>
        <tr>
            <td rowspan="3">Transform (parquet)</td>
            <td>10</td>
            <td>1min 5s</td>
        </tr>
        <tr>
            <td>100</td>
            <td><strong>1min 1s</strong></td>
        </tr>
        <tr>
            <td>1000</td>
            <td>1min 19s</td>
        </tr>
    </tbody>
</table>

</div>