---
layout: model
title: Sentence Embeddings - sbiobert (tuned)
author: John Snow Labs
name: sbiobert_jsl_cased
date: 2021-06-30
tags: [embeddings, clinical, licensed, en]
task: Embeddings
language: en
edition: Spark NLP for Healthcare 3.1.0
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is trained to generate contextual sentence embeddings of input sentences.

{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/demo){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobert_jsl_cased_en_3.1.0_2.4_1625050229429.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

This sample snippet may not include all the required components of the pipeline for readability purposes. However, you can find a complete example of all the end-to-end components of the pipeline by clicking the "Open in Colab" link included above.




<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
sbiobert_embeddings = BertSentenceEmbeddings         .pretrained("sbiobert_jsl_cased","en","clinical/models")         .setInputCols(["sentence"])         .setOutputCol("sbert_embeddings")
```
```scala
val sbiobert_embeddings = BertSentenceEmbeddings
        .pretrained("sbiobert_jsl_cased","en","clinical/models")
        .setInputCols(Array("sentence"))
        .setOutputCol("sbert_embeddings")
```
</div>

## Results

```bash
Gives a 768 dimensional vector representation of the sentence.
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobert_jsl_cased|
|Compatibility:|Spark NLP for Healthcare 3.1.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Case sensitive:|true|

## Data Source

Tuned on MedNLI and UMLS dataset

## Benchmarking

```bash
MedNLI Acc: 0.788, STS (cos): 0.736
```

## Benchmarking

```bash
MedNLI Acc: 0.788, STS (cos): 0.736
```