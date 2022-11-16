---
layout: model
title: Cyberbullying Classifier Pipeline in Turkish texts
author: John Snow Labs
name: classifierdl_berturk_cyberbullying_pipeline
date: 2021-08-13
tags: [tr, cyberbullying, pipeline, open_source]
task: Pipeline Public
language: tr
edition: Spark NLP 3.1.3
spark_version: 2.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pre-trained pipeline Identifies whether a Turkish text contains cyberbullying or not.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_berturk_cyberbullying_pipeline_tr_3.1.3_2.4_1628848526053.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("classifierdl_berturk_cyberbullying_pipeline", "tr")

result = pipeline.fullAnnotate("""Gidişin olsun, dönüşün olmasın inşallah senin..""")
```
```scala
val pipeline = new PretrainedPipeline("classifierdl_berturk_cyberbulling_pipeline", "tr")

val result = pipeline.fullAnnotate("Gidişin olsun, dönüşün olmasın inşallah senin..")(0)

```
</div>

## Results

```bash
["Negative"]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_berturk_cyberbullying_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.1.3+|
|License:|Open Source|
|Edition:|Official|
|Language:|tr|

## Included Models

- DocumentAssembler
- TokenizerModel
- NormalizerModel
- StopWordsCleaner
- LemmatizerModel
- BertEmbeddings
- SentenceEmbeddings
- ClassifierDLModel