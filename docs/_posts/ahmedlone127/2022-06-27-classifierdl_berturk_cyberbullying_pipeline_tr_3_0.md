---
layout: model
title: Cyberbullying Classifier Pipeline in Turkish texts
author: John Snow Labs
name: classifierdl_berturk_cyberbullying_pipeline
date: 2022-06-27
tags: [tr, cyberbullying, pipeline, open_source]
task: Sentiment Analysis
language: tr
edition: Spark NLP 4.0.0
spark_version: 3.0
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_berturk_cyberbullying_pipeline_tr_4.0.0_3.0_1656361913070.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|tr|
|Size:|454.6 MB|

## Included Models

- DocumentAssembler
- TokenizerModel
- NormalizerModel
- StopWordsCleaner
- LemmatizerModel
- BertEmbeddings
- SentenceEmbeddings
- ClassifierDLModel