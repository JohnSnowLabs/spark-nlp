---
layout: model
title: English emnlp_test_3clusters_medical_pipeline pipeline BartTransformer from hungngo04
author: John Snow Labs
name: emnlp_test_3clusters_medical_pipeline
date: 2025-04-01
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`emnlp_test_3clusters_medical_pipeline` is a English model originally trained by hungngo04.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/emnlp_test_3clusters_medical_pipeline_en_5.5.1_3.0_1743491071137.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/emnlp_test_3clusters_medical_pipeline_en_5.5.1_3.0_1743491071137.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("emnlp_test_3clusters_medical_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("emnlp_test_3clusters_medical_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|emnlp_test_3clusters_medical_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|518.2 MB|

## References

https://huggingface.co/hungngo04/emnlp_test_3clusters_medical

## Included Models

- DocumentAssembler
- BartTransformer