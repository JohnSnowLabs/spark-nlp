---
layout: model
title: English hospital_ai_nupe_tonga_tonga_islands_yor_pipeline pipeline MarianTransformer from gabri3l
author: John Snow Labs
name: hospital_ai_nupe_tonga_tonga_islands_yor_pipeline
date: 2025-02-03
tags: [en, open_source, pipeline, onnx]
task: Translation
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hospital_ai_nupe_tonga_tonga_islands_yor_pipeline` is a English model originally trained by gabri3l.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hospital_ai_nupe_tonga_tonga_islands_yor_pipeline_en_5.5.1_3.0_1738615898723.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hospital_ai_nupe_tonga_tonga_islands_yor_pipeline_en_5.5.1_3.0_1738615898723.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hospital_ai_nupe_tonga_tonga_islands_yor_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hospital_ai_nupe_tonga_tonga_islands_yor_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hospital_ai_nupe_tonga_tonga_islands_yor_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|415.1 MB|

## References

https://huggingface.co/gabri3l/hospital-ai-nupe-to-yor

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer