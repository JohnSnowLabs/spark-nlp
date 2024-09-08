---
layout: model
title: Lithuanian scoris_maltese_lithuanian_english_pipeline pipeline MarianTransformer from scoris
author: John Snow Labs
name: scoris_maltese_lithuanian_english_pipeline
date: 2024-09-05
tags: [lt, open_source, pipeline, onnx]
task: Translation
language: lt
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`scoris_maltese_lithuanian_english_pipeline` is a Lithuanian model originally trained by scoris.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/scoris_maltese_lithuanian_english_pipeline_lt_5.5.0_3.0_1725545857263.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/scoris_maltese_lithuanian_english_pipeline_lt_5.5.0_3.0_1725545857263.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("scoris_maltese_lithuanian_english_pipeline", lang = "lt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("scoris_maltese_lithuanian_english_pipeline", lang = "lt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|scoris_maltese_lithuanian_english_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|lt|
|Size:|1.3 GB|

## References

https://huggingface.co/scoris/scoris-mt-lt-en

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer