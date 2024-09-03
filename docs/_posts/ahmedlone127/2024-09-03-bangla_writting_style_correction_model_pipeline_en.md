---
layout: model
title: English bangla_writting_style_correction_model_pipeline pipeline MarianTransformer from shahidul034
author: John Snow Labs
name: bangla_writting_style_correction_model_pipeline
date: 2024-09-03
tags: [en, open_source, pipeline, onnx]
task: Translation
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bangla_writting_style_correction_model_pipeline` is a English model originally trained by shahidul034.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bangla_writting_style_correction_model_pipeline_en_5.5.0_3.0_1725403793906.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bangla_writting_style_correction_model_pipeline_en_5.5.0_3.0_1725403793906.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bangla_writting_style_correction_model_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bangla_writting_style_correction_model_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bangla_writting_style_correction_model_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|531.0 MB|

## References

https://huggingface.co/shahidul034/Bangla_writting_style_correction_model

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer