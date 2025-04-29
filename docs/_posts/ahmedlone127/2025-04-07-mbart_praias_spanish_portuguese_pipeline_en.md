---
layout: model
title: English mbart_praias_spanish_portuguese_pipeline pipeline MarianTransformer from feserrm
author: John Snow Labs
name: mbart_praias_spanish_portuguese_pipeline
date: 2025-04-07
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mbart_praias_spanish_portuguese_pipeline` is a English model originally trained by feserrm.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mbart_praias_spanish_portuguese_pipeline_en_5.5.1_3.0_1744019572189.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mbart_praias_spanish_portuguese_pipeline_en_5.5.1_3.0_1744019572189.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mbart_praias_spanish_portuguese_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mbart_praias_spanish_portuguese_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mbart_praias_spanish_portuguese_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|418.0 MB|

## References

https://huggingface.co/feserrm/mbart-praias_es-pt

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer