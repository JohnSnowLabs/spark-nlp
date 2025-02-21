---
layout: model
title: English latin_italian_translatorv6_pipeline pipeline MarianTransformer from Dddixyy
author: John Snow Labs
name: latin_italian_translatorv6_pipeline
date: 2025-01-26
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`latin_italian_translatorv6_pipeline` is a English model originally trained by Dddixyy.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/latin_italian_translatorv6_pipeline_en_5.5.1_3.0_1737863287698.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/latin_italian_translatorv6_pipeline_en_5.5.1_3.0_1737863287698.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("latin_italian_translatorv6_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("latin_italian_translatorv6_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|latin_italian_translatorv6_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|476.3 MB|

## References

https://huggingface.co/Dddixyy/latin-italian-translatorV6

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer