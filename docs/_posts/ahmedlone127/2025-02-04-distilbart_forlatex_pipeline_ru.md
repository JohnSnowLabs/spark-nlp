---
layout: model
title: Russian distilbart_forlatex_pipeline pipeline BartTransformer from kostyabuh21
author: John Snow Labs
name: distilbart_forlatex_pipeline
date: 2025-02-04
tags: [ru, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: ru
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilbart_forlatex_pipeline` is a Russian model originally trained by kostyabuh21.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbart_forlatex_pipeline_ru_5.5.1_3.0_1738706488835.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbart_forlatex_pipeline_ru_5.5.1_3.0_1738706488835.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distilbart_forlatex_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distilbart_forlatex_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbart_forlatex_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|1.5 GB|

## References

https://huggingface.co/kostyabuh21/DistilBART_forLaTeX

## Included Models

- DocumentAssembler
- BartTransformer