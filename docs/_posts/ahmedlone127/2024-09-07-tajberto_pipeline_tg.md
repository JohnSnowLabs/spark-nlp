---
layout: model
title: Tajik tajberto_pipeline pipeline RoBertaEmbeddings from muhtasham
author: John Snow Labs
name: tajberto_pipeline
date: 2024-09-07
tags: [tg, open_source, pipeline, onnx]
task: Embeddings
language: tg
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tajberto_pipeline` is a Tajik model originally trained by muhtasham.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tajberto_pipeline_tg_5.5.0_3.0_1725673328834.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tajberto_pipeline_tg_5.5.0_3.0_1725673328834.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tajberto_pipeline", lang = "tg")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tajberto_pipeline", lang = "tg")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tajberto_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|tg|
|Size:|311.7 MB|

## References

https://huggingface.co/muhtasham/TajBERTo

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings