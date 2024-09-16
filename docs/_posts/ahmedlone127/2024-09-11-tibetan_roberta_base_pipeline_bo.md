---
layout: model
title: Tibetan tibetan_roberta_base_pipeline pipeline RoBertaEmbeddings from sangjeedondrub
author: John Snow Labs
name: tibetan_roberta_base_pipeline
date: 2024-09-11
tags: [bo, open_source, pipeline, onnx]
task: Embeddings
language: bo
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tibetan_roberta_base_pipeline` is a Tibetan model originally trained by sangjeedondrub.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tibetan_roberta_base_pipeline_bo_5.5.0_3.0_1726094027601.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tibetan_roberta_base_pipeline_bo_5.5.0_3.0_1726094027601.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tibetan_roberta_base_pipeline", lang = "bo")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tibetan_roberta_base_pipeline", lang = "bo")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tibetan_roberta_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|bo|
|Size:|311.2 MB|

## References

https://huggingface.co/sangjeedondrub/tibetan-roberta-base

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings