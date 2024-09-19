---
layout: model
title: None pro_berta_pipeline pipeline RoBertaEmbeddings from donal
author: John Snow Labs
name: pro_berta_pipeline
date: 2024-09-18
tags: [nan, open_source, pipeline, onnx]
task: Embeddings
language: nan
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`pro_berta_pipeline` is a None model originally trained by donal.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pro_berta_pipeline_nan_5.5.0_3.0_1726652095316.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pro_berta_pipeline_nan_5.5.0_3.0_1726652095316.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("pro_berta_pipeline", lang = "nan")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("pro_berta_pipeline", lang = "nan")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pro_berta_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|nan|
|Size:|310.0 MB|

## References

https://huggingface.co/donal/Pro_Berta

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings