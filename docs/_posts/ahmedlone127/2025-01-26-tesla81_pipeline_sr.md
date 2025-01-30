---
layout: model
title: Serbian tesla81_pipeline pipeline RoBertaEmbeddings from te-sla
author: John Snow Labs
name: tesla81_pipeline
date: 2025-01-26
tags: [sr, open_source, pipeline, onnx]
task: Embeddings
language: sr
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tesla81_pipeline` is a Serbian model originally trained by te-sla.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tesla81_pipeline_sr_5.5.1_3.0_1737866411329.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tesla81_pipeline_sr_5.5.1_3.0_1737866411329.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tesla81_pipeline", lang = "sr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tesla81_pipeline", lang = "sr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tesla81_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|sr|
|Size:|290.5 MB|

## References

https://huggingface.co/te-sla/Tesla81

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings