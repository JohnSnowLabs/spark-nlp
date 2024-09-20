---
layout: model
title: Tsonga takalane_tso_roberta_pipeline pipeline RoBertaEmbeddings from jannesg
author: John Snow Labs
name: takalane_tso_roberta_pipeline
date: 2024-09-03
tags: [ts, open_source, pipeline, onnx]
task: Embeddings
language: ts
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`takalane_tso_roberta_pipeline` is a Tsonga model originally trained by jannesg.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/takalane_tso_roberta_pipeline_ts_5.5.0_3.0_1725381717542.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/takalane_tso_roberta_pipeline_ts_5.5.0_3.0_1725381717542.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("takalane_tso_roberta_pipeline", lang = "ts")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("takalane_tso_roberta_pipeline", lang = "ts")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|takalane_tso_roberta_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ts|
|Size:|310.8 MB|

## References

https://huggingface.co/jannesg/takalane_tso_roberta

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings