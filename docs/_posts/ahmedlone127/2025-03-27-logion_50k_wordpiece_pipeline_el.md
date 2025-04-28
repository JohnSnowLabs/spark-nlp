---
layout: model
title: Modern Greek (1453-) logion_50k_wordpiece_pipeline pipeline BertEmbeddings from princeton-logion
author: John Snow Labs
name: logion_50k_wordpiece_pipeline
date: 2025-03-27
tags: [el, open_source, pipeline, onnx]
task: Embeddings
language: el
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`logion_50k_wordpiece_pipeline` is a Modern Greek (1453-) model originally trained by princeton-logion.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/logion_50k_wordpiece_pipeline_el_5.5.1_3.0_1743110099535.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/logion_50k_wordpiece_pipeline_el_5.5.1_3.0_1743110099535.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("logion_50k_wordpiece_pipeline", lang = "el")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("logion_50k_wordpiece_pipeline", lang = "el")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|logion_50k_wordpiece_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|el|
|Size:|464.1 MB|

## References

https://huggingface.co/princeton-logion/LOGION-50k_wordpiece

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings