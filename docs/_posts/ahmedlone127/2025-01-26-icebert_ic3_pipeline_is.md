---
layout: model
title: Icelandic icebert_ic3_pipeline pipeline RoBertaEmbeddings from mideind
author: John Snow Labs
name: icebert_ic3_pipeline
date: 2025-01-26
tags: [is, open_source, pipeline, onnx]
task: Embeddings
language: is
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`icebert_ic3_pipeline` is a Icelandic model originally trained by mideind.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/icebert_ic3_pipeline_is_5.5.1_3.0_1737906277908.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/icebert_ic3_pipeline_is_5.5.1_3.0_1737906277908.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("icebert_ic3_pipeline", lang = "is")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("icebert_ic3_pipeline", lang = "is")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|icebert_ic3_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|is|
|Size:|297.2 MB|

## References

https://huggingface.co/mideind/IceBERT-ic3

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings