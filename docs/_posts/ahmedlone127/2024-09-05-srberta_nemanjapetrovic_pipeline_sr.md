---
layout: model
title: Serbian srberta_nemanjapetrovic_pipeline pipeline RoBertaEmbeddings from nemanjaPetrovic
author: John Snow Labs
name: srberta_nemanjapetrovic_pipeline
date: 2024-09-05
tags: [sr, open_source, pipeline, onnx]
task: Embeddings
language: sr
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`srberta_nemanjapetrovic_pipeline` is a Serbian model originally trained by nemanjaPetrovic.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/srberta_nemanjapetrovic_pipeline_sr_5.5.0_3.0_1725572680810.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/srberta_nemanjapetrovic_pipeline_sr_5.5.0_3.0_1725572680810.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("srberta_nemanjapetrovic_pipeline", lang = "sr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("srberta_nemanjapetrovic_pipeline", lang = "sr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|srberta_nemanjapetrovic_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|sr|
|Size:|466.3 MB|

## References

https://huggingface.co/nemanjaPetrovic/SrBERTa

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings