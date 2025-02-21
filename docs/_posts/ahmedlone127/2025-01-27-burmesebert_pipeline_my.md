---
layout: model
title: Burmese burmesebert_pipeline pipeline BertEmbeddings from jojo-ai-mst
author: John Snow Labs
name: burmesebert_pipeline
date: 2025-01-27
tags: [my, open_source, pipeline, onnx]
task: Embeddings
language: my
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`burmesebert_pipeline` is a Burmese model originally trained by jojo-ai-mst.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/burmesebert_pipeline_my_5.5.1_3.0_1737985513688.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/burmesebert_pipeline_my_5.5.1_3.0_1737985513688.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("burmesebert_pipeline", lang = "my")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("burmesebert_pipeline", lang = "my")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|burmesebert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|my|
|Size:|1.4 GB|

## References

https://huggingface.co/jojo-ai-mst/BurmeseBert

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings