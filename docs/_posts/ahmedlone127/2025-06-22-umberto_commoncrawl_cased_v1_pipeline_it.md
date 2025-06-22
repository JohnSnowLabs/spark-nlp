---
layout: model
title: Italian umberto_commoncrawl_cased_v1_pipeline pipeline CamemBertEmbeddings from Musixmatch
author: John Snow Labs
name: umberto_commoncrawl_cased_v1_pipeline
date: 2025-06-22
tags: [it, open_source, pipeline, onnx]
task: Embeddings
language: it
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`umberto_commoncrawl_cased_v1_pipeline` is a Italian model originally trained by Musixmatch.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/umberto_commoncrawl_cased_v1_pipeline_it_5.5.1_3.0_1750618383780.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/umberto_commoncrawl_cased_v1_pipeline_it_5.5.1_3.0_1750618383780.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("umberto_commoncrawl_cased_v1_pipeline", lang = "it")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("umberto_commoncrawl_cased_v1_pipeline", lang = "it")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|umberto_commoncrawl_cased_v1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|it|
|Size:|263.0 MB|

## References

References

https://huggingface.co/Musixmatch/umberto-commoncrawl-cased-v1

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertEmbeddings