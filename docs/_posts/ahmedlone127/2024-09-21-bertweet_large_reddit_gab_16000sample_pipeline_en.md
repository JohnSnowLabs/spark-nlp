---
layout: model
title: English bertweet_large_reddit_gab_16000sample_pipeline pipeline RoBertaEmbeddings from HPL
author: John Snow Labs
name: bertweet_large_reddit_gab_16000sample_pipeline
date: 2024-09-21
tags: [en, open_source, pipeline, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bertweet_large_reddit_gab_16000sample_pipeline` is a English model originally trained by HPL.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bertweet_large_reddit_gab_16000sample_pipeline_en_5.5.0_3.0_1726957912393.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bertweet_large_reddit_gab_16000sample_pipeline_en_5.5.0_3.0_1726957912393.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bertweet_large_reddit_gab_16000sample_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bertweet_large_reddit_gab_16000sample_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bertweet_large_reddit_gab_16000sample_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/HPL/bertweet-large-reddit-gab-16000sample

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings