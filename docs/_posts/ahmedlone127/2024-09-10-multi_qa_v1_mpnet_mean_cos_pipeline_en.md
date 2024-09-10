---
layout: model
title: English multi_qa_v1_mpnet_mean_cos_pipeline pipeline MPNetEmbeddings from flax-sentence-embeddings
author: John Snow Labs
name: multi_qa_v1_mpnet_mean_cos_pipeline
date: 2024-09-10
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

Pretrained MPNetEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`multi_qa_v1_mpnet_mean_cos_pipeline` is a English model originally trained by flax-sentence-embeddings.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/multi_qa_v1_mpnet_mean_cos_pipeline_en_5.5.0_3.0_1725978238808.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/multi_qa_v1_mpnet_mean_cos_pipeline_en_5.5.0_3.0_1725978238808.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("multi_qa_v1_mpnet_mean_cos_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("multi_qa_v1_mpnet_mean_cos_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|multi_qa_v1_mpnet_mean_cos_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|406.8 MB|

## References

https://huggingface.co/flax-sentence-embeddings/multi-qa_v1-mpnet-mean_cos

## Included Models

- DocumentAssembler
- MPNetEmbeddings