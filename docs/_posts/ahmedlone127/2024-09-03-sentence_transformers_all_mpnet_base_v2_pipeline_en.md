---
layout: model
title: English sentence_transformers_all_mpnet_base_v2_pipeline pipeline MPNetEmbeddings from ai-human-lab
author: John Snow Labs
name: sentence_transformers_all_mpnet_base_v2_pipeline
date: 2024-09-03
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

Pretrained MPNetEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sentence_transformers_all_mpnet_base_v2_pipeline` is a English model originally trained by ai-human-lab.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_transformers_all_mpnet_base_v2_pipeline_en_5.5.0_3.0_1725350443151.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentence_transformers_all_mpnet_base_v2_pipeline_en_5.5.0_3.0_1725350443151.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sentence_transformers_all_mpnet_base_v2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sentence_transformers_all_mpnet_base_v2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentence_transformers_all_mpnet_base_v2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|406.7 MB|

## References

https://huggingface.co/ai-human-lab/sentence-transformers_all-mpnet-base-v2

## Included Models

- DocumentAssembler
- MPNetEmbeddings