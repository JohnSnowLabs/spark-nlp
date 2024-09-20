---
layout: model
title: Indonesian roberta_embeddings_indonesian_roberta_large_pipeline pipeline RoBertaEmbeddings from flax-community
author: John Snow Labs
name: roberta_embeddings_indonesian_roberta_large_pipeline
date: 2024-09-03
tags: [id, open_source, pipeline, onnx]
task: Embeddings
language: id
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_embeddings_indonesian_roberta_large_pipeline` is a Indonesian model originally trained by flax-community.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_embeddings_indonesian_roberta_large_pipeline_id_5.5.0_3.0_1725375194313.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_embeddings_indonesian_roberta_large_pipeline_id_5.5.0_3.0_1725375194313.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_embeddings_indonesian_roberta_large_pipeline", lang = "id")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_embeddings_indonesian_roberta_large_pipeline", lang = "id")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_embeddings_indonesian_roberta_large_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|id|
|Size:|627.1 MB|

## References

https://huggingface.co/flax-community/indonesian-roberta-large

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings