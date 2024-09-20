---
layout: model
title: Indonesian esteler_distilbert_indonesian_pipeline pipeline DistilBertEmbeddings from zaenalium
author: John Snow Labs
name: esteler_distilbert_indonesian_pipeline
date: 2024-09-10
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

Pretrained DistilBertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`esteler_distilbert_indonesian_pipeline` is a Indonesian model originally trained by zaenalium.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/esteler_distilbert_indonesian_pipeline_id_5.5.0_3.0_1725946536744.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/esteler_distilbert_indonesian_pipeline_id_5.5.0_3.0_1725946536744.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("esteler_distilbert_indonesian_pipeline", lang = "id")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("esteler_distilbert_indonesian_pipeline", lang = "id")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|esteler_distilbert_indonesian_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|id|
|Size:|307.6 MB|

## References

https://huggingface.co/zaenalium/Esteler-DistilBERT-id

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertEmbeddings