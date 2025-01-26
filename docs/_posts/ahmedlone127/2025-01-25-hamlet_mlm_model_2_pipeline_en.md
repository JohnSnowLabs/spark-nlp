---
layout: model
title: English hamlet_mlm_model_2_pipeline pipeline RoBertaEmbeddings from kazisami
author: John Snow Labs
name: hamlet_mlm_model_2_pipeline
date: 2025-01-25
tags: [en, open_source, pipeline, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hamlet_mlm_model_2_pipeline` is a English model originally trained by kazisami.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hamlet_mlm_model_2_pipeline_en_5.5.1_3.0_1737796983940.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hamlet_mlm_model_2_pipeline_en_5.5.1_3.0_1737796983940.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hamlet_mlm_model_2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hamlet_mlm_model_2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hamlet_mlm_model_2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|306.5 MB|

## References

https://huggingface.co/kazisami/hamlet_mlm_model_2

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings