---
layout: model
title: English dummy_model_vicente_felino_pipeline pipeline CamemBertEmbeddings from Vicente-Felino
author: John Snow Labs
name: dummy_model_vicente_felino_pipeline
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

Pretrained CamemBertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`dummy_model_vicente_felino_pipeline` is a English model originally trained by Vicente-Felino.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dummy_model_vicente_felino_pipeline_en_5.5.0_3.0_1726007310745.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/dummy_model_vicente_felino_pipeline_en_5.5.0_3.0_1726007310745.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("dummy_model_vicente_felino_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("dummy_model_vicente_felino_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|dummy_model_vicente_felino_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|264.0 MB|

## References

https://huggingface.co/Vicente-Felino/dummy-model

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertEmbeddings