---
layout: model
title: English robako_base_asante_twi_uncased_pipeline pipeline RoBertaEmbeddings from Ghana-NLP
author: John Snow Labs
name: robako_base_asante_twi_uncased_pipeline
date: 2024-09-04
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`robako_base_asante_twi_uncased_pipeline` is a English model originally trained by Ghana-NLP.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/robako_base_asante_twi_uncased_pipeline_en_5.5.0_3.0_1725412678556.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/robako_base_asante_twi_uncased_pipeline_en_5.5.0_3.0_1725412678556.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("robako_base_asante_twi_uncased_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("robako_base_asante_twi_uncased_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|robako_base_asante_twi_uncased_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|311.8 MB|

## References

https://huggingface.co/Ghana-NLP/robako-base-asante-twi-uncased

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings