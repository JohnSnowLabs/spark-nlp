---
layout: model
title: Hebrew hero_pipeline pipeline RoBertaEmbeddings from HeNLP
author: John Snow Labs
name: hero_pipeline
date: 2024-09-01
tags: [he, open_source, pipeline, onnx]
task: Embeddings
language: he
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hero_pipeline` is a Hebrew model originally trained by HeNLP.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hero_pipeline_he_5.4.2_3.0_1725187082484.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hero_pipeline_he_5.4.2_3.0_1725187082484.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hero_pipeline", lang = "he")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hero_pipeline", lang = "he")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hero_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|he|
|Size:|465.9 MB|

## References

https://huggingface.co/HeNLP/HeRo

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings