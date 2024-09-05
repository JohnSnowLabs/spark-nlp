---
layout: model
title: Estonian estroberta_pipeline pipeline XlmRoBertaEmbeddings from tartuNLP
author: John Snow Labs
name: estroberta_pipeline
date: 2024-09-03
tags: [et, open_source, pipeline, onnx]
task: Embeddings
language: et
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`estroberta_pipeline` is a Estonian model originally trained by tartuNLP.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/estroberta_pipeline_et_5.5.0_3.0_1725400175648.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/estroberta_pipeline_et_5.5.0_3.0_1725400175648.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("estroberta_pipeline", lang = "et")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("estroberta_pipeline", lang = "et")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|estroberta_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|et|
|Size:|1.0 GB|

## References

https://huggingface.co/tartuNLP/EstRoBERTa

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaEmbeddings