---
layout: model
title: English mnli_6x9n_var33_9_pipeline pipeline RoBertaEmbeddings from mahdiyar
author: John Snow Labs
name: mnli_6x9n_var33_9_pipeline
date: 2024-09-13
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mnli_6x9n_var33_9_pipeline` is a English model originally trained by mahdiyar.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mnli_6x9n_var33_9_pipeline_en_5.5.0_3.0_1726197801630.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mnli_6x9n_var33_9_pipeline_en_5.5.0_3.0_1726197801630.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mnli_6x9n_var33_9_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mnli_6x9n_var33_9_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mnli_6x9n_var33_9_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|460.3 MB|

## References

https://huggingface.co/mahdiyar/mnli-6x9N-var33-9

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings