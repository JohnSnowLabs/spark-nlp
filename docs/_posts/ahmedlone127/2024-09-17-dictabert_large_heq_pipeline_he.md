---
layout: model
title: Hebrew dictabert_large_heq_pipeline pipeline BertForQuestionAnswering from dicta-il
author: John Snow Labs
name: dictabert_large_heq_pipeline
date: 2024-09-17
tags: [he, open_source, pipeline, onnx]
task: Question Answering
language: he
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`dictabert_large_heq_pipeline` is a Hebrew model originally trained by dicta-il.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dictabert_large_heq_pipeline_he_5.5.0_3.0_1726544293376.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/dictabert_large_heq_pipeline_he_5.5.0_3.0_1726544293376.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("dictabert_large_heq_pipeline", lang = "he")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("dictabert_large_heq_pipeline", lang = "he")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|dictabert_large_heq_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|he|
|Size:|1.6 GB|

## References

https://huggingface.co/dicta-il/dictabert-large-heq

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering