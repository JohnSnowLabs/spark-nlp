---
layout: model
title: English indicbert_finetuned_tamil_pipeline pipeline AlbertEmbeddings from sanufa
author: John Snow Labs
name: indicbert_finetuned_tamil_pipeline
date: 2025-03-27
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

Pretrained AlbertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`indicbert_finetuned_tamil_pipeline` is a English model originally trained by sanufa.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/indicbert_finetuned_tamil_pipeline_en_5.5.1_3.0_1743102948276.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/indicbert_finetuned_tamil_pipeline_en_5.5.1_3.0_1743102948276.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("indicbert_finetuned_tamil_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("indicbert_finetuned_tamil_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|indicbert_finetuned_tamil_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|125.5 MB|

## References

https://huggingface.co/sanufa/indicbert-finetuned-tamil

## Included Models

- DocumentAssembler
- TokenizerModel
- AlbertEmbeddings