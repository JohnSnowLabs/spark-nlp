---
layout: model
title: English indicbert_urdu_pipeline pipeline AlbertForTokenClassification from anwesham
author: John Snow Labs
name: indicbert_urdu_pipeline
date: 2024-09-04
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained AlbertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`indicbert_urdu_pipeline` is a English model originally trained by anwesham.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/indicbert_urdu_pipeline_en_5.5.0_3.0_1725486628645.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/indicbert_urdu_pipeline_en_5.5.0_3.0_1725486628645.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("indicbert_urdu_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("indicbert_urdu_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|indicbert_urdu_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|125.6 MB|

## References

https://huggingface.co/anwesham/indicbert_ur

## Included Models

- DocumentAssembler
- TokenizerModel
- AlbertForTokenClassification