---
layout: model
title: Hungarian hupunct_pipeline pipeline BertForTokenClassification from gyenist
author: John Snow Labs
name: hupunct_pipeline
date: 2025-01-24
tags: [hu, open_source, pipeline, onnx]
task: Named Entity Recognition
language: hu
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hupunct_pipeline` is a Hungarian model originally trained by gyenist.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hupunct_pipeline_hu_5.5.1_3.0_1737719891142.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hupunct_pipeline_hu_5.5.1_3.0_1737719891142.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hupunct_pipeline", lang = "hu")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hupunct_pipeline", lang = "hu")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hupunct_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|hu|
|Size:|412.5 MB|

## References

https://huggingface.co/gyenist/hupunct

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification