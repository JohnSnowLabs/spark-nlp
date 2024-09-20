---
layout: model
title: Russian rupunct_big_pipeline pipeline BertForTokenClassification from RUPunct
author: John Snow Labs
name: rupunct_big_pipeline
date: 2024-09-05
tags: [ru, open_source, pipeline, onnx]
task: Named Entity Recognition
language: ru
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`rupunct_big_pipeline` is a Russian model originally trained by RUPunct.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/rupunct_big_pipeline_ru_5.5.0_3.0_1725539410649.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/rupunct_big_pipeline_ru_5.5.0_3.0_1725539410649.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("rupunct_big_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("rupunct_big_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|rupunct_big_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|667.1 MB|

## References

https://huggingface.co/RUPunct/RUPunct_big

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification