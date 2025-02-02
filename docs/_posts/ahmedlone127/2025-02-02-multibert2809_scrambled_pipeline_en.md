---
layout: model
title: English multibert2809_scrambled_pipeline pipeline BertForTokenClassification from Tommert25
author: John Snow Labs
name: multibert2809_scrambled_pipeline
date: 2025-02-02
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`multibert2809_scrambled_pipeline` is a English model originally trained by Tommert25.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/multibert2809_scrambled_pipeline_en_5.5.1_3.0_1738530737674.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/multibert2809_scrambled_pipeline_en_5.5.1_3.0_1738530737674.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("multibert2809_scrambled_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("multibert2809_scrambled_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|multibert2809_scrambled_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|625.6 MB|

## References

https://huggingface.co/Tommert25/multibert2809_scrambled

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification