---
layout: model
title: English tmvar_2e_05_0404_es6_strict_tok_pipeline pipeline BertForTokenClassification from Brizape
author: John Snow Labs
name: tmvar_2e_05_0404_es6_strict_tok_pipeline
date: 2025-02-08
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tmvar_2e_05_0404_es6_strict_tok_pipeline` is a English model originally trained by Brizape.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tmvar_2e_05_0404_es6_strict_tok_pipeline_en_5.5.1_3.0_1738989263455.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tmvar_2e_05_0404_es6_strict_tok_pipeline_en_5.5.1_3.0_1738989263455.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tmvar_2e_05_0404_es6_strict_tok_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tmvar_2e_05_0404_es6_strict_tok_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tmvar_2e_05_0404_es6_strict_tok_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|408.2 MB|

## References

https://huggingface.co/Brizape/tmvar_2e-05_0404_ES6_strict_tok

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification