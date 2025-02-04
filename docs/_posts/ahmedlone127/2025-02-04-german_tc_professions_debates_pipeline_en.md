---
layout: model
title: English german_tc_professions_debates_pipeline pipeline BertForTokenClassification from johannabi
author: John Snow Labs
name: german_tc_professions_debates_pipeline
date: 2025-02-04
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`german_tc_professions_debates_pipeline` is a English model originally trained by johannabi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/german_tc_professions_debates_pipeline_en_5.5.1_3.0_1738629723863.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/german_tc_professions_debates_pipeline_en_5.5.1_3.0_1738629723863.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("german_tc_professions_debates_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("german_tc_professions_debates_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|german_tc_professions_debates_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|406.9 MB|

## References

https://huggingface.co/johannabi/german_tc_professions_debates

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification