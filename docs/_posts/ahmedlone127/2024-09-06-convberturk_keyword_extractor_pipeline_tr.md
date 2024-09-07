---
layout: model
title: Turkish convberturk_keyword_extractor_pipeline pipeline BertForTokenClassification from yanekyuk
author: John Snow Labs
name: convberturk_keyword_extractor_pipeline
date: 2024-09-06
tags: [tr, open_source, pipeline, onnx]
task: Named Entity Recognition
language: tr
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`convberturk_keyword_extractor_pipeline` is a Turkish model originally trained by yanekyuk.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/convberturk_keyword_extractor_pipeline_tr_5.5.0_3.0_1725663337352.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/convberturk_keyword_extractor_pipeline_tr_5.5.0_3.0_1725663337352.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("convberturk_keyword_extractor_pipeline", lang = "tr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("convberturk_keyword_extractor_pipeline", lang = "tr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|convberturk_keyword_extractor_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|tr|
|Size:|400.1 MB|

## References

https://huggingface.co/yanekyuk/convberturk-keyword-extractor

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification