---
layout: model
title: Korean final_crf_pipeline pipeline BertForTokenClassification from jinwoowef
author: John Snow Labs
name: final_crf_pipeline
date: 2024-09-11
tags: [ko, open_source, pipeline, onnx]
task: Named Entity Recognition
language: ko
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`final_crf_pipeline` is a Korean model originally trained by jinwoowef.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/final_crf_pipeline_ko_5.5.0_3.0_1726025849144.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/final_crf_pipeline_ko_5.5.0_3.0_1726025849144.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("final_crf_pipeline", lang = "ko")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("final_crf_pipeline", lang = "ko")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|final_crf_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ko|
|Size:|425.9 MB|

## References

https://huggingface.co/jinwoowef/final_crf

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification