---
layout: model
title: English bc5cdr_disease_wlt_256_bluebert_latest_wlt_pipeline pipeline BertForTokenClassification from mobashgr
author: John Snow Labs
name: bc5cdr_disease_wlt_256_bluebert_latest_wlt_pipeline
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bc5cdr_disease_wlt_256_bluebert_latest_wlt_pipeline` is a English model originally trained by mobashgr.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bc5cdr_disease_wlt_256_bluebert_latest_wlt_pipeline_en_5.5.1_3.0_1739023360655.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bc5cdr_disease_wlt_256_bluebert_latest_wlt_pipeline_en_5.5.1_3.0_1739023360655.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bc5cdr_disease_wlt_256_bluebert_latest_wlt_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bc5cdr_disease_wlt_256_bluebert_latest_wlt_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bc5cdr_disease_wlt_256_bluebert_latest_wlt_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.1 MB|

## References

https://huggingface.co/mobashgr/BC5CDR-disease-WLT-256-BlueBERT-latest-wlt

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification