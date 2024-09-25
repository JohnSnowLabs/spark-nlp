---
layout: model
title: English albertv2_dc_unsorted_dec_cf_pipeline pipeline BertForSequenceClassification from rpii2023
author: John Snow Labs
name: albertv2_dc_unsorted_dec_cf_pipeline
date: 2024-09-25
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`albertv2_dc_unsorted_dec_cf_pipeline` is a English model originally trained by rpii2023.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albertv2_dc_unsorted_dec_cf_pipeline_en_5.5.0_3.0_1727239484248.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albertv2_dc_unsorted_dec_cf_pipeline_en_5.5.0_3.0_1727239484248.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("albertv2_dc_unsorted_dec_cf_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("albertv2_dc_unsorted_dec_cf_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albertv2_dc_unsorted_dec_cf_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|409.4 MB|

## References

https://huggingface.co/rpii2023/albertv2_DC_unsorted_DEC_CF

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification