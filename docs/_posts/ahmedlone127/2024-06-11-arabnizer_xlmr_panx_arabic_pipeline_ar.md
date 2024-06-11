---
layout: model
title: Arabic arabnizer_xlmr_panx_arabic_pipeline pipeline XlmRoBertaForTokenClassification from mohammedaly22
author: John Snow Labs
name: arabnizer_xlmr_panx_arabic_pipeline
date: 2024-06-11
tags: [ar, open_source, pipeline, onnx]
task: Named Entity Recognition
language: ar
edition: Spark NLP 5.4.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`arabnizer_xlmr_panx_arabic_pipeline` is a Arabic model originally trained by mohammedaly22.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/arabnizer_xlmr_panx_arabic_pipeline_ar_5.4.0_3.0_1718131237090.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/arabnizer_xlmr_panx_arabic_pipeline_ar_5.4.0_3.0_1718131237090.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("arabnizer_xlmr_panx_arabic_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("arabnizer_xlmr_panx_arabic_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|arabnizer_xlmr_panx_arabic_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|831.3 MB|

## References

https://huggingface.co/mohammedaly22/arabnizer-xlmr-panx-ar

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification