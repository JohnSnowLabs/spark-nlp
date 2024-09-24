---
layout: model
title: Arabic marbertv2_finetuned_egyptian_hate_speech_detection_pipeline pipeline BertForSequenceClassification from IbrahimAmin
author: John Snow Labs
name: marbertv2_finetuned_egyptian_hate_speech_detection_pipeline
date: 2024-09-20
tags: [ar, open_source, pipeline, onnx]
task: Text Classification
language: ar
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`marbertv2_finetuned_egyptian_hate_speech_detection_pipeline` is a Arabic model originally trained by IbrahimAmin.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/marbertv2_finetuned_egyptian_hate_speech_detection_pipeline_ar_5.5.0_3.0_1726860461568.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/marbertv2_finetuned_egyptian_hate_speech_detection_pipeline_ar_5.5.0_3.0_1726860461568.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("marbertv2_finetuned_egyptian_hate_speech_detection_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("marbertv2_finetuned_egyptian_hate_speech_detection_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|marbertv2_finetuned_egyptian_hate_speech_detection_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|608.8 MB|

## References

https://huggingface.co/IbrahimAmin/marbertv2-finetuned-egyptian-hate-speech-detection

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification