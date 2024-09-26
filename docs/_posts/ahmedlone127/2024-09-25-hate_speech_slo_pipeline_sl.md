---
layout: model
title: Slovenian hate_speech_slo_pipeline pipeline BertForSequenceClassification from IMSyPP
author: John Snow Labs
name: hate_speech_slo_pipeline
date: 2024-09-25
tags: [sl, open_source, pipeline, onnx]
task: Text Classification
language: sl
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hate_speech_slo_pipeline` is a Slovenian model originally trained by IMSyPP.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hate_speech_slo_pipeline_sl_5.5.0_3.0_1727245744831.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hate_speech_slo_pipeline_sl_5.5.0_3.0_1727245744831.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hate_speech_slo_pipeline", lang = "sl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hate_speech_slo_pipeline", lang = "sl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hate_speech_slo_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|sl|
|Size:|465.7 MB|

## References

https://huggingface.co/IMSyPP/hate_speech_slo

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification