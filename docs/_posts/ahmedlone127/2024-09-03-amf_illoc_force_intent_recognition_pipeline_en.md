---
layout: model
title: English amf_illoc_force_intent_recognition_pipeline pipeline RoBertaForSequenceClassification from Godfrey2712
author: John Snow Labs
name: amf_illoc_force_intent_recognition_pipeline
date: 2024-09-03
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`amf_illoc_force_intent_recognition_pipeline` is a English model originally trained by Godfrey2712.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/amf_illoc_force_intent_recognition_pipeline_en_5.5.0_3.0_1725369117537.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/amf_illoc_force_intent_recognition_pipeline_en_5.5.0_3.0_1725369117537.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("amf_illoc_force_intent_recognition_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("amf_illoc_force_intent_recognition_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|amf_illoc_force_intent_recognition_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/Godfrey2712/amf_illoc_force_intent_recognition

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification