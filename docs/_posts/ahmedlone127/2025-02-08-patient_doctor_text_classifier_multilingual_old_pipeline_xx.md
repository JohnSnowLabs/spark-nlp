---
layout: model
title: Multilingual patient_doctor_text_classifier_multilingual_old_pipeline pipeline DistilBertForSequenceClassification from LukeGPT88
author: John Snow Labs
name: patient_doctor_text_classifier_multilingual_old_pipeline
date: 2025-02-08
tags: [xx, open_source, pipeline, onnx]
task: Text Classification
language: xx
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`patient_doctor_text_classifier_multilingual_old_pipeline` is a Multilingual model originally trained by LukeGPT88.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/patient_doctor_text_classifier_multilingual_old_pipeline_xx_5.5.1_3.0_1739010929592.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/patient_doctor_text_classifier_multilingual_old_pipeline_xx_5.5.1_3.0_1739010929592.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("patient_doctor_text_classifier_multilingual_old_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("patient_doctor_text_classifier_multilingual_old_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|patient_doctor_text_classifier_multilingual_old_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|507.6 MB|

## References

https://huggingface.co/LukeGPT88/patient-doctor-text-classifier-multilingual-old

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification