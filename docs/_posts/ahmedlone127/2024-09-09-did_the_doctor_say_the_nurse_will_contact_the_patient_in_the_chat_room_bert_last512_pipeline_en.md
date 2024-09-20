---
layout: model
title: English did_the_doctor_say_the_nurse_will_contact_the_patient_in_the_chat_room_bert_last512_pipeline pipeline BertForSequenceClassification from etadevosyan
author: John Snow Labs
name: did_the_doctor_say_the_nurse_will_contact_the_patient_in_the_chat_room_bert_last512_pipeline
date: 2024-09-09
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`did_the_doctor_say_the_nurse_will_contact_the_patient_in_the_chat_room_bert_last512_pipeline` is a English model originally trained by etadevosyan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/did_the_doctor_say_the_nurse_will_contact_the_patient_in_the_chat_room_bert_last512_pipeline_en_5.5.0_3.0_1725900998719.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/did_the_doctor_say_the_nurse_will_contact_the_patient_in_the_chat_room_bert_last512_pipeline_en_5.5.0_3.0_1725900998719.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("did_the_doctor_say_the_nurse_will_contact_the_patient_in_the_chat_room_bert_last512_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("did_the_doctor_say_the_nurse_will_contact_the_patient_in_the_chat_room_bert_last512_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|did_the_doctor_say_the_nurse_will_contact_the_patient_in_the_chat_room_bert_last512_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|666.5 MB|

## References

https://huggingface.co/etadevosyan/did_the_doctor_say_the_nurse_will_contact_the_patient_in_the_chat_room_bert_Last512

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification