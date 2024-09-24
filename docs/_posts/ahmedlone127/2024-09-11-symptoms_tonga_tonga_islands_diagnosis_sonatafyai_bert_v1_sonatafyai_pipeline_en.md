---
layout: model
title: English symptoms_tonga_tonga_islands_diagnosis_sonatafyai_bert_v1_sonatafyai_pipeline pipeline BertForSequenceClassification from Sonatafyai
author: John Snow Labs
name: symptoms_tonga_tonga_islands_diagnosis_sonatafyai_bert_v1_sonatafyai_pipeline
date: 2024-09-11
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`symptoms_tonga_tonga_islands_diagnosis_sonatafyai_bert_v1_sonatafyai_pipeline` is a English model originally trained by Sonatafyai.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/symptoms_tonga_tonga_islands_diagnosis_sonatafyai_bert_v1_sonatafyai_pipeline_en_5.5.0_3.0_1726015182769.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/symptoms_tonga_tonga_islands_diagnosis_sonatafyai_bert_v1_sonatafyai_pipeline_en_5.5.0_3.0_1726015182769.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("symptoms_tonga_tonga_islands_diagnosis_sonatafyai_bert_v1_sonatafyai_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("symptoms_tonga_tonga_islands_diagnosis_sonatafyai_bert_v1_sonatafyai_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|symptoms_tonga_tonga_islands_diagnosis_sonatafyai_bert_v1_sonatafyai_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|409.5 MB|

## References

https://huggingface.co/Sonatafyai/Symptoms_to_Diagnosis_SonatafyAI_BERT_v1

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification