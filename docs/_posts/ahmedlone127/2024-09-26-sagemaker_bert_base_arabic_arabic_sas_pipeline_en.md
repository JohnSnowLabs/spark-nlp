---
layout: model
title: English sagemaker_bert_base_arabic_arabic_sas_pipeline pipeline BertForSequenceClassification from Osaleh
author: John Snow Labs
name: sagemaker_bert_base_arabic_arabic_sas_pipeline
date: 2024-09-26
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sagemaker_bert_base_arabic_arabic_sas_pipeline` is a English model originally trained by Osaleh.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sagemaker_bert_base_arabic_arabic_sas_pipeline_en_5.5.0_3.0_1727358999355.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sagemaker_bert_base_arabic_arabic_sas_pipeline_en_5.5.0_3.0_1727358999355.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sagemaker_bert_base_arabic_arabic_sas_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sagemaker_bert_base_arabic_arabic_sas_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sagemaker_bert_base_arabic_arabic_sas_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|414.2 MB|

## References

https://huggingface.co/Osaleh/sagemaker-bert-base-arabic-ar-SAS

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification