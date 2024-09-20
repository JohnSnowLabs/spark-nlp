---
layout: model
title: English w266_model4_bert_automodelforsequenceclassification_pipeline pipeline BertForSequenceClassification from arindamatcalgm
author: John Snow Labs
name: w266_model4_bert_automodelforsequenceclassification_pipeline
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`w266_model4_bert_automodelforsequenceclassification_pipeline` is a English model originally trained by arindamatcalgm.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/w266_model4_bert_automodelforsequenceclassification_pipeline_en_5.5.0_3.0_1725856810265.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/w266_model4_bert_automodelforsequenceclassification_pipeline_en_5.5.0_3.0_1725856810265.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("w266_model4_bert_automodelforsequenceclassification_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("w266_model4_bert_automodelforsequenceclassification_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|w266_model4_bert_automodelforsequenceclassification_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|409.4 MB|

## References

https://huggingface.co/arindamatcalgm/w266_model4_BERT_AutoModelForSequenceClassification

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification