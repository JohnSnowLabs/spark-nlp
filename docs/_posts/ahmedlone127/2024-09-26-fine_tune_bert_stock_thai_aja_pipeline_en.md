---
layout: model
title: English fine_tune_bert_stock_thai_aja_pipeline pipeline BertForSequenceClassification from jab11769
author: John Snow Labs
name: fine_tune_bert_stock_thai_aja_pipeline
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`fine_tune_bert_stock_thai_aja_pipeline` is a English model originally trained by jab11769.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/fine_tune_bert_stock_thai_aja_pipeline_en_5.5.0_3.0_1727338808855.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/fine_tune_bert_stock_thai_aja_pipeline_en_5.5.0_3.0_1727338808855.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("fine_tune_bert_stock_thai_aja_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("fine_tune_bert_stock_thai_aja_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|fine_tune_bert_stock_thai_aja_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|667.3 MB|

## References

https://huggingface.co/jab11769/fine-tune-bert-stock-thai-AJA

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification