---
layout: model
title: English clarify_the_city_bert_first256_pipeline pipeline BertForSequenceClassification from etadevosyan
author: John Snow Labs
name: clarify_the_city_bert_first256_pipeline
date: 2024-09-08
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`clarify_the_city_bert_first256_pipeline` is a English model originally trained by etadevosyan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/clarify_the_city_bert_first256_pipeline_en_5.5.0_3.0_1725819914165.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/clarify_the_city_bert_first256_pipeline_en_5.5.0_3.0_1725819914165.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("clarify_the_city_bert_first256_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("clarify_the_city_bert_first256_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clarify_the_city_bert_first256_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|666.5 MB|

## References

https://huggingface.co/etadevosyan/clarify_the_city_bert_First256

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification