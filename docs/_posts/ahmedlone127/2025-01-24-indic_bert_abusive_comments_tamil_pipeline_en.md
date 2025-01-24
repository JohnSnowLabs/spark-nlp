---
layout: model
title: English indic_bert_abusive_comments_tamil_pipeline pipeline AlbertForSequenceClassification from callmesan
author: John Snow Labs
name: indic_bert_abusive_comments_tamil_pipeline
date: 2025-01-24
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AlbertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`indic_bert_abusive_comments_tamil_pipeline` is a English model originally trained by callmesan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/indic_bert_abusive_comments_tamil_pipeline_en_5.5.1_3.0_1737693140004.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/indic_bert_abusive_comments_tamil_pipeline_en_5.5.1_3.0_1737693140004.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("indic_bert_abusive_comments_tamil_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("indic_bert_abusive_comments_tamil_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|indic_bert_abusive_comments_tamil_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|127.8 MB|

## References

https://huggingface.co/callmesan/indic-bert-abusive-comments-ta

## Included Models

- DocumentAssembler
- TokenizerModel
- AlbertForSequenceClassification