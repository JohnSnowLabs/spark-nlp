---
layout: model
title: English caribbean_english_sentiment_fine_tuned_bert_mrarish320_pipeline pipeline BertForSequenceClassification from mrarish320
author: John Snow Labs
name: caribbean_english_sentiment_fine_tuned_bert_mrarish320_pipeline
date: 2024-12-19
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`caribbean_english_sentiment_fine_tuned_bert_mrarish320_pipeline` is a English model originally trained by mrarish320.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/caribbean_english_sentiment_fine_tuned_bert_mrarish320_pipeline_en_5.5.1_3.0_1734572163715.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/caribbean_english_sentiment_fine_tuned_bert_mrarish320_pipeline_en_5.5.1_3.0_1734572163715.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("caribbean_english_sentiment_fine_tuned_bert_mrarish320_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("caribbean_english_sentiment_fine_tuned_bert_mrarish320_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|caribbean_english_sentiment_fine_tuned_bert_mrarish320_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|409.4 MB|

## References

https://huggingface.co/mrarish320/caribbean_english_sentiment_fine_tuned_bert

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification