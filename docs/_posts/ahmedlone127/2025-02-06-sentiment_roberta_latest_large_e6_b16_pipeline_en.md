---
layout: model
title: English sentiment_roberta_latest_large_e6_b16_pipeline pipeline RoBertaForSequenceClassification from YanJiangJerry
author: John Snow Labs
name: sentiment_roberta_latest_large_e6_b16_pipeline
date: 2025-02-06
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sentiment_roberta_latest_large_e6_b16_pipeline` is a English model originally trained by YanJiangJerry.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentiment_roberta_latest_large_e6_b16_pipeline_en_5.5.1_3.0_1738800111261.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentiment_roberta_latest_large_e6_b16_pipeline_en_5.5.1_3.0_1738800111261.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sentiment_roberta_latest_large_e6_b16_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sentiment_roberta_latest_large_e6_b16_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentiment_roberta_latest_large_e6_b16_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|468.3 MB|

## References

https://huggingface.co/YanJiangJerry/sentiment-roberta-latest-large-e6-b16

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification