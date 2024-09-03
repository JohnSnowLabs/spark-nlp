---
layout: model
title: Urdu urdish_roberta_base_sentiment_pipeline pipeline RoBertaForSequenceClassification from dataranch
author: John Snow Labs
name: urdish_roberta_base_sentiment_pipeline
date: 2024-09-02
tags: [ur, open_source, pipeline, onnx]
task: Text Classification
language: ur
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`urdish_roberta_base_sentiment_pipeline` is a Urdu model originally trained by dataranch.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/urdish_roberta_base_sentiment_pipeline_ur_5.5.0_3.0_1725277898934.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/urdish_roberta_base_sentiment_pipeline_ur_5.5.0_3.0_1725277898934.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("urdish_roberta_base_sentiment_pipeline", lang = "ur")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("urdish_roberta_base_sentiment_pipeline", lang = "ur")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|urdish_roberta_base_sentiment_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ur|
|Size:|468.3 MB|

## References

https://huggingface.co/dataranch/urdish-roberta-base-sentiment

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification