---
layout: model
title: Malay (macrolanguage) malay_sentiment_deberta_xsmall_pipeline pipeline DeBertaForSequenceClassification from malaysia-ai
author: John Snow Labs
name: malay_sentiment_deberta_xsmall_pipeline
date: 2024-09-06
tags: [ms, open_source, pipeline, onnx]
task: Text Classification
language: ms
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`malay_sentiment_deberta_xsmall_pipeline` is a Malay (macrolanguage) model originally trained by malaysia-ai.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/malay_sentiment_deberta_xsmall_pipeline_ms_5.5.0_3.0_1725609888005.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/malay_sentiment_deberta_xsmall_pipeline_ms_5.5.0_3.0_1725609888005.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("malay_sentiment_deberta_xsmall_pipeline", lang = "ms")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("malay_sentiment_deberta_xsmall_pipeline", lang = "ms")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|malay_sentiment_deberta_xsmall_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ms|
|Size:|240.4 MB|

## References

https://huggingface.co/malaysia-ai/malay-sentiment-deberta-xsmall

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification