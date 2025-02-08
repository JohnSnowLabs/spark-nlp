---
layout: model
title: English twitter_roberta_fine_tuned_finance_sentiment_pipeline pipeline RoBertaForSequenceClassification from steve1989
author: John Snow Labs
name: twitter_roberta_fine_tuned_finance_sentiment_pipeline
date: 2025-02-05
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`twitter_roberta_fine_tuned_finance_sentiment_pipeline` is a English model originally trained by steve1989.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/twitter_roberta_fine_tuned_finance_sentiment_pipeline_en_5.5.1_3.0_1738799151509.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/twitter_roberta_fine_tuned_finance_sentiment_pipeline_en_5.5.1_3.0_1738799151509.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("twitter_roberta_fine_tuned_finance_sentiment_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("twitter_roberta_fine_tuned_finance_sentiment_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|twitter_roberta_fine_tuned_finance_sentiment_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|468.2 MB|

## References

https://huggingface.co/steve1989/twitter-roberta-fine-tuned-finance-sentiment

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification