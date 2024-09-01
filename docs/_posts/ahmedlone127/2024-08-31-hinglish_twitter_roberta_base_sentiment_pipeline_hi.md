---
layout: model
title: Hindi hinglish_twitter_roberta_base_sentiment_pipeline pipeline RoBertaForSequenceClassification from pascalrai
author: John Snow Labs
name: hinglish_twitter_roberta_base_sentiment_pipeline
date: 2024-08-31
tags: [hi, open_source, pipeline, onnx]
task: Text Classification
language: hi
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hinglish_twitter_roberta_base_sentiment_pipeline` is a Hindi model originally trained by pascalrai.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hinglish_twitter_roberta_base_sentiment_pipeline_hi_5.4.2_3.0_1725119324829.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hinglish_twitter_roberta_base_sentiment_pipeline_hi_5.4.2_3.0_1725119324829.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hinglish_twitter_roberta_base_sentiment_pipeline", lang = "hi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hinglish_twitter_roberta_base_sentiment_pipeline", lang = "hi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hinglish_twitter_roberta_base_sentiment_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|hi|
|Size:|468.3 MB|

## References

https://huggingface.co/pascalrai/hinglish-twitter-roberta-base-sentiment

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification