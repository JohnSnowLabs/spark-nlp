---
layout: model
title: English vaccine_tweet_sentiments_analysis_model_2_pipeline pipeline RoBertaForSequenceClassification from Ausbel
author: John Snow Labs
name: vaccine_tweet_sentiments_analysis_model_2_pipeline
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`vaccine_tweet_sentiments_analysis_model_2_pipeline` is a English model originally trained by Ausbel.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/vaccine_tweet_sentiments_analysis_model_2_pipeline_en_5.5.0_3.0_1725829681996.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/vaccine_tweet_sentiments_analysis_model_2_pipeline_en_5.5.0_3.0_1725829681996.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("vaccine_tweet_sentiments_analysis_model_2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("vaccine_tweet_sentiments_analysis_model_2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|vaccine_tweet_sentiments_analysis_model_2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|468.2 MB|

## References

https://huggingface.co/Ausbel/Vaccine-tweet-sentiments-analysis-model-2

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification