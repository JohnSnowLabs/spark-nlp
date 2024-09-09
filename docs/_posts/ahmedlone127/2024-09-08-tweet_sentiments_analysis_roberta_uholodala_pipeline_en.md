---
layout: model
title: English tweet_sentiments_analysis_roberta_uholodala_pipeline pipeline RoBertaForSequenceClassification from UholoDala
author: John Snow Labs
name: tweet_sentiments_analysis_roberta_uholodala_pipeline
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tweet_sentiments_analysis_roberta_uholodala_pipeline` is a English model originally trained by UholoDala.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tweet_sentiments_analysis_roberta_uholodala_pipeline_en_5.5.0_3.0_1725830568378.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tweet_sentiments_analysis_roberta_uholodala_pipeline_en_5.5.0_3.0_1725830568378.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tweet_sentiments_analysis_roberta_uholodala_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tweet_sentiments_analysis_roberta_uholodala_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tweet_sentiments_analysis_roberta_uholodala_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|436.4 MB|

## References

https://huggingface.co/UholoDala/tweet_sentiments_analysis_roberta

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification