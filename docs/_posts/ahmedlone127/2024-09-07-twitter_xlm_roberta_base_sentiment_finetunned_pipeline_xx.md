---
layout: model
title: Multilingual twitter_xlm_roberta_base_sentiment_finetunned_pipeline pipeline XlmRoBertaForSequenceClassification from citizenlab
author: John Snow Labs
name: twitter_xlm_roberta_base_sentiment_finetunned_pipeline
date: 2024-09-07
tags: [xx, open_source, pipeline, onnx]
task: Text Classification
language: xx
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`twitter_xlm_roberta_base_sentiment_finetunned_pipeline` is a Multilingual model originally trained by citizenlab.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/twitter_xlm_roberta_base_sentiment_finetunned_pipeline_xx_5.5.0_3.0_1725712209001.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/twitter_xlm_roberta_base_sentiment_finetunned_pipeline_xx_5.5.0_3.0_1725712209001.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("twitter_xlm_roberta_base_sentiment_finetunned_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("twitter_xlm_roberta_base_sentiment_finetunned_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|twitter_xlm_roberta_base_sentiment_finetunned_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|1.0 GB|

## References

https://huggingface.co/citizenlab/twitter-xlm-roberta-base-sentiment-finetunned

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification