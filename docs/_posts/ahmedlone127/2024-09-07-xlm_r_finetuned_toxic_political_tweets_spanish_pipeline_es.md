---
layout: model
title: Castilian, Spanish xlm_r_finetuned_toxic_political_tweets_spanish_pipeline pipeline XlmRoBertaForSequenceClassification from Newtral
author: John Snow Labs
name: xlm_r_finetuned_toxic_political_tweets_spanish_pipeline
date: 2024-09-07
tags: [es, open_source, pipeline, onnx]
task: Text Classification
language: es
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlm_r_finetuned_toxic_political_tweets_spanish_pipeline` is a Castilian, Spanish model originally trained by Newtral.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_r_finetuned_toxic_political_tweets_spanish_pipeline_es_5.5.0_3.0_1725712481735.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_r_finetuned_toxic_political_tweets_spanish_pipeline_es_5.5.0_3.0_1725712481735.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlm_r_finetuned_toxic_political_tweets_spanish_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlm_r_finetuned_toxic_political_tweets_spanish_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_r_finetuned_toxic_political_tweets_spanish_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|801.4 MB|

## References

https://huggingface.co/Newtral/xlm-r-finetuned-toxic-political-tweets-es

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification