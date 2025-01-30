---
layout: model
title: English danish_mrm8488_distilroberta_finetuned_financial_news_sentiment_analysis_pipeline pipeline RoBertaEmbeddings from rnribeiro
author: John Snow Labs
name: danish_mrm8488_distilroberta_finetuned_financial_news_sentiment_analysis_pipeline
date: 2025-01-26
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`danish_mrm8488_distilroberta_finetuned_financial_news_sentiment_analysis_pipeline` is a English model originally trained by rnribeiro.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/danish_mrm8488_distilroberta_finetuned_financial_news_sentiment_analysis_pipeline_en_5.5.1_3.0_1737865792833.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/danish_mrm8488_distilroberta_finetuned_financial_news_sentiment_analysis_pipeline_en_5.5.1_3.0_1737865792833.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("danish_mrm8488_distilroberta_finetuned_financial_news_sentiment_analysis_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("danish_mrm8488_distilroberta_finetuned_financial_news_sentiment_analysis_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|danish_mrm8488_distilroberta_finetuned_financial_news_sentiment_analysis_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|306.8 MB|

## References

https://huggingface.co/rnribeiro/DA-mrm8488-distilroberta-finetuned-financial-news-sentiment-analysis

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings