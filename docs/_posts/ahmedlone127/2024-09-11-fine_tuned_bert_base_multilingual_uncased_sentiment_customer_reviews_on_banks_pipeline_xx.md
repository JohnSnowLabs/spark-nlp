---
layout: model
title: Multilingual fine_tuned_bert_base_multilingual_uncased_sentiment_customer_reviews_on_banks_pipeline pipeline BertForSequenceClassification from Wilsen
author: John Snow Labs
name: fine_tuned_bert_base_multilingual_uncased_sentiment_customer_reviews_on_banks_pipeline
date: 2024-09-11
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`fine_tuned_bert_base_multilingual_uncased_sentiment_customer_reviews_on_banks_pipeline` is a Multilingual model originally trained by Wilsen.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/fine_tuned_bert_base_multilingual_uncased_sentiment_customer_reviews_on_banks_pipeline_xx_5.5.0_3.0_1726059611574.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/fine_tuned_bert_base_multilingual_uncased_sentiment_customer_reviews_on_banks_pipeline_xx_5.5.0_3.0_1726059611574.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("fine_tuned_bert_base_multilingual_uncased_sentiment_customer_reviews_on_banks_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("fine_tuned_bert_base_multilingual_uncased_sentiment_customer_reviews_on_banks_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|fine_tuned_bert_base_multilingual_uncased_sentiment_customer_reviews_on_banks_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|627.8 MB|

## References

https://huggingface.co/Wilsen/fine-tuned-bert-base-multilingual-uncased-sentiment-customer-reviews-on-banks

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification