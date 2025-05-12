---
layout: model
title: English fb_bart_large_finetuned_trade_the_event_finance_summarizer_pipeline pipeline BartTransformer from nickmuchi
author: John Snow Labs
name: fb_bart_large_finetuned_trade_the_event_finance_summarizer_pipeline
date: 2025-04-03
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
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

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`fb_bart_large_finetuned_trade_the_event_finance_summarizer_pipeline` is a English model originally trained by nickmuchi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/fb_bart_large_finetuned_trade_the_event_finance_summarizer_pipeline_en_5.5.1_3.0_1743658332690.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/fb_bart_large_finetuned_trade_the_event_finance_summarizer_pipeline_en_5.5.1_3.0_1743658332690.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("fb_bart_large_finetuned_trade_the_event_finance_summarizer_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("fb_bart_large_finetuned_trade_the_event_finance_summarizer_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|fb_bart_large_finetuned_trade_the_event_finance_summarizer_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.9 GB|

## References

https://huggingface.co/nickmuchi/fb-bart-large-finetuned-trade-the-event-finance-summarizer

## Included Models

- DocumentAssembler
- BartTransformer