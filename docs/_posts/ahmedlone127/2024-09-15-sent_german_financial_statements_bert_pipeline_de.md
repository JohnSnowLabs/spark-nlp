---
layout: model
title: German sent_german_financial_statements_bert_pipeline pipeline BertSentenceEmbeddings from fabianrausch
author: John Snow Labs
name: sent_german_financial_statements_bert_pipeline
date: 2024-09-15
tags: [de, open_source, pipeline, onnx]
task: Embeddings
language: de
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_german_financial_statements_bert_pipeline` is a German model originally trained by fabianrausch.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_german_financial_statements_bert_pipeline_de_5.5.0_3.0_1726436602603.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_german_financial_statements_bert_pipeline_de_5.5.0_3.0_1726436602603.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_german_financial_statements_bert_pipeline", lang = "de")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_german_financial_statements_bert_pipeline", lang = "de")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_german_financial_statements_bert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|de|
|Size:|407.5 MB|

## References

https://huggingface.co/fabianrausch/german-financial-statements-bert

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings