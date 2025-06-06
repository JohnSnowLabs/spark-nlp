---
layout: model
title: Arabic sent_bert_base_qarib60_860k_pipeline pipeline BertSentenceEmbeddings from qarib
author: John Snow Labs
name: sent_bert_base_qarib60_860k_pipeline
date: 2025-03-31
tags: [ar, open_source, pipeline, onnx]
task: Embeddings
language: ar
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_bert_base_qarib60_860k_pipeline` is a Arabic model originally trained by qarib.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_base_qarib60_860k_pipeline_ar_5.5.1_3.0_1743396466295.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_base_qarib60_860k_pipeline_ar_5.5.1_3.0_1743396466295.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("sent_bert_base_qarib60_860k_pipeline", lang = "ar")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("sent_bert_base_qarib60_860k_pipeline", lang = "ar")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_base_qarib60_860k_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|505.5 MB|

## References

References

https://huggingface.co/qarib/bert-base-qarib60_860k

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings