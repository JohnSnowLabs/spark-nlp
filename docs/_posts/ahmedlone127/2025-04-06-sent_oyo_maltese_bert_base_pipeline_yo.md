---
layout: model
title: Yoruba sent_oyo_maltese_bert_base_pipeline pipeline BertSentenceEmbeddings from Davlan
author: John Snow Labs
name: sent_oyo_maltese_bert_base_pipeline
date: 2025-04-06
tags: [yo, open_source, pipeline, onnx]
task: Embeddings
language: yo
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_oyo_maltese_bert_base_pipeline` is a Yoruba model originally trained by Davlan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_oyo_maltese_bert_base_pipeline_yo_5.5.1_3.0_1743912133833.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_oyo_maltese_bert_base_pipeline_yo_5.5.1_3.0_1743912133833.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_oyo_maltese_bert_base_pipeline", lang = "yo")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_oyo_maltese_bert_base_pipeline", lang = "yo")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_oyo_maltese_bert_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|yo|
|Size:|413.1 MB|

## References

https://huggingface.co/Davlan/oyo-mt-bert-base

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings