---
layout: model
title: Swahili (macrolanguage) sent_bert_base_uncased_swahili_pipeline pipeline BertSentenceEmbeddings from flax-community
author: John Snow Labs
name: sent_bert_base_uncased_swahili_pipeline
date: 2024-09-11
tags: [sw, open_source, pipeline, onnx]
task: Embeddings
language: sw
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_bert_base_uncased_swahili_pipeline` is a Swahili (macrolanguage) model originally trained by flax-community.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_base_uncased_swahili_pipeline_sw_5.5.0_3.0_1726079683547.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_base_uncased_swahili_pipeline_sw_5.5.0_3.0_1726079683547.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_bert_base_uncased_swahili_pipeline", lang = "sw")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_bert_base_uncased_swahili_pipeline", lang = "sw")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_base_uncased_swahili_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|sw|
|Size:|408.6 MB|

## References

https://huggingface.co/flax-community/bert-base-uncased-swahili

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings