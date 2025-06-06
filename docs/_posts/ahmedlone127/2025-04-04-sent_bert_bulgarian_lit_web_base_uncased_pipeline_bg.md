---
layout: model
title: Bulgarian sent_bert_bulgarian_lit_web_base_uncased_pipeline pipeline BertSentenceEmbeddings from AIaLT-IICT
author: John Snow Labs
name: sent_bert_bulgarian_lit_web_base_uncased_pipeline
date: 2025-04-04
tags: [bg, open_source, pipeline, onnx]
task: Embeddings
language: bg
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_bert_bulgarian_lit_web_base_uncased_pipeline` is a Bulgarian model originally trained by AIaLT-IICT.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_bulgarian_lit_web_base_uncased_pipeline_bg_5.5.1_3.0_1743790973051.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_bulgarian_lit_web_base_uncased_pipeline_bg_5.5.1_3.0_1743790973051.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_bert_bulgarian_lit_web_base_uncased_pipeline", lang = "bg")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_bert_bulgarian_lit_web_base_uncased_pipeline", lang = "bg")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_bulgarian_lit_web_base_uncased_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|bg|
|Size:|465.5 MB|

## References

https://huggingface.co/AIaLT-IICT/bert_bg_lit_web_base_uncased

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings