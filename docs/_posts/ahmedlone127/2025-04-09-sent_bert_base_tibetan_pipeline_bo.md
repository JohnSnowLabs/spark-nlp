---
layout: model
title: Tibetan sent_bert_base_tibetan_pipeline pipeline BertSentenceEmbeddings from KoichiYasuoka
author: John Snow Labs
name: sent_bert_base_tibetan_pipeline
date: 2025-04-09
tags: [bo, open_source, pipeline, onnx]
task: Embeddings
language: bo
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_bert_base_tibetan_pipeline` is a Tibetan model originally trained by KoichiYasuoka.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_base_tibetan_pipeline_bo_5.5.1_3.0_1744207195666.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_base_tibetan_pipeline_bo_5.5.1_3.0_1744207195666.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_bert_base_tibetan_pipeline", lang = "bo")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_bert_base_tibetan_pipeline", lang = "bo")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_base_tibetan_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|bo|
|Size:|406.4 MB|

## References

https://huggingface.co/KoichiYasuoka/bert-base-tibetan

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings