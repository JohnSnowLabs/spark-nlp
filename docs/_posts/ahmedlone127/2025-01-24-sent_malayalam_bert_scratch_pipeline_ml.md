---
layout: model
title: Malayalam sent_malayalam_bert_scratch_pipeline pipeline BertSentenceEmbeddings from l3cube-pune
author: John Snow Labs
name: sent_malayalam_bert_scratch_pipeline
date: 2025-01-24
tags: [ml, open_source, pipeline, onnx]
task: Embeddings
language: ml
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_malayalam_bert_scratch_pipeline` is a Malayalam model originally trained by l3cube-pune.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_malayalam_bert_scratch_pipeline_ml_5.5.1_3.0_1737689242434.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_malayalam_bert_scratch_pipeline_ml_5.5.1_3.0_1737689242434.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_malayalam_bert_scratch_pipeline", lang = "ml")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_malayalam_bert_scratch_pipeline", lang = "ml")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_malayalam_bert_scratch_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ml|
|Size:|471.3 MB|

## References

https://huggingface.co/l3cube-pune/malayalam-bert-scratch

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings