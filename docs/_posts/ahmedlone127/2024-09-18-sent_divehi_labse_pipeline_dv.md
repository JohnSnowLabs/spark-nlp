---
layout: model
title: Dhivehi, Divehi, Maldivian sent_divehi_labse_pipeline pipeline BertSentenceEmbeddings from monsoon-nlp
author: John Snow Labs
name: sent_divehi_labse_pipeline
date: 2024-09-18
tags: [dv, open_source, pipeline, onnx]
task: Embeddings
language: dv
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_divehi_labse_pipeline` is a Dhivehi, Divehi, Maldivian model originally trained by monsoon-nlp.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_divehi_labse_pipeline_dv_5.5.0_3.0_1726687422412.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_divehi_labse_pipeline_dv_5.5.0_3.0_1726687422412.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_divehi_labse_pipeline", lang = "dv")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_divehi_labse_pipeline", lang = "dv")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_divehi_labse_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|dv|
|Size:|1.9 GB|

## References

https://huggingface.co/monsoon-nlp/dv-labse

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings