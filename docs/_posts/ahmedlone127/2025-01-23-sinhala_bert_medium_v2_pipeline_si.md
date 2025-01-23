---
layout: model
title: Sinhala, Sinhalese sinhala_bert_medium_v2_pipeline pipeline BertEmbeddings from Ransaka
author: John Snow Labs
name: sinhala_bert_medium_v2_pipeline
date: 2025-01-23
tags: [si, open_source, pipeline, onnx]
task: Embeddings
language: si
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sinhala_bert_medium_v2_pipeline` is a Sinhala, Sinhalese model originally trained by Ransaka.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sinhala_bert_medium_v2_pipeline_si_5.5.1_3.0_1737637257242.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sinhala_bert_medium_v2_pipeline_si_5.5.1_3.0_1737637257242.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sinhala_bert_medium_v2_pipeline", lang = "si")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sinhala_bert_medium_v2_pipeline", lang = "si")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sinhala_bert_medium_v2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|si|
|Size:|187.4 MB|

## References

https://huggingface.co/Ransaka/sinhala-bert-medium-v2

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings