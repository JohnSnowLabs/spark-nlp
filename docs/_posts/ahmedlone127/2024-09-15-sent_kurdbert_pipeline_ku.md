---
layout: model
title: Kurdish sent_kurdbert_pipeline pipeline BertSentenceEmbeddings from language-ml-lab
author: John Snow Labs
name: sent_kurdbert_pipeline
date: 2024-09-15
tags: [ku, open_source, pipeline, onnx]
task: Embeddings
language: ku
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_kurdbert_pipeline` is a Kurdish model originally trained by language-ml-lab.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_kurdbert_pipeline_ku_5.5.0_3.0_1726394815259.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_kurdbert_pipeline_ku_5.5.0_3.0_1726394815259.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_kurdbert_pipeline", lang = "ku")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_kurdbert_pipeline", lang = "ku")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_kurdbert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ku|
|Size:|407.3 MB|

## References

https://huggingface.co/language-ml-lab/KurdBert

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings