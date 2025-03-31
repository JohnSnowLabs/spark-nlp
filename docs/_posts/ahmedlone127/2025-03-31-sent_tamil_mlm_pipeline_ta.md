---
layout: model
title: Tamil sent_tamil_mlm_pipeline pipeline BertSentenceEmbeddings from viswadarshan06
author: John Snow Labs
name: sent_tamil_mlm_pipeline
date: 2025-03-31
tags: [ta, open_source, pipeline, onnx]
task: Embeddings
language: ta
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_tamil_mlm_pipeline` is a Tamil model originally trained by viswadarshan06.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_tamil_mlm_pipeline_ta_5.5.1_3.0_1743422318348.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_tamil_mlm_pipeline_ta_5.5.1_3.0_1743422318348.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_tamil_mlm_pipeline", lang = "ta")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_tamil_mlm_pipeline", lang = "ta")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_tamil_mlm_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ta|
|Size:|1.0 GB|

## References

https://huggingface.co/viswadarshan06/Tamil-MLM

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings