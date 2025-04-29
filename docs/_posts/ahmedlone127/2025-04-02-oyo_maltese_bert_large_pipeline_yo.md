---
layout: model
title: Yoruba oyo_maltese_bert_large_pipeline pipeline BertEmbeddings from Davlan
author: John Snow Labs
name: oyo_maltese_bert_large_pipeline
date: 2025-04-02
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

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`oyo_maltese_bert_large_pipeline` is a Yoruba model originally trained by Davlan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/oyo_maltese_bert_large_pipeline_yo_5.5.1_3.0_1743634763626.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/oyo_maltese_bert_large_pipeline_yo_5.5.1_3.0_1743634763626.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("oyo_maltese_bert_large_pipeline", lang = "yo")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("oyo_maltese_bert_large_pipeline", lang = "yo")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|oyo_maltese_bert_large_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|yo|
|Size:|1.3 GB|

## References

https://huggingface.co/Davlan/oyo-mt-bert-large

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings