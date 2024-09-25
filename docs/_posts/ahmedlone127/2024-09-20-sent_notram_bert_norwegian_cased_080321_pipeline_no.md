---
layout: model
title: Norwegian sent_notram_bert_norwegian_cased_080321_pipeline pipeline BertSentenceEmbeddings from NbAiLab
author: John Snow Labs
name: sent_notram_bert_norwegian_cased_080321_pipeline
date: 2024-09-20
tags: ["no", open_source, pipeline, onnx]
task: Embeddings
language: "no"
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_notram_bert_norwegian_cased_080321_pipeline` is a Norwegian model originally trained by NbAiLab.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_notram_bert_norwegian_cased_080321_pipeline_no_5.5.0_3.0_1726867078698.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_notram_bert_norwegian_cased_080321_pipeline_no_5.5.0_3.0_1726867078698.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_notram_bert_norwegian_cased_080321_pipeline", lang = "no")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_notram_bert_norwegian_cased_080321_pipeline", lang = "no")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_notram_bert_norwegian_cased_080321_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|no|
|Size:|663.6 MB|

## References

https://huggingface.co/NbAiLab/notram-bert-norwegian-cased-080321

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings