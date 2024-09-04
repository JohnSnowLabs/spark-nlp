---
layout: model
title: Hebrew sent_berel_dicta_il_pipeline pipeline BertSentenceEmbeddings from dicta-il
author: John Snow Labs
name: sent_berel_dicta_il_pipeline
date: 2024-09-04
tags: [he, open_source, pipeline, onnx]
task: Embeddings
language: he
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_berel_dicta_il_pipeline` is a Hebrew model originally trained by dicta-il.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_berel_dicta_il_pipeline_he_5.5.0_3.0_1725453930843.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_berel_dicta_il_pipeline_he_5.5.0_3.0_1725453930843.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_berel_dicta_il_pipeline", lang = "he")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_berel_dicta_il_pipeline", lang = "he")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_berel_dicta_il_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|he|
|Size:|690.7 MB|

## References

https://huggingface.co/dicta-il/BEREL

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings