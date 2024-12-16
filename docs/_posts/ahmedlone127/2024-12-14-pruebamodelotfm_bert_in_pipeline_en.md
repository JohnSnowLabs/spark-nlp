---
layout: model
title: English pruebamodelotfm_bert_in_pipeline pipeline BertForQuestionAnswering from pamelapaolacb
author: John Snow Labs
name: pruebamodelotfm_bert_in_pipeline
date: 2024-12-14
tags: [en, open_source, pipeline, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`pruebamodelotfm_bert_in_pipeline` is a English model originally trained by pamelapaolacb.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pruebamodelotfm_bert_in_pipeline_en_5.5.1_3.0_1734216377089.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pruebamodelotfm_bert_in_pipeline_en_5.5.1_3.0_1734216377089.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("pruebamodelotfm_bert_in_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("pruebamodelotfm_bert_in_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pruebamodelotfm_bert_in_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|403.7 MB|

## References

https://huggingface.co/pamelapaolacb/pruebaModeloTFM_Bert_in

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering