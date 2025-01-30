---
layout: model
title: English mbert_bengali_qa_squad_bengali_finetuned_pipeline pipeline BertForQuestionAnswering from AsifAbrar6
author: John Snow Labs
name: mbert_bengali_qa_squad_bengali_finetuned_pipeline
date: 2025-01-26
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mbert_bengali_qa_squad_bengali_finetuned_pipeline` is a English model originally trained by AsifAbrar6.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mbert_bengali_qa_squad_bengali_finetuned_pipeline_en_5.5.1_3.0_1737918761767.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mbert_bengali_qa_squad_bengali_finetuned_pipeline_en_5.5.1_3.0_1737918761767.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mbert_bengali_qa_squad_bengali_finetuned_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mbert_bengali_qa_squad_bengali_finetuned_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mbert_bengali_qa_squad_bengali_finetuned_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|625.5 MB|

## References

https://huggingface.co/AsifAbrar6/mbert-bengali-qa-squad_bn-finetuned

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering