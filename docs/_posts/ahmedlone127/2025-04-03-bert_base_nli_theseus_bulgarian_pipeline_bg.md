---
layout: model
title: Bulgarian bert_base_nli_theseus_bulgarian_pipeline pipeline BertForSequenceClassification from rmihaylov
author: John Snow Labs
name: bert_base_nli_theseus_bulgarian_pipeline
date: 2025-04-03
tags: [bg, open_source, pipeline, onnx]
task: Text Classification
language: bg
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_base_nli_theseus_bulgarian_pipeline` is a Bulgarian model originally trained by rmihaylov.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_nli_theseus_bulgarian_pipeline_bg_5.5.1_3.0_1743641973898.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_nli_theseus_bulgarian_pipeline_bg_5.5.1_3.0_1743641973898.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("bert_base_nli_theseus_bulgarian_pipeline", lang = "bg")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("bert_base_nli_theseus_bulgarian_pipeline", lang = "bg")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_nli_theseus_bulgarian_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|bg|
|Size:|507.9 MB|

## References

References

https://huggingface.co/rmihaylov/bert-base-nli-theseus-bg

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification