---
layout: model
title: Portuguese berna_bert_fato_tese_pipeline pipeline BertForSequenceClassification from DIACDE
author: John Snow Labs
name: berna_bert_fato_tese_pipeline
date: 2025-01-29
tags: [pt, open_source, pipeline, onnx]
task: Text Classification
language: pt
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`berna_bert_fato_tese_pipeline` is a Portuguese model originally trained by DIACDE.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/berna_bert_fato_tese_pipeline_pt_5.5.1_3.0_1738145303136.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/berna_bert_fato_tese_pipeline_pt_5.5.1_3.0_1738145303136.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("berna_bert_fato_tese_pipeline", lang = "pt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("berna_bert_fato_tese_pipeline", lang = "pt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|berna_bert_fato_tese_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|pt|
|Size:|408.2 MB|

## References

https://huggingface.co/DIACDE/BERNA_BERT_FATO_TESE

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification