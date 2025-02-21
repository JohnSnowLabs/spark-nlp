---
layout: model
title: English finetuned_deberta_carer_4isear_4_pipeline pipeline DeBertaForSequenceClassification from UO282436
author: John Snow Labs
name: finetuned_deberta_carer_4isear_4_pipeline
date: 2025-01-26
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finetuned_deberta_carer_4isear_4_pipeline` is a English model originally trained by UO282436.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finetuned_deberta_carer_4isear_4_pipeline_en_5.5.1_3.0_1737917770918.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finetuned_deberta_carer_4isear_4_pipeline_en_5.5.1_3.0_1737917770918.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("finetuned_deberta_carer_4isear_4_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("finetuned_deberta_carer_4isear_4_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finetuned_deberta_carer_4isear_4_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|582.6 MB|

## References

https://huggingface.co/UO282436/finetuned-deberta-carer_4isear_4

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification