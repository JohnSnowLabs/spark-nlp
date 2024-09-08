---
layout: model
title: Castilian, Spanish beto_finetuned_ner_3_pipeline pipeline BertForTokenClassification from ifis
author: John Snow Labs
name: beto_finetuned_ner_3_pipeline
date: 2024-09-04
tags: [es, open_source, pipeline, onnx]
task: Named Entity Recognition
language: es
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`beto_finetuned_ner_3_pipeline` is a Castilian, Spanish model originally trained by ifis.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/beto_finetuned_ner_3_pipeline_es_5.5.0_3.0_1725450384642.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/beto_finetuned_ner_3_pipeline_es_5.5.0_3.0_1725450384642.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("beto_finetuned_ner_3_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("beto_finetuned_ner_3_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|beto_finetuned_ner_3_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|409.6 MB|

## References

https://huggingface.co/ifis/BETO-finetuned-ner-3

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification