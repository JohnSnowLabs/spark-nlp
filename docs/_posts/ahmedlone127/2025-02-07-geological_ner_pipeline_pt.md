---
layout: model
title: Portuguese geological_ner_pipeline pipeline BertForTokenClassification from vabatista
author: John Snow Labs
name: geological_ner_pipeline
date: 2025-02-07
tags: [pt, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`geological_ner_pipeline` is a Portuguese model originally trained by vabatista.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/geological_ner_pipeline_pt_5.5.1_3.0_1738961173169.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/geological_ner_pipeline_pt_5.5.1_3.0_1738961173169.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("geological_ner_pipeline", lang = "pt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("geological_ner_pipeline", lang = "pt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|geological_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|pt|
|Size:|406.1 MB|

## References

https://huggingface.co/vabatista/geological-ner

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification