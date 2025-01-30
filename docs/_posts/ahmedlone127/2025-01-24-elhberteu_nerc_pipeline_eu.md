---
layout: model
title: Basque elhberteu_nerc_pipeline pipeline BertForTokenClassification from orai-nlp
author: John Snow Labs
name: elhberteu_nerc_pipeline
date: 2025-01-24
tags: [eu, open_source, pipeline, onnx]
task: Named Entity Recognition
language: eu
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`elhberteu_nerc_pipeline` is a Basque model originally trained by orai-nlp.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/elhberteu_nerc_pipeline_eu_5.5.1_3.0_1737720159044.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/elhberteu_nerc_pipeline_eu_5.5.1_3.0_1737720159044.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("elhberteu_nerc_pipeline", lang = "eu")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("elhberteu_nerc_pipeline", lang = "eu")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|elhberteu_nerc_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|eu|
|Size:|464.7 MB|

## References

https://huggingface.co/orai-nlp/ElhBERTeu-nerc

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification