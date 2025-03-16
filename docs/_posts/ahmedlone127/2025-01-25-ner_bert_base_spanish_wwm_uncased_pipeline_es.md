---
layout: model
title: Castilian, Spanish ner_bert_base_spanish_wwm_uncased_pipeline pipeline BertForTokenClassification from jpherrerap
author: John Snow Labs
name: ner_bert_base_spanish_wwm_uncased_pipeline
date: 2025-01-25
tags: [es, open_source, pipeline, onnx]
task: Named Entity Recognition
language: es
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ner_bert_base_spanish_wwm_uncased_pipeline` is a Castilian, Spanish model originally trained by jpherrerap.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_bert_base_spanish_wwm_uncased_pipeline_es_5.5.1_3.0_1737845217726.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_bert_base_spanish_wwm_uncased_pipeline_es_5.5.1_3.0_1737845217726.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ner_bert_base_spanish_wwm_uncased_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ner_bert_base_spanish_wwm_uncased_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_bert_base_spanish_wwm_uncased_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|409.7 MB|

## References

https://huggingface.co/jpherrerap/ner-bert-base-spanish-wwm-uncased

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification