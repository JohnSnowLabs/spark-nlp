---
layout: model
title: Icelandic icelandic_ner_bert_pipeline pipeline BertForTokenClassification from grammatek
author: John Snow Labs
name: icelandic_ner_bert_pipeline
date: 2025-02-05
tags: [is, open_source, pipeline, onnx]
task: Named Entity Recognition
language: is
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`icelandic_ner_bert_pipeline` is a Icelandic model originally trained by grammatek.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/icelandic_ner_bert_pipeline_is_5.5.1_3.0_1738741512731.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/icelandic_ner_bert_pipeline_is_5.5.1_3.0_1738741512731.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("icelandic_ner_bert_pipeline", lang = "is")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("icelandic_ner_bert_pipeline", lang = "is")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|icelandic_ner_bert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|is|
|Size:|665.1 MB|

## References

https://huggingface.co/grammatek/icelandic-ner-bert

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification