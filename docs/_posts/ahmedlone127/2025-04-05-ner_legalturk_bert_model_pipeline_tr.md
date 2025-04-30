---
layout: model
title: Turkish ner_legalturk_bert_model_pipeline pipeline BertForTokenClassification from farnazzeidi
author: John Snow Labs
name: ner_legalturk_bert_model_pipeline
date: 2025-04-05
tags: [tr, open_source, pipeline, onnx]
task: Named Entity Recognition
language: tr
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ner_legalturk_bert_model_pipeline` is a Turkish model originally trained by farnazzeidi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_legalturk_bert_model_pipeline_tr_5.5.1_3.0_1743825425031.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_legalturk_bert_model_pipeline_tr_5.5.1_3.0_1743825425031.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ner_legalturk_bert_model_pipeline", lang = "tr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ner_legalturk_bert_model_pipeline", lang = "tr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_legalturk_bert_model_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|tr|
|Size:|412.7 MB|

## References

https://huggingface.co/farnazzeidi/ner-legalturk-bert-model

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification