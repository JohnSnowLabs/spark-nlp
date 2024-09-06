---
layout: model
title: Persian bert_token_classifier_parsbert_peymaner_pipeline pipeline BertForTokenClassification from HooshvareLab
author: John Snow Labs
name: bert_token_classifier_parsbert_peymaner_pipeline
date: 2024-09-05
tags: [fa, open_source, pipeline, onnx]
task: Named Entity Recognition
language: fa
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_token_classifier_parsbert_peymaner_pipeline` is a Persian model originally trained by HooshvareLab.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_parsbert_peymaner_pipeline_fa_5.5.0_3.0_1725563333293.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_parsbert_peymaner_pipeline_fa_5.5.0_3.0_1725563333293.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_token_classifier_parsbert_peymaner_pipeline", lang = "fa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_token_classifier_parsbert_peymaner_pipeline", lang = "fa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_parsbert_peymaner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fa|
|Size:|606.5 MB|

## References

https://huggingface.co/HooshvareLab/bert-base-parsbert-peymaner-uncased

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification