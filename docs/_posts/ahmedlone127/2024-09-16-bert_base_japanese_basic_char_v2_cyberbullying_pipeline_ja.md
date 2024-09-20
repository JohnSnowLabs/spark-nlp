---
layout: model
title: Japanese bert_base_japanese_basic_char_v2_cyberbullying_pipeline pipeline BertForSequenceClassification from kit-nlp
author: John Snow Labs
name: bert_base_japanese_basic_char_v2_cyberbullying_pipeline
date: 2024-09-16
tags: [ja, open_source, pipeline, onnx]
task: Text Classification
language: ja
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_base_japanese_basic_char_v2_cyberbullying_pipeline` is a Japanese model originally trained by kit-nlp.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_japanese_basic_char_v2_cyberbullying_pipeline_ja_5.5.0_3.0_1726462638509.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_japanese_basic_char_v2_cyberbullying_pipeline_ja_5.5.0_3.0_1726462638509.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_base_japanese_basic_char_v2_cyberbullying_pipeline", lang = "ja")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_base_japanese_basic_char_v2_cyberbullying_pipeline", lang = "ja")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_japanese_basic_char_v2_cyberbullying_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ja|
|Size:|339.9 MB|

## References

https://huggingface.co/kit-nlp/bert-base-japanese-basic-char-v2-cyberbullying

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification