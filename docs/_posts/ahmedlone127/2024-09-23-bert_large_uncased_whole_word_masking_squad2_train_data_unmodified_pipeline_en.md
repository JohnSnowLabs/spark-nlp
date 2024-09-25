---
layout: model
title: English bert_large_uncased_whole_word_masking_squad2_train_data_unmodified_pipeline pipeline BertForQuestionAnswering from mdzrg
author: John Snow Labs
name: bert_large_uncased_whole_word_masking_squad2_train_data_unmodified_pipeline
date: 2024-09-23
tags: [en, open_source, pipeline, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_large_uncased_whole_word_masking_squad2_train_data_unmodified_pipeline` is a English model originally trained by mdzrg.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_large_uncased_whole_word_masking_squad2_train_data_unmodified_pipeline_en_5.5.0_3.0_1727128703878.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_large_uncased_whole_word_masking_squad2_train_data_unmodified_pipeline_en_5.5.0_3.0_1727128703878.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_large_uncased_whole_word_masking_squad2_train_data_unmodified_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_large_uncased_whole_word_masking_squad2_train_data_unmodified_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_large_uncased_whole_word_masking_squad2_train_data_unmodified_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/mdzrg/bert-large-uncased-whole-word-masking-squad2-train-data-unmodified

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering