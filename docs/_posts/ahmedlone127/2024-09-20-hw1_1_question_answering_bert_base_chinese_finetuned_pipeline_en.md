---
layout: model
title: English hw1_1_question_answering_bert_base_chinese_finetuned_pipeline pipeline BertForQuestionAnswering from b10401015
author: John Snow Labs
name: hw1_1_question_answering_bert_base_chinese_finetuned_pipeline
date: 2024-09-20
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hw1_1_question_answering_bert_base_chinese_finetuned_pipeline` is a English model originally trained by b10401015.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hw1_1_question_answering_bert_base_chinese_finetuned_pipeline_en_5.5.0_3.0_1726833847120.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hw1_1_question_answering_bert_base_chinese_finetuned_pipeline_en_5.5.0_3.0_1726833847120.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hw1_1_question_answering_bert_base_chinese_finetuned_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hw1_1_question_answering_bert_base_chinese_finetuned_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hw1_1_question_answering_bert_base_chinese_finetuned_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|381.1 MB|

## References

https://huggingface.co/b10401015/hw1-1-question_answering-bert-base-chinese-finetuned

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering