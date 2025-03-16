---
layout: model
title: English distil_bert_ft_qa_model_7up_v6_pipeline pipeline BertForQuestionAnswering from cadzchua
author: John Snow Labs
name: distil_bert_ft_qa_model_7up_v6_pipeline
date: 2025-01-24
tags: [en, open_source, pipeline, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distil_bert_ft_qa_model_7up_v6_pipeline` is a English model originally trained by cadzchua.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distil_bert_ft_qa_model_7up_v6_pipeline_en_5.5.1_3.0_1737691363588.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distil_bert_ft_qa_model_7up_v6_pipeline_en_5.5.1_3.0_1737691363588.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distil_bert_ft_qa_model_7up_v6_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distil_bert_ft_qa_model_7up_v6_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distil_bert_ft_qa_model_7up_v6_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.2 MB|

## References

https://huggingface.co/cadzchua/distil-bert-ft-qa-model-7up-v6

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering