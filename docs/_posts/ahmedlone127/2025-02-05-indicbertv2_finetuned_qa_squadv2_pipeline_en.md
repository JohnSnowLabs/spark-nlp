---
layout: model
title: English indicbertv2_finetuned_qa_squadv2_pipeline pipeline BertForQuestionAnswering from Poulami
author: John Snow Labs
name: indicbertv2_finetuned_qa_squadv2_pipeline
date: 2025-02-05
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`indicbertv2_finetuned_qa_squadv2_pipeline` is a English model originally trained by Poulami.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/indicbertv2_finetuned_qa_squadv2_pipeline_en_5.5.1_3.0_1738788735989.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/indicbertv2_finetuned_qa_squadv2_pipeline_en_5.5.1_3.0_1738788735989.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("indicbertv2_finetuned_qa_squadv2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("indicbertv2_finetuned_qa_squadv2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|indicbertv2_finetuned_qa_squadv2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.0 GB|

## References

https://huggingface.co/Poulami/IndicBERTv2-finetuned-QA-SQuADv2

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering