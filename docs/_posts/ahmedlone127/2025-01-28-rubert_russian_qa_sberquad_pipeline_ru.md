---
layout: model
title: Russian rubert_russian_qa_sberquad_pipeline pipeline BertForQuestionAnswering from milyausha2801
author: John Snow Labs
name: rubert_russian_qa_sberquad_pipeline
date: 2025-01-28
tags: [ru, open_source, pipeline, onnx]
task: Question Answering
language: ru
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`rubert_russian_qa_sberquad_pipeline` is a Russian model originally trained by milyausha2801.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/rubert_russian_qa_sberquad_pipeline_ru_5.5.1_3.0_1738062076380.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/rubert_russian_qa_sberquad_pipeline_ru_5.5.1_3.0_1738062076380.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("rubert_russian_qa_sberquad_pipeline", lang = "ru")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("rubert_russian_qa_sberquad_pipeline", lang = "ru")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|rubert_russian_qa_sberquad_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|664.3 MB|

## References

References

https://huggingface.co/milyausha2801/rubert-russian-qa-sberquad

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering