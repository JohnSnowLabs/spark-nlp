---
layout: model
title: Multilingual xlm_roberta_large_qa_multilingual_finedtuned_russian_pipeline pipeline XlmRoBertaForQuestionAnswering from AlexKay
author: John Snow Labs
name: xlm_roberta_large_qa_multilingual_finedtuned_russian_pipeline
date: 2025-05-21
tags: [xx, open_source, pipeline, onnx]
task: Question Answering
language: xx
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlm_roberta_large_qa_multilingual_finedtuned_russian_pipeline` is a Multilingual model originally trained by AlexKay.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_large_qa_multilingual_finedtuned_russian_pipeline_xx_5.5.1_3.0_1747822894293.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_large_qa_multilingual_finedtuned_russian_pipeline_xx_5.5.1_3.0_1747822894293.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlm_roberta_large_qa_multilingual_finedtuned_russian_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlm_roberta_large_qa_multilingual_finedtuned_russian_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_large_qa_multilingual_finedtuned_russian_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|1.9 GB|

## References

https://huggingface.co/AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru

## Included Models

- MultiDocumentAssembler
- XlmRoBertaForQuestionAnswering