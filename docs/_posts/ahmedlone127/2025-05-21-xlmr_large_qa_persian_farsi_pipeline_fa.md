---
layout: model
title: Persian xlmr_large_qa_persian_farsi_pipeline pipeline XlmRoBertaForQuestionAnswering from m3hrdadfi
author: John Snow Labs
name: xlmr_large_qa_persian_farsi_pipeline
date: 2025-05-21
tags: [fa, open_source, pipeline, onnx]
task: Question Answering
language: fa
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlmr_large_qa_persian_farsi_pipeline` is a Persian model originally trained by m3hrdadfi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmr_large_qa_persian_farsi_pipeline_fa_5.5.1_3.0_1747822699111.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmr_large_qa_persian_farsi_pipeline_fa_5.5.1_3.0_1747822699111.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlmr_large_qa_persian_farsi_pipeline", lang = "fa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlmr_large_qa_persian_farsi_pipeline", lang = "fa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmr_large_qa_persian_farsi_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|fa|
|Size:|1.9 GB|

## References

https://huggingface.co/m3hrdadfi/xlmr-large-qa-fa

## Included Models

- MultiDocumentAssembler
- XlmRoBertaForQuestionAnswering