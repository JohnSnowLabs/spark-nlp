---
layout: model
title: English xlmr_enis_qa_icelandic_finetune_hindi_course_pipeline pipeline XlmRoBertaForQuestionAnswering from nozagleh
author: John Snow Labs
name: xlmr_enis_qa_icelandic_finetune_hindi_course_pipeline
date: 2024-09-06
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

Pretrained XlmRoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlmr_enis_qa_icelandic_finetune_hindi_course_pipeline` is a English model originally trained by nozagleh.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmr_enis_qa_icelandic_finetune_hindi_course_pipeline_en_5.5.0_3.0_1725598546946.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmr_enis_qa_icelandic_finetune_hindi_course_pipeline_en_5.5.0_3.0_1725598546946.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlmr_enis_qa_icelandic_finetune_hindi_course_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlmr_enis_qa_icelandic_finetune_hindi_course_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmr_enis_qa_icelandic_finetune_hindi_course_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|456.7 MB|

## References

https://huggingface.co/nozagleh/XLMr-ENIS-QA-Is-finetune-hi-course

## Included Models

- MultiDocumentAssembler
- XlmRoBertaForQuestionAnswering