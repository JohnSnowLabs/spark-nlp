---
layout: model
title: English multi_doc_sum_t5_slide_norwegian_prompt_pipeline pipeline T5Transformer from whu9
author: John Snow Labs
name: multi_doc_sum_t5_slide_norwegian_prompt_pipeline
date: 2024-08-21
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`multi_doc_sum_t5_slide_norwegian_prompt_pipeline` is a English model originally trained by whu9.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/multi_doc_sum_t5_slide_norwegian_prompt_pipeline_en_5.4.2_3.0_1724280144187.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/multi_doc_sum_t5_slide_norwegian_prompt_pipeline_en_5.4.2_3.0_1724280144187.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("multi_doc_sum_t5_slide_norwegian_prompt_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("multi_doc_sum_t5_slide_norwegian_prompt_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|multi_doc_sum_t5_slide_norwegian_prompt_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|225.2 MB|

## References

https://huggingface.co/whu9/multi_doc_sum_t5_slide_no_prompt

## Included Models

- DocumentAssembler
- T5Transformer