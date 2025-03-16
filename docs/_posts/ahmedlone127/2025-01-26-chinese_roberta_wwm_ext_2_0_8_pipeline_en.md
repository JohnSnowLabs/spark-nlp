---
layout: model
title: English chinese_roberta_wwm_ext_2_0_8_pipeline pipeline BertForQuestionAnswering from DaydreamerF
author: John Snow Labs
name: chinese_roberta_wwm_ext_2_0_8_pipeline
date: 2025-01-26
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`chinese_roberta_wwm_ext_2_0_8_pipeline` is a English model originally trained by DaydreamerF.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/chinese_roberta_wwm_ext_2_0_8_pipeline_en_5.5.1_3.0_1737919190569.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/chinese_roberta_wwm_ext_2_0_8_pipeline_en_5.5.1_3.0_1737919190569.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("chinese_roberta_wwm_ext_2_0_8_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("chinese_roberta_wwm_ext_2_0_8_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|chinese_roberta_wwm_ext_2_0_8_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|381.0 MB|

## References

https://huggingface.co/DaydreamerF/chinese-roberta-wwm-ext-2.0-8

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering