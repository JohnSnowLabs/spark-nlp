---
layout: model
title: Multilingual markuus_bert_base_multilingual_squad_cqa_l1_l2_pipeline pipeline BertForQuestionAnswering from imrazaa
author: John Snow Labs
name: markuus_bert_base_multilingual_squad_cqa_l1_l2_pipeline
date: 2025-02-01
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`markuus_bert_base_multilingual_squad_cqa_l1_l2_pipeline` is a Multilingual model originally trained by imrazaa.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/markuus_bert_base_multilingual_squad_cqa_l1_l2_pipeline_xx_5.5.1_3.0_1738376816311.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/markuus_bert_base_multilingual_squad_cqa_l1_l2_pipeline_xx_5.5.1_3.0_1738376816311.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("markuus_bert_base_multilingual_squad_cqa_l1_l2_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("markuus_bert_base_multilingual_squad_cqa_l1_l2_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|markuus_bert_base_multilingual_squad_cqa_l1_l2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|625.5 MB|

## References

https://huggingface.co/imrazaa/markuus-bert-base-multilingual-squad_cqa_L1-L2

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering