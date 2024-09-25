---
layout: model
title: Multilingual markuus_bert_base_multilingual_squad_cqa_english_pipeline pipeline BertForQuestionAnswering from imrazaa
author: John Snow Labs
name: markuus_bert_base_multilingual_squad_cqa_english_pipeline
date: 2024-09-21
tags: [xx, open_source, pipeline, onnx]
task: Question Answering
language: xx
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`markuus_bert_base_multilingual_squad_cqa_english_pipeline` is a Multilingual model originally trained by imrazaa.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/markuus_bert_base_multilingual_squad_cqa_english_pipeline_xx_5.5.0_3.0_1726946516988.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/markuus_bert_base_multilingual_squad_cqa_english_pipeline_xx_5.5.0_3.0_1726946516988.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("markuus_bert_base_multilingual_squad_cqa_english_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("markuus_bert_base_multilingual_squad_cqa_english_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|markuus_bert_base_multilingual_squad_cqa_english_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|625.5 MB|

## References

https://huggingface.co/imrazaa/markuus-bert-base-multilingual-squad-cqa-en

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering