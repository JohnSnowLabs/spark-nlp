---
layout: model
title: Chinese roberta_base_wechsel_chinese_pipeline pipeline RoBertaEmbeddings from benjamin
author: John Snow Labs
name: roberta_base_wechsel_chinese_pipeline
date: 2024-09-14
tags: [zh, open_source, pipeline, onnx]
task: Embeddings
language: zh
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_base_wechsel_chinese_pipeline` is a Chinese model originally trained by benjamin.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_wechsel_chinese_pipeline_zh_5.5.0_3.0_1726300316163.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_base_wechsel_chinese_pipeline_zh_5.5.0_3.0_1726300316163.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_base_wechsel_chinese_pipeline", lang = "zh")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_base_wechsel_chinese_pipeline", lang = "zh")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_wechsel_chinese_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|zh|
|Size:|466.1 MB|

## References

https://huggingface.co/benjamin/roberta-base-wechsel-chinese

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings