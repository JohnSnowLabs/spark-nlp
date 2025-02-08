---
layout: model
title: Chinese roberta_base_finetuned_cluener2020_chinese_pipeline pipeline BertForTokenClassification from daman043
author: John Snow Labs
name: roberta_base_finetuned_cluener2020_chinese_pipeline
date: 2025-02-07
tags: [zh, open_source, pipeline, onnx]
task: Named Entity Recognition
language: zh
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_base_finetuned_cluener2020_chinese_pipeline` is a Chinese model originally trained by daman043.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_finetuned_cluener2020_chinese_pipeline_zh_5.5.1_3.0_1738962124088.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_base_finetuned_cluener2020_chinese_pipeline_zh_5.5.1_3.0_1738962124088.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_base_finetuned_cluener2020_chinese_pipeline", lang = "zh")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_base_finetuned_cluener2020_chinese_pipeline", lang = "zh")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_finetuned_cluener2020_chinese_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|zh|
|Size:|380.9 MB|

## References

https://huggingface.co/daman043/roberta-base-finetuned-cluener2020-chinese

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification