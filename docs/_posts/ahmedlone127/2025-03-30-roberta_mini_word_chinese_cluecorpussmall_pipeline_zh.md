---
layout: model
title: Chinese roberta_mini_word_chinese_cluecorpussmall_pipeline pipeline BertEmbeddings from uer
author: John Snow Labs
name: roberta_mini_word_chinese_cluecorpussmall_pipeline
date: 2025-03-30
tags: [zh, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_mini_word_chinese_cluecorpussmall_pipeline` is a Chinese model originally trained by uer.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_mini_word_chinese_cluecorpussmall_pipeline_zh_5.5.1_3.0_1743322969825.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_mini_word_chinese_cluecorpussmall_pipeline_zh_5.5.1_3.0_1743322969825.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_mini_word_chinese_cluecorpussmall_pipeline", lang = "zh")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_mini_word_chinese_cluecorpussmall_pipeline", lang = "zh")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_mini_word_chinese_cluecorpussmall_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|zh|
|Size:|109.4 MB|

## References

https://huggingface.co/uer/roberta-mini-word-chinese-cluecorpussmall

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings