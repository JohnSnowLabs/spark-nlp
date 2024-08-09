---
layout: model
title: Multilingual t5_comment_summarization4designtutor_pipeline pipeline T5Transformer from qiaoyi
author: John Snow Labs
name: t5_comment_summarization4designtutor_pipeline
date: 2024-07-29
tags: [xx, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: xx
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`t5_comment_summarization4designtutor_pipeline` is a Multilingual model originally trained by qiaoyi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_comment_summarization4designtutor_pipeline_xx_5.4.2_3.0_1722248384057.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_comment_summarization4designtutor_pipeline_xx_5.4.2_3.0_1722248384057.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("t5_comment_summarization4designtutor_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("t5_comment_summarization4designtutor_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_comment_summarization4designtutor_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|324.5 MB|

## References

https://huggingface.co/qiaoyi/Comment_Summarization4DesignTutor

## Included Models

- DocumentAssembler
- T5Transformer