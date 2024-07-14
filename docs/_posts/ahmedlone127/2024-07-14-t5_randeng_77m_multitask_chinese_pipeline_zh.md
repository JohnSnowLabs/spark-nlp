---
layout: model
title: Chinese t5_randeng_77m_multitask_chinese_pipeline pipeline T5Transformer from IDEA-CCNL
author: John Snow Labs
name: t5_randeng_77m_multitask_chinese_pipeline
date: 2024-07-14
tags: [zh, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: zh
edition: Spark NLP 5.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`t5_randeng_77m_multitask_chinese_pipeline` is a Chinese model originally trained by IDEA-CCNL.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_randeng_77m_multitask_chinese_pipeline_zh_5.4.1_3.0_1720962951649.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_randeng_77m_multitask_chinese_pipeline_zh_5.4.1_3.0_1720962951649.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("t5_randeng_77m_multitask_chinese_pipeline", lang = "zh")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("t5_randeng_77m_multitask_chinese_pipeline", lang = "zh")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_randeng_77m_multitask_chinese_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|zh|
|Size:|349.1 MB|

## References

https://huggingface.co/IDEA-CCNL/Randeng-T5-77M-MultiTask-Chinese

## Included Models

- DocumentAssembler
- T5Transformer