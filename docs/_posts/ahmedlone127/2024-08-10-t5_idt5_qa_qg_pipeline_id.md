---
layout: model
title: Indonesian t5_idt5_qa_qg_pipeline pipeline T5Transformer from muchad
author: John Snow Labs
name: t5_idt5_qa_qg_pipeline
date: 2024-08-10
tags: [id, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: id
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`t5_idt5_qa_qg_pipeline` is a Indonesian model originally trained by muchad.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_idt5_qa_qg_pipeline_id_5.4.2_3.0_1723330013815.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_idt5_qa_qg_pipeline_id_5.4.2_3.0_1723330013815.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("t5_idt5_qa_qg_pipeline", lang = "id")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("t5_idt5_qa_qg_pipeline", lang = "id")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_idt5_qa_qg_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|id|
|Size:|981.0 MB|

## References

https://huggingface.co/muchad/idt5-qa-qg

## Included Models

- DocumentAssembler
- T5Transformer