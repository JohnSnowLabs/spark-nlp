---
layout: model
title: Italian it5_efficient_small_el32_question_generation_pipeline pipeline T5Transformer from gsarti
author: John Snow Labs
name: it5_efficient_small_el32_question_generation_pipeline
date: 2024-08-07
tags: [it, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: it
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`it5_efficient_small_el32_question_generation_pipeline` is a Italian model originally trained by gsarti.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/it5_efficient_small_el32_question_generation_pipeline_it_5.4.2_3.0_1723056478595.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/it5_efficient_small_el32_question_generation_pipeline_it_5.4.2_3.0_1723056478595.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("it5_efficient_small_el32_question_generation_pipeline", lang = "it")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("it5_efficient_small_el32_question_generation_pipeline", lang = "it")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|it5_efficient_small_el32_question_generation_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|it|
|Size:|654.8 MB|

## References

https://huggingface.co/gsarti/it5-efficient-small-el32-question-generation

## Included Models

- DocumentAssembler
- T5Transformer