---
layout: model
title: German gpt2_small_german_finetune_oscar_pipeline pipeline GPT2Transformer from ml6team
author: John Snow Labs
name: gpt2_small_german_finetune_oscar_pipeline
date: 2025-04-06
tags: [de, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: de
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gpt2_small_german_finetune_oscar_pipeline` is a German model originally trained by ml6team.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gpt2_small_german_finetune_oscar_pipeline_de_5.5.1_3.0_1743900321967.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gpt2_small_german_finetune_oscar_pipeline_de_5.5.1_3.0_1743900321967.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gpt2_small_german_finetune_oscar_pipeline", lang = "de")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gpt2_small_german_finetune_oscar_pipeline", lang = "de")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gpt2_small_german_finetune_oscar_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|de|
|Size:|298.3 MB|

## References

https://huggingface.co/ml6team/gpt2-small-german-finetune-oscar

## Included Models

- DocumentAssembler
- GPT2Transformer