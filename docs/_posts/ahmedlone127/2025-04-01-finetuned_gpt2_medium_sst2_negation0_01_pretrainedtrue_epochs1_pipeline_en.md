---
layout: model
title: English finetuned_gpt2_medium_sst2_negation0_01_pretrainedtrue_epochs1_pipeline pipeline GPT2Transformer from jhaochenz
author: John Snow Labs
name: finetuned_gpt2_medium_sst2_negation0_01_pretrainedtrue_epochs1_pipeline
date: 2025-04-01
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finetuned_gpt2_medium_sst2_negation0_01_pretrainedtrue_epochs1_pipeline` is a English model originally trained by jhaochenz.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finetuned_gpt2_medium_sst2_negation0_01_pretrainedtrue_epochs1_pipeline_en_5.5.1_3.0_1743473312277.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finetuned_gpt2_medium_sst2_negation0_01_pretrainedtrue_epochs1_pipeline_en_5.5.1_3.0_1743473312277.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("finetuned_gpt2_medium_sst2_negation0_01_pretrainedtrue_epochs1_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("finetuned_gpt2_medium_sst2_negation0_01_pretrainedtrue_epochs1_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finetuned_gpt2_medium_sst2_negation0_01_pretrainedtrue_epochs1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/jhaochenz/finetuned_gpt2-medium_sst2_negation0.01_pretrainedTrue_epochs1

## Included Models

- DocumentAssembler
- GPT2Transformer