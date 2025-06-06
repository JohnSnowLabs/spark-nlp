---
layout: model
title: English zinc10m_gpt2_smiles_bpe_combined_step2_pipeline pipeline GPT2Transformer from jarod0411
author: John Snow Labs
name: zinc10m_gpt2_smiles_bpe_combined_step2_pipeline
date: 2024-12-19
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

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`zinc10m_gpt2_smiles_bpe_combined_step2_pipeline` is a English model originally trained by jarod0411.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/zinc10m_gpt2_smiles_bpe_combined_step2_pipeline_en_5.5.1_3.0_1734591509122.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/zinc10m_gpt2_smiles_bpe_combined_step2_pipeline_en_5.5.1_3.0_1734591509122.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("zinc10m_gpt2_smiles_bpe_combined_step2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("zinc10m_gpt2_smiles_bpe_combined_step2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|zinc10m_gpt2_smiles_bpe_combined_step2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|466.8 MB|

## References

https://huggingface.co/jarod0411/zinc10M_gpt2_SMILES_bpe_combined_step2

## Included Models

- DocumentAssembler
- GPT2Transformer