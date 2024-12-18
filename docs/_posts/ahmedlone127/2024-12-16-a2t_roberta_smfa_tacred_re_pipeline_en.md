---
layout: model
title: English a2t_roberta_smfa_tacred_re_pipeline pipeline RoBertaForZeroShotClassification from HiTZ
author: John Snow Labs
name: a2t_roberta_smfa_tacred_re_pipeline
date: 2024-12-16
tags: [en, open_source, pipeline, onnx]
task: Zero-Shot Classification
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

Pretrained RoBertaForZeroShotClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`a2t_roberta_smfa_tacred_re_pipeline` is a English model originally trained by HiTZ.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/a2t_roberta_smfa_tacred_re_pipeline_en_5.5.1_3.0_1734341802286.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/a2t_roberta_smfa_tacred_re_pipeline_en_5.5.1_3.0_1734341802286.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("a2t_roberta_smfa_tacred_re_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("a2t_roberta_smfa_tacred_re_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|a2t_roberta_smfa_tacred_re_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/HiTZ/A2T_RoBERTa_SMFA_TACRED-re

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForZeroShotClassification