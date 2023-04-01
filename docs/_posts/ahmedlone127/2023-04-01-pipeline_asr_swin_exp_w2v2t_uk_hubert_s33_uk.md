---
layout: model
title: Ukrainian asr_swin_exp_w2v2t_uk_hubert_s33 TFHubertForCTC from jonatasgrosman
author: John Snow Labs
name: pipeline_asr_swin_exp_w2v2t_uk_hubert_s33
date: 2023-04-01
tags: [hubert, uk, open_source, audio, asr, pipeline]
task: Automatic Speech Recognition
language: uk
edition: Spark NLP 4.4.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained  Hubert  pipeline, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_swin_exp_w2v2t_uk_hubert_s33` is a Ukrainian model originally trained by jonatasgrosman.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_asr_swin_exp_w2v2t_uk_hubert_s33_uk_4.4.0_3.0_1680367983368.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pipeline_asr_swin_exp_w2v2t_uk_hubert_s33_uk_4.4.0_3.0_1680367983368.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_asr_swin_exp_w2v2t_uk_hubert_s33', lang = 'uk')
    annotations =  pipeline.transform(audioDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_asr_swin_exp_w2v2t_uk_hubert_s33", lang = "uk")
    val annotations = pipeline.transform(audioDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_asr_swin_exp_w2v2t_uk_hubert_s33|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|uk|
|Size:|2.4 GB|

## Included Models

- AudioAssembler
- HubertForCTC