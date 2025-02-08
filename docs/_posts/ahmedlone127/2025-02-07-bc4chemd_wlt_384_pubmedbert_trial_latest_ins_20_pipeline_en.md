---
layout: model
title: English bc4chemd_wlt_384_pubmedbert_trial_latest_ins_20_pipeline pipeline BertForTokenClassification from mobashgr
author: John Snow Labs
name: bc4chemd_wlt_384_pubmedbert_trial_latest_ins_20_pipeline
date: 2025-02-07
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bc4chemd_wlt_384_pubmedbert_trial_latest_ins_20_pipeline` is a English model originally trained by mobashgr.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bc4chemd_wlt_384_pubmedbert_trial_latest_ins_20_pipeline_en_5.5.1_3.0_1738960847858.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bc4chemd_wlt_384_pubmedbert_trial_latest_ins_20_pipeline_en_5.5.1_3.0_1738960847858.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bc4chemd_wlt_384_pubmedbert_trial_latest_ins_20_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bc4chemd_wlt_384_pubmedbert_trial_latest_ins_20_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bc4chemd_wlt_384_pubmedbert_trial_latest_ins_20_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|408.0 MB|

## References

https://huggingface.co/mobashgr/BC4CHEMD-WLT-384-PubMedBERT-Trial-latest-ins-20

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification