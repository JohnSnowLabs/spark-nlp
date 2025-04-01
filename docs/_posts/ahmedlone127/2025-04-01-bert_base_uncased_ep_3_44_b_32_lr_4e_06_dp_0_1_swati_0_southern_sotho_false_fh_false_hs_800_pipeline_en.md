---
layout: model
title: English bert_base_uncased_ep_3_44_b_32_lr_4e_06_dp_0_1_swati_0_southern_sotho_false_fh_false_hs_800_pipeline pipeline BertForQuestionAnswering from danielkty22
author: John Snow Labs
name: bert_base_uncased_ep_3_44_b_32_lr_4e_06_dp_0_1_swati_0_southern_sotho_false_fh_false_hs_800_pipeline
date: 2025-04-01
tags: [en, open_source, pipeline, onnx]
task: Question Answering
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_base_uncased_ep_3_44_b_32_lr_4e_06_dp_0_1_swati_0_southern_sotho_false_fh_false_hs_800_pipeline` is a English model originally trained by danielkty22.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_uncased_ep_3_44_b_32_lr_4e_06_dp_0_1_swati_0_southern_sotho_false_fh_false_hs_800_pipeline_en_5.5.1_3.0_1743516856299.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_uncased_ep_3_44_b_32_lr_4e_06_dp_0_1_swati_0_southern_sotho_false_fh_false_hs_800_pipeline_en_5.5.1_3.0_1743516856299.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_base_uncased_ep_3_44_b_32_lr_4e_06_dp_0_1_swati_0_southern_sotho_false_fh_false_hs_800_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_base_uncased_ep_3_44_b_32_lr_4e_06_dp_0_1_swati_0_southern_sotho_false_fh_false_hs_800_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_uncased_ep_3_44_b_32_lr_4e_06_dp_0_1_swati_0_southern_sotho_false_fh_false_hs_800_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.2 MB|

## References

https://huggingface.co/danielkty22/bert-base-uncased-ep-3.44-b-32-lr-4e-06-dp-0.1-ss-0-st-False-fh-False-hs-800

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering