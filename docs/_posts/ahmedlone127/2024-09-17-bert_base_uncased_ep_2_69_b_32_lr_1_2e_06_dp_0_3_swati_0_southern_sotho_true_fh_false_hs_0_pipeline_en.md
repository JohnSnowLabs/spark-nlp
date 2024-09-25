---
layout: model
title: English bert_base_uncased_ep_2_69_b_32_lr_1_2e_06_dp_0_3_swati_0_southern_sotho_true_fh_false_hs_0_pipeline pipeline BertForQuestionAnswering from danielkty22
author: John Snow Labs
name: bert_base_uncased_ep_2_69_b_32_lr_1_2e_06_dp_0_3_swati_0_southern_sotho_true_fh_false_hs_0_pipeline
date: 2024-09-17
tags: [en, open_source, pipeline, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_base_uncased_ep_2_69_b_32_lr_1_2e_06_dp_0_3_swati_0_southern_sotho_true_fh_false_hs_0_pipeline` is a English model originally trained by danielkty22.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_uncased_ep_2_69_b_32_lr_1_2e_06_dp_0_3_swati_0_southern_sotho_true_fh_false_hs_0_pipeline_en_5.5.0_3.0_1726589908256.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_uncased_ep_2_69_b_32_lr_1_2e_06_dp_0_3_swati_0_southern_sotho_true_fh_false_hs_0_pipeline_en_5.5.0_3.0_1726589908256.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_base_uncased_ep_2_69_b_32_lr_1_2e_06_dp_0_3_swati_0_southern_sotho_true_fh_false_hs_0_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_base_uncased_ep_2_69_b_32_lr_1_2e_06_dp_0_3_swati_0_southern_sotho_true_fh_false_hs_0_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_uncased_ep_2_69_b_32_lr_1_2e_06_dp_0_3_swati_0_southern_sotho_true_fh_false_hs_0_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.2 MB|

## References

https://huggingface.co/danielkty22/bert-base-uncased-ep-2.69-b-32-lr-1.2e-06-dp-0.3-ss-0-st-True-fh-False-hs-0

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering