---
layout: model
title: English mbert_kmeans_0_05_mmarco_l_all_english_s_all_30k_distill_test_mt5_base_mmarco_v2_30k_pipeline pipeline BertEmbeddings from spear-model
author: John Snow Labs
name: mbert_kmeans_0_05_mmarco_l_all_english_s_all_30k_distill_test_mt5_base_mmarco_v2_30k_pipeline
date: 2025-01-26
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mbert_kmeans_0_05_mmarco_l_all_english_s_all_30k_distill_test_mt5_base_mmarco_v2_30k_pipeline` is a English model originally trained by spear-model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mbert_kmeans_0_05_mmarco_l_all_english_s_all_30k_distill_test_mt5_base_mmarco_v2_30k_pipeline_en_5.5.1_3.0_1737861576563.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mbert_kmeans_0_05_mmarco_l_all_english_s_all_30k_distill_test_mt5_base_mmarco_v2_30k_pipeline_en_5.5.1_3.0_1737861576563.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mbert_kmeans_0_05_mmarco_l_all_english_s_all_30k_distill_test_mt5_base_mmarco_v2_30k_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mbert_kmeans_0_05_mmarco_l_all_english_s_all_30k_distill_test_mt5_base_mmarco_v2_30k_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mbert_kmeans_0_05_mmarco_l_all_english_s_all_30k_distill_test_mt5_base_mmarco_v2_30k_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|665.1 MB|

## References

https://huggingface.co/spear-model/mbert-kmeans-0.05.mmarco.L-all-en.S-all.30K.distill-test.mt5-base-mmarco-v2.30K

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings