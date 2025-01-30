---
layout: model
title: English summarization_sft_reward_model_deberta_v3_large_v2_romansh_gemma_2b_mask_partial_romansh_random_length_pipeline pipeline DeBertaForSequenceClassification from weepcat
author: John Snow Labs
name: summarization_sft_reward_model_deberta_v3_large_v2_romansh_gemma_2b_mask_partial_romansh_random_length_pipeline
date: 2025-01-24
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`summarization_sft_reward_model_deberta_v3_large_v2_romansh_gemma_2b_mask_partial_romansh_random_length_pipeline` is a English model originally trained by weepcat.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/summarization_sft_reward_model_deberta_v3_large_v2_romansh_gemma_2b_mask_partial_romansh_random_length_pipeline_en_5.5.1_3.0_1737728652073.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/summarization_sft_reward_model_deberta_v3_large_v2_romansh_gemma_2b_mask_partial_romansh_random_length_pipeline_en_5.5.1_3.0_1737728652073.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("summarization_sft_reward_model_deberta_v3_large_v2_romansh_gemma_2b_mask_partial_romansh_random_length_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("summarization_sft_reward_model_deberta_v3_large_v2_romansh_gemma_2b_mask_partial_romansh_random_length_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|summarization_sft_reward_model_deberta_v3_large_v2_romansh_gemma_2b_mask_partial_romansh_random_length_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|832.6 MB|

## References

https://huggingface.co/weepcat/summarization_sft_reward-model-deberta-v3-large-v2_RM-Gemma-2B_mask_partial_rm_random_length

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification