---
layout: model
title: English lsg_bart_base_4096_mediqa_chat_taskb_sectionwise_past_medical_family_and_social_history_pipeline pipeline BartTransformer from dhananjay2912
author: John Snow Labs
name: lsg_bart_base_4096_mediqa_chat_taskb_sectionwise_past_medical_family_and_social_history_pipeline
date: 2025-04-03
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

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`lsg_bart_base_4096_mediqa_chat_taskb_sectionwise_past_medical_family_and_social_history_pipeline` is a English model originally trained by dhananjay2912.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lsg_bart_base_4096_mediqa_chat_taskb_sectionwise_past_medical_family_and_social_history_pipeline_en_5.5.1_3.0_1743658768349.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/lsg_bart_base_4096_mediqa_chat_taskb_sectionwise_past_medical_family_and_social_history_pipeline_en_5.5.1_3.0_1743658768349.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("lsg_bart_base_4096_mediqa_chat_taskb_sectionwise_past_medical_family_and_social_history_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("lsg_bart_base_4096_mediqa_chat_taskb_sectionwise_past_medical_family_and_social_history_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|lsg_bart_base_4096_mediqa_chat_taskb_sectionwise_past_medical_family_and_social_history_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|824.6 MB|

## References

https://huggingface.co/dhananjay2912/lsg-bart-base-4096-mediqa-chat-taskb-sectionwise-past_medical_family_and_social_history

## Included Models

- DocumentAssembler
- BartTransformer