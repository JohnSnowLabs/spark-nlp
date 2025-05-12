---
layout: model
title: English libri_alpha_0_75_temp_1_attention_3_layers_distil_with_6_layers_take_2_pipeline pipeline Wav2Vec2ForCTC from rohitp1
author: John Snow Labs
name: libri_alpha_0_75_temp_1_attention_3_layers_distil_with_6_layers_take_2_pipeline
date: 2025-03-31
tags: [en, open_source, pipeline, onnx]
task: Automatic Speech Recognition
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

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`libri_alpha_0_75_temp_1_attention_3_layers_distil_with_6_layers_take_2_pipeline` is a English model originally trained by rohitp1.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/libri_alpha_0_75_temp_1_attention_3_layers_distil_with_6_layers_take_2_pipeline_en_5.5.1_3.0_1743463081702.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/libri_alpha_0_75_temp_1_attention_3_layers_distil_with_6_layers_take_2_pipeline_en_5.5.1_3.0_1743463081702.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("libri_alpha_0_75_temp_1_attention_3_layers_distil_with_6_layers_take_2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("libri_alpha_0_75_temp_1_attention_3_layers_distil_with_6_layers_take_2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|libri_alpha_0_75_temp_1_attention_3_layers_distil_with_6_layers_take_2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|109.4 MB|

## References

https://huggingface.co/rohitp1/libri-alpha-0.75-Temp-1-attention-3-layers-distil-with-6-layers-take-2

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC