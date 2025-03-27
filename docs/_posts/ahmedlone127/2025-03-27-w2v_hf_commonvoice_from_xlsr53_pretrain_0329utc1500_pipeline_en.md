---
layout: model
title: English w2v_hf_commonvoice_from_xlsr53_pretrain_0329utc1500_pipeline pipeline Wav2Vec2ForCTC from qqpann
author: John Snow Labs
name: w2v_hf_commonvoice_from_xlsr53_pretrain_0329utc1500_pipeline
date: 2025-03-27
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

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`w2v_hf_commonvoice_from_xlsr53_pretrain_0329utc1500_pipeline` is a English model originally trained by qqpann.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/w2v_hf_commonvoice_from_xlsr53_pretrain_0329utc1500_pipeline_en_5.5.1_3.0_1743077502806.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/w2v_hf_commonvoice_from_xlsr53_pretrain_0329utc1500_pipeline_en_5.5.1_3.0_1743077502806.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("w2v_hf_commonvoice_from_xlsr53_pretrain_0329utc1500_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("w2v_hf_commonvoice_from_xlsr53_pretrain_0329utc1500_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|w2v_hf_commonvoice_from_xlsr53_pretrain_0329utc1500_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.2 GB|

## References

https://huggingface.co/qqpann/w2v_hf_commonvoice_from_xlsr53_pretrain_0329UTC1500

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC