---
layout: model
title: English distilroberta_nsfw_prompt_stable_diffusion_pipeline pipeline RoBertaForSequenceClassification from AdamCodd
author: John Snow Labs
name: distilroberta_nsfw_prompt_stable_diffusion_pipeline
date: 2024-09-03
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilroberta_nsfw_prompt_stable_diffusion_pipeline` is a English model originally trained by AdamCodd.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilroberta_nsfw_prompt_stable_diffusion_pipeline_en_5.5.0_3.0_1725336560276.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilroberta_nsfw_prompt_stable_diffusion_pipeline_en_5.5.0_3.0_1725336560276.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distilroberta_nsfw_prompt_stable_diffusion_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distilroberta_nsfw_prompt_stable_diffusion_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilroberta_nsfw_prompt_stable_diffusion_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|309.1 MB|

## References

https://huggingface.co/AdamCodd/distilroberta-nsfw-prompt-stable-diffusion

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification