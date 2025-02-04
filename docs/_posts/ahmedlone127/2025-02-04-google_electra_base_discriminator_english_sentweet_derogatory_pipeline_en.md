---
layout: model
title: English google_electra_base_discriminator_english_sentweet_derogatory_pipeline pipeline XlmRoBertaForSequenceClassification from jayanta
author: John Snow Labs
name: google_electra_base_discriminator_english_sentweet_derogatory_pipeline
date: 2025-02-04
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`google_electra_base_discriminator_english_sentweet_derogatory_pipeline` is a English model originally trained by jayanta.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/google_electra_base_discriminator_english_sentweet_derogatory_pipeline_en_5.5.1_3.0_1738693490931.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/google_electra_base_discriminator_english_sentweet_derogatory_pipeline_en_5.5.1_3.0_1738693490931.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("google_electra_base_discriminator_english_sentweet_derogatory_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("google_electra_base_discriminator_english_sentweet_derogatory_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|google_electra_base_discriminator_english_sentweet_derogatory_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|785.7 MB|

## References

https://huggingface.co/jayanta/google-electra-base-discriminator-english-sentweet-derogatory

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification