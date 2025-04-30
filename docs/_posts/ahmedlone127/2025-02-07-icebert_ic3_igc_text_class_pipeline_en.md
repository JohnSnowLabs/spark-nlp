---
layout: model
title: English icebert_ic3_igc_text_class_pipeline pipeline RoBertaForSequenceClassification from elenaovv
author: John Snow Labs
name: icebert_ic3_igc_text_class_pipeline
date: 2025-02-07
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`icebert_ic3_igc_text_class_pipeline` is a English model originally trained by elenaovv.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/icebert_ic3_igc_text_class_pipeline_en_5.5.1_3.0_1738963443769.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/icebert_ic3_igc_text_class_pipeline_en_5.5.1_3.0_1738963443769.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("icebert_ic3_igc_text_class_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("icebert_ic3_igc_text_class_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|icebert_ic3_igc_text_class_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|466.7 MB|

## References

https://huggingface.co/elenaovv/IceBERT-ic3-igc-text-class

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification