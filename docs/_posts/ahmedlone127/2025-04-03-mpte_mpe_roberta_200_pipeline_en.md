---
layout: model
title: English mpte_mpe_roberta_200_pipeline pipeline RoBertaForSequenceClassification from veronica320
author: John Snow Labs
name: mpte_mpe_roberta_200_pipeline
date: 2025-04-03
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mpte_mpe_roberta_200_pipeline` is a English model originally trained by veronica320.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mpte_mpe_roberta_200_pipeline_en_5.5.1_3.0_1743638973957.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mpte_mpe_roberta_200_pipeline_en_5.5.1_3.0_1743638973957.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mpte_mpe_roberta_200_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mpte_mpe_roberta_200_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mpte_mpe_roberta_200_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|429.9 MB|

## References

https://huggingface.co/veronica320/MPTE_MPE_roberta_200

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification