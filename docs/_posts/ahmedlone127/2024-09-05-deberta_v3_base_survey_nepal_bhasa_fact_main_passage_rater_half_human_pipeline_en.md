---
layout: model
title: English deberta_v3_base_survey_nepal_bhasa_fact_main_passage_rater_half_human_pipeline pipeline DeBertaForSequenceClassification from domenicrosati
author: John Snow Labs
name: deberta_v3_base_survey_nepal_bhasa_fact_main_passage_rater_half_human_pipeline
date: 2024-09-05
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

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`deberta_v3_base_survey_nepal_bhasa_fact_main_passage_rater_half_human_pipeline` is a English model originally trained by domenicrosati.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deberta_v3_base_survey_nepal_bhasa_fact_main_passage_rater_half_human_pipeline_en_5.5.0_3.0_1725561450795.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/deberta_v3_base_survey_nepal_bhasa_fact_main_passage_rater_half_human_pipeline_en_5.5.0_3.0_1725561450795.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("deberta_v3_base_survey_nepal_bhasa_fact_main_passage_rater_half_human_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("deberta_v3_base_survey_nepal_bhasa_fact_main_passage_rater_half_human_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deberta_v3_base_survey_nepal_bhasa_fact_main_passage_rater_half_human_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|610.5 MB|

## References

https://huggingface.co/domenicrosati/deberta-v3-base-survey-new_fact_main_passage-rater-half-human

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification