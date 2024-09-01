---
layout: model
title: English deberta_v3_large_survey_cross_passage_consistency_rater_gpt4_pipeline pipeline DeBertaForSequenceClassification from domenicrosati
author: John Snow Labs
name: deberta_v3_large_survey_cross_passage_consistency_rater_gpt4_pipeline
date: 2024-09-01
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`deberta_v3_large_survey_cross_passage_consistency_rater_gpt4_pipeline` is a English model originally trained by domenicrosati.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deberta_v3_large_survey_cross_passage_consistency_rater_gpt4_pipeline_en_5.4.2_3.0_1725183888113.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/deberta_v3_large_survey_cross_passage_consistency_rater_gpt4_pipeline_en_5.4.2_3.0_1725183888113.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("deberta_v3_large_survey_cross_passage_consistency_rater_gpt4_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("deberta_v3_large_survey_cross_passage_consistency_rater_gpt4_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deberta_v3_large_survey_cross_passage_consistency_rater_gpt4_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.5 GB|

## References

https://huggingface.co/domenicrosati/deberta-v3-large-survey-cross_passage_consistency-rater-gpt4

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification