---
layout: model
title: Pipeline to Detect Time-related Terminology
author: John Snow Labs
name: roberta_token_classifier_timex_semeval_pipeline
date: 2022-03-19
tags: [timex, semeval, ner, en, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [roberta_token_classifier_timex_semeval](https://nlp.johnsnowlabs.com/2021/12/28/roberta_token_classifier_timex_semeval_en.html) model.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_TIMEX_SEMEVAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_timex_semeval_pipeline_en_3.4.1_3.0_1647699181926.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_timex_semeval_pipeline_en_3.4.1_3.0_1647699181926.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
timex_pipeline = PretrainedPipeline("roberta_token_classifier_timex_semeval_pipeline", lang = "en")

timex_pipeline.annotate("Model training was started at 22:12C and it took 3 days from Tuesday to Friday.")
```
```scala
val timex_pipeline = new PretrainedPipeline("roberta_token_classifier_timex_semeval_pipeline", lang = "en")

timex_pipeline.annotate("Model training was started at 22:12C and it took 3 days from Tuesday to Friday.")
```
</div>

## Results

```bash
+-------+-----------------+
|chunk  |ner_label        |
+-------+-----------------+
|22:12C |Period           |
|3      |Number           |
|days   |Calendar-Interval|
|Tuesday|Day-Of-Week      |
|to     |Between          |
|Friday |Day-Of-Week      |
+-------+-----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_token_classifier_timex_semeval_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.4.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|439.5 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- RoBertaForTokenClassification
- NerConverter
- Finisher