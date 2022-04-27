---
layout: model
title: Pipeline to Detect Medical Risk Factors
author: John Snow Labs
name: ner_risk_factors_biobert_pipeline
date: 2022-03-21
tags: [licensed, ner, biobert, risk_factor, en]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 3.4.1
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_risk_factors_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_risk_factors_biobert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_risk_factors_biobert_pipeline_en_3.4.1_3.0_1647871536746.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

The sample code snippet may not contain all required fields of a pipeline. In this case, you can reach out a related colab notebook containing the end-to-end pipeline and more by clicking the "Open in Colab" link above.




<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_risk_factors_biobert_pipeline", "en", "clinical/models")

pipeline.annotate('HISTORY OF PRESENT ILLNESS: The patient is a 40-year-old white male who presents with a chief complaint of "chest pain". The patient is diabetic and has a prior history of coronary artery disease. The patient presents today stating that his chest pain started yesterday evening and has been somewhat intermittent. The severity of the pain has progressively increased. He describes the pain as a sharp and heavy pain which radiates to his neck & left arm. He ranks the pain a 7 on a scale of 1-10. He admits some shortness of breath & diaphoresis. He states that he has had nausea & 3 episodes of vomiting tonight. He denies any fever or chills. He admits prior episodes of similar pain prior to his PTCA in 1995. He states the pain is somewhat worse with walking and seems to be relieved with rest. There is no change in pain with positioning. He states that he took 3 nitroglycerin tablets sublingually over the past 1 hour, which he states has partially relieved his pain. The patient ranks his present pain a 4 on a scale of 1-10. The most recent episode of pain has lasted one-hour. The patient denies any history of recent surgery, head trauma, recent stroke, abnormal bleeding such as blood in urine or stool or nosebleed.\n\nREVIEW OF SYSTEMS: All other systems reviewed & are negative.\n\nPAST MEDICAL HISTORY: Diabetes mellitus type II, hypertension, coronary artery disease, atrial fibrillation, status post PTCA in 1995 by Dr. ABC.\n\nSOCIAL HISTORY: Denies alcohol or drugs. Smokes 2 packs of cigarettes per day. Works as a banker.\n\nFAMILY HISTORY: Positive for coronary artery disease (father & brother).')
```
```scala
val pipeline = new PretrainedPipeline("ner_risk_factors_biobert_pipeline", "en", "clinical/models")

pipeline.annotate("HISTORY OF PRESENT ILLNESS: The patient is a 40-year-old white male who presents with a chief complaint of "chest pain". The patient is diabetic and has a prior history of coronary artery disease. The patient presents today stating that his chest pain started yesterday evening and has been somewhat intermittent. The severity of the pain has progressively increased. He describes the pain as a sharp and heavy pain which radiates to his neck & left arm. He ranks the pain a 7 on a scale of 1-10. He admits some shortness of breath & diaphoresis. He states that he has had nausea & 3 episodes of vomiting tonight. He denies any fever or chills. He admits prior episodes of similar pain prior to his PTCA in 1995. He states the pain is somewhat worse with walking and seems to be relieved with rest. There is no change in pain with positioning. He states that he took 3 nitroglycerin tablets sublingually over the past 1 hour, which he states has partially relieved his pain. The patient ranks his present pain a 4 on a scale of 1-10. The most recent episode of pain has lasted one-hour. The patient denies any history of recent surgery, head trauma, recent stroke, abnormal bleeding such as blood in urine or stool or nosebleed.\n\nREVIEW OF SYSTEMS: All other systems reviewed & are negative.\n\nPAST MEDICAL HISTORY: Diabetes mellitus type II, hypertension, coronary artery disease, atrial fibrillation, status post PTCA in 1995 by Dr. ABC.\n\nSOCIAL HISTORY: Denies alcohol or drugs. Smokes 2 packs of cigarettes per day. Works as a banker.\n\nFAMILY HISTORY: Positive for coronary artery disease (father & brother).")
```
</div>

## Results

```bash
+----------------------------------------+------------+
|chunks                                  |entities    |
+----------------------------------------+------------+
|diabetic                                |DIABETES    |
|prior history of coronary artery disease|CAD         |
|PTCA in 1995.                           |CAD         |
|Diabetes mellitus type II               |DIABETES    |
|hypertension                            |HYPERTENSION|
|coronary artery disease                 |CAD         |
|Smokes 2 packs of cigarettes per day    |SMOKER      |
+----------------------------------------+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_risk_factors_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP for Healthcare 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|422.0 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverter
