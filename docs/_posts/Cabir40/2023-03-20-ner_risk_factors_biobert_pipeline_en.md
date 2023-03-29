---
layout: model
title: Pipeline to Detect medical risk factors (biobert)
author: John Snow Labs
name: ner_risk_factors_biobert_pipeline
date: 2023-03-20
tags: [ner, clinical, licensed, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_risk_factors_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_risk_factors_biobert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_risk_factors_biobert_pipeline_en_4.3.0_3.2_1679314882627.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_risk_factors_biobert_pipeline_en_4.3.0_3.2_1679314882627.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_risk_factors_biobert_pipeline", "en", "clinical/models")

text = '''ISTORY OF PRESENT ILLNESS: The patient is a 40-year-old white male who presents with a chief complaint of "chest pain". The patient is diabetic and has a prior history of coronary artery disease. The patient presents today stating that his chest pain started yesterday evening and has been somewhat intermittent. The severity of the pain has progressively increased. He describes the pain as a sharp and heavy pain which radiates to his neck & left arm. He ranks the pain a 7 on a scale of 1-10. He admits some shortness of breath & diaphoresis. He states that he has had nausea & 3 episodes of vomiting tonight. He denies any fever or chills. He admits prior episodes of similar pain prior to his PTCA in 1995. He states the pain is somewhat worse with walking and seems to be relieved with rest. There is no change in pain with positioning. He states that he took 3 nitroglycerin tablets sublingually over the past 1 hour, which he states has partially relieved his pain. The patient ranks his present pain a 4 on a scale of 1-10. The most recent episode of pain has lasted one-hour. The patient denies any history of recent surgery, head trauma, recent stroke, abnormal bleeding such as blood in urine or stool or nosebleed.

REVIEW OF SYSTEMS: All other systems reviewed & are negative.

PAST MEDICAL HISTORY: Diabetes mellitus type II, hypertension, coronary artery disease, atrial fibrillation, status post PTCA in 1995 by Dr. ABC.

SOCIAL HISTORY: Denies alcohol or drugs. Smokes 2 packs of cigarettes per day. Works as a banker.

FAMILY HISTORY: Positive for coronary artery disease (father & brother).'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_risk_factors_biobert_pipeline", "en", "clinical/models")

val text = "ISTORY OF PRESENT ILLNESS: The patient is a 40-year-old white male who presents with a chief complaint of "chest pain". The patient is diabetic and has a prior history of coronary artery disease. The patient presents today stating that his chest pain started yesterday evening and has been somewhat intermittent. The severity of the pain has progressively increased. He describes the pain as a sharp and heavy pain which radiates to his neck & left arm. He ranks the pain a 7 on a scale of 1-10. He admits some shortness of breath & diaphoresis. He states that he has had nausea & 3 episodes of vomiting tonight. He denies any fever or chills. He admits prior episodes of similar pain prior to his PTCA in 1995. He states the pain is somewhat worse with walking and seems to be relieved with rest. There is no change in pain with positioning. He states that he took 3 nitroglycerin tablets sublingually over the past 1 hour, which he states has partially relieved his pain. The patient ranks his present pain a 4 on a scale of 1-10. The most recent episode of pain has lasted one-hour. The patient denies any history of recent surgery, head trauma, recent stroke, abnormal bleeding such as blood in urine or stool or nosebleed.

REVIEW OF SYSTEMS: All other systems reviewed & are negative.

PAST MEDICAL HISTORY: Diabetes mellitus type II, hypertension, coronary artery disease, atrial fibrillation, status post PTCA in 1995 by Dr. ABC.

SOCIAL HISTORY: Denies alcohol or drugs. Smokes 2 packs of cigarettes per day. Works as a banker.

FAMILY HISTORY: Positive for coronary artery disease (father & brother)."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                                |   begin |   end | ner_label    |   confidence |
|---:|:-----------------------------------------|--------:|------:|:-------------|-------------:|
|  0 | diabetic                                 |     135 |   142 | DIABETES     |     0.9689   |
|  1 | prior history of coronary artery disease |     154 |   193 | CAD          |     0.419617 |
|  2 | PTCA in 1995.                            |     698 |   710 | CAD          |     0.574925 |
|  3 | Diabetes mellitus type II                |    1314 |  1338 | DIABETES     |     0.946325 |
|  4 | hypertension                             |    1341 |  1352 | HYPERTENSION |     0.956    |
|  5 | coronary artery disease                  |    1355 |  1377 | CAD          |     0.7962   |
|  6 | Smokes 2 packs of cigarettes per day     |    1480 |  1515 | SMOKER       |     0.461643 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_risk_factors_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|422.2 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverterInternalModel