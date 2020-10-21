---
layout: model
title: Named Entity Recognition for Healthcare in German
author: John Snow Labs
name: ner_healthcare
class: NerDLModel
language: de
repository: clinical/models
date: 2020-09-06
tags: [clinical,ner,de]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.

## Predicted Entities 
BIOLOGICAL_CHEMISTRY, BIOLOGICAL_PARAMETER, BODY_FLUID, BODY_PART, DEGREE, DIAGLAB_PROCEDURE, DOSING, LOCAL_SPECIFICATION, MEASUREMENT, MEDICAL_CONDITION, MEDICAL_DEVICE, MEDICAL_SPECIFICATION, MEDICATION, PERSON, PROCESS, STATE_OF_HEALTH, TIME_INFORMATION, TISSUE, TREATMENT

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_HEALTHCARE_DE/){:.button.button-orange.button-orange-trans.co.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/14.German_Healthcare_Models.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_healthcare_de_2.5.5_2.4_1599433028253.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...
model = NerDLModel.pretrained("ner_healthcare","de","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")

clinical_ner_converter = NerConverterInternal().setInputCols(["sentence", "token", "ner"]).setOutputCol("ner_chunk")

clinical_ner_pipeline = Pipeline(
      stages = [
      documentAssembler,
      sentenceDetector,
      tokenizer,
      word_embeddings,
      clinical_ner,
      clinical_ner_converter
      ])
empty_df = spark.createDataFrame([['']]).toDF("text")

clinical_light_model = LightPipeline(clinical_ner_pipeline.fit(empty_df))

light_model.fullAnnotate('''Das Kleinzellige Bronchialkarzinom (Kleinzelliger Lungenkrebs, SCLC) ist Hernia femoralis, Akne, einseitig, ein hochmalignes bronchogenes Karzinom, das überwiegend im Zentrum der Lunge, in einem Hauptbronchus entsteht. Die mittlere Prävalenz wird auf 1/20.000 geschätzt. Vom SCLC sind hauptsächlich Peronen mittleren Alters (27-66 Jahre) mit Raucheranamnese betroffen.''')

 
```

```scala
val model = NerDLModel.pretrained("ner_healthcare","de","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>

## Results

```bash
+----+-------------------+-----------------------+
|  # |            chunks |              entities |
+----+-------------------+-----------------------+
|  0 |      Kleinzellige |           MEASUREMENT |
+----+-------------------+-----------------------+
|  1 | Bronchialkarzinom |     MEDICAL_CONDITION |
+----+-------------------+-----------------------+
|  2 |     Kleinzelliger |             BODY_PART |
+----+-------------------+-----------------------+
|  3 |       Lungenkrebs |     MEDICAL_CONDITION |
+----+-------------------+-----------------------+
|  4 |              SCLC |     MEDICAL_CONDITION |
+----+-------------------+-----------------------+
|  5 |            Hernia |     MEDICAL_CONDITION |
+----+-------------------+-----------------------+
|  6 |         femoralis |   LOCAL_SPECIFICATION |
+----+-------------------+-----------------------+
|  7 |              Akne |     MEDICAL_CONDITION |
+----+-------------------+-----------------------+
|  8 |         einseitig |           MEASUREMENT |
+----+-------------------+-----------------------+
|  9 |      hochmalignes |     MEDICAL_CONDITION |
+----+-------------------+-----------------------+
| 10 |      bronchogenes |   LOCAL_SPECIFICATION |
+----+-------------------+-----------------------+
| 11 |          Karzinom |             BODY_PART |
+----+-------------------+-----------------------+
| 12 |             Lunge |             BODY_PART |
+----+-------------------+-----------------------+
| 13 |     Hauptbronchus |             BODY_PART |
+----+-------------------+-----------------------+
| 14 |          mittlere |           MEASUREMENT |
+----+-------------------+-----------------------+
| 15 |         Prävalenz |     MEDICAL_CONDITION |
+----+-------------------+-----------------------+
| 16 |              SCLC |     DIAGLAB_PROCEDURE |
+----+-------------------+-----------------------+
| 17 |       27-66 Jahre |           MEASUREMENT |
+----+-------------------+-----------------------+
| 18 |   Raucheranamnese | MEDICAL_SPECIFICATION |
+----+-------------------+-----------------------+
| 19 |         betroffen |     MEDICAL_CONDITION |
+----+-------------------+-----------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|----------------|----------------------------------|
| Name:          | ner_healthcare                   |
| Type:   | NerDLModel                       		|
| Compatibility: | Spark NLP 2.5.5+                 |
| License:       | Licensed                         |
| Edition:       | Official                       	|
|Input labels:   | [sentence, token, word_embeddings] |
|Output labels:  | [ner]                              |
| Language:      | de                               |
| Dependencies:  | w2v_cc_300d                           |

{:.h2_title}
## Data Source
Trained on 2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text with `w2v_cc_300d`