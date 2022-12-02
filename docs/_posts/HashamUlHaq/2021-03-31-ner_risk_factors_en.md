---
layout: model
title: Detect Risk Factors
author: John Snow Labs
name: ner_risk_factors
date: 2021-03-31
tags: [ner, clinical, licensed, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.0.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained named entity recognition deep learning model for Heart Disease Risk Factors and Personal Health Information. The SparkNLP deep learning model (MedicalNerModel) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN.

## Predicted Entities

`CAD`, `DIABETES`, `FAMILY_HIST`, `HYPERLIPIDEMIA`, `HYPERTENSION`, `MEDICATION`, `OBESE`, `PHI`, `SMOKER`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_RISK_FACTORS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_risk_factors_en_3.0.0_3.0_1617208442757.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
         
sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_risk_factors", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter()\
 	.setInputCols(["sentence", "token", "ner"])\
 	.setOutputCol("ner_chunk")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])

model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([['HISTORY OF PRESENT ILLNESS: The patient is a 40-year-old white male who presents with a chief complaint of "chest pain". The patient is diabetic and has a prior history of coronary artery disease. The patient presents today stating that his chest pain started yesterday evening and has been somewhat intermittent. The severity of the pain has progressively increased. He describes the pain as a sharp and heavy pain which radiates to his neck & left arm. He ranks the pain a 7 on a scale of 1-10. He admits some shortness of breath & diaphoresis. He states that he has had nausea & 3 episodes of vomiting tonight. He denies any fever or chills. He admits prior episodes of similar pain prior to his PTCA in 1995. He states the pain is somewhat worse with walking and seems to be relieved with rest. There is no change in pain with positioning. He states that he took 3 nitroglycerin tablets sublingually over the past 1 hour, which he states has partially relieved his pain. The patient ranks his present pain a 4 on a scale of 1-10. The most recent episode of pain has lasted one-hour. The patient denies any history of recent surgery, head trauma, recent stroke, abnormal bleeding such as blood in urine or stool or nosebleed.\n\nREVIEW OF SYSTEMS: All other systems reviewed & are negative.\n\nPAST MEDICAL HISTORY: Diabetes mellitus type II, hypertension, coronary artery disease, atrial fibrillation, status post PTCA in 1995 by Dr. ABC.\n\nSOCIAL HISTORY: Denies alcohol or drugs. Smokes 2 packs of cigarettes per day. Works as a banker.\n\nFAMILY HISTORY: Positive for coronary artery disease (father & brother).']], ["text"]))
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
         
val sentence_detector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_risk_factors", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
 	.setInputCols(Array("sentence", "token", "ner"))
 	.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))

val data = Seq("""HISTORY OF PRESENT ILLNESS: The patient is a 40-year-old white male who presents with a chief complaint of "chest pain".The patient is diabetic and has a prior history of coronary artery disease. The patient presents today stating that his chest pain started yesterday evening and has been somewhat intermittent. The severity of the pain has progressively increased. He describes the pain as a sharp and heavy pain which radiates to his neck & left arm. He ranks the pain a 7 on a scale of 1-10. He admits some shortness of breath & diaphoresis. He states that he has had nausea & 3 episodes of vomiting tonight. He denies any fever or chills. He admits prior episodes of similar pain prior to his PTCA in 1995. He states the pain is somewhat worse with walking and seems to be relieved with rest. There is no change in pain with positioning. He states that he took 3 nitroglycerin tablets sublingually over the past 1 hour, which he states has partially relieved his pain. The patient ranks his present pain a 4 on a scale of 1-10. The most recent episode of pain has lasted one-hour.The patient denies any history of recent surgery, head trauma, recent stroke, abnormal bleeding such as blood in urine or stool or nosebleed.REVIEW OF SYSTEMS: All other systems reviewed & are negative.PAST MEDICAL HISTORY: Diabetes mellitus type II, hypertension, coronary artery disease, atrial fibrillation, status post PTCA in 1995 by Dr. ABC.SOCIAL HISTORY: Denies alcohol or drugs. Smokes 2 packs of cigarettes per day. Works as a banker.FAMILY HISTORY: Positive for coronary artery disease (father & brother).""").toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.risk_factors").predict("""chest pain""")
```

</div>

## Results

```bash
+------------------------------------+------------+
|chunk                               |ner         |
+------------------------------------+------------+
|diabetic                            |DIABETES    |
|coronary artery disease             |CAD         |
|Diabetes mellitus type II           |DIABETES    |
|hypertension                        |HYPERTENSION|
|coronary artery disease             |CAD         |
|1995                                |PHI         |
|ABC                                 |PHI         |
|Smokes 2 packs of cigarettes per day|SMOKER      |
|banker                              |PHI         |
+------------------------------------+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_risk_factors|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

Trained on plain n2c2 2014: De-identification and Heart Disease Risk Factors Challenge datasets with ``embeddings_clinical``.
https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/

## Benchmarking

```bash
|    | label            |    tp |    fp |    fn |     prec |      rec |       f1 |
|---:|-----------------:|------:|------:|------:|---------:|---------:|---------:|
|  1 | I-HYPERLIPIDEMIA |     7 |     7 |     7 | 0.5      | 0.5      | 0.5      |
|  2 | B-CAD            |   104 |    52 |   101 | 0.666667 | 0.507317 | 0.576177 |
|  3 | I-DIABETES       |   127 |    67 |    92 | 0.654639 | 0.579909 | 0.615012 |
|  4 | B-HYPERTENSION   |   173 |    52 |    64 | 0.768889 | 0.729958 | 0.748918 |
|  5 | B-OBESE          |    46 |    20 |     3 | 0.69697  | 0.938776 | 0.8      |
|  6 | B-PHI            |  1968 |   599 |   252 | 0.766654 | 0.886486 | 0.822227 |
|  7 | B-HYPERLIPIDEMIA |    71 |    17 |    14 | 0.806818 | 0.835294 | 0.820809 |
|  8 | I-SMOKER         |   116 |    73 |    94 | 0.613757 | 0.552381 | 0.581454 |
|  9 | I-OBESE          |     9 |     8 |     4 | 0.529412 | 0.692308 | 0.6      |
| 10 | I-FAMILY_HIST    |     5 |     0 |    10 | 1        | 0.333333 | 0.5      |
| 11 | B-DIABETES       |   190 |    59 |    58 | 0.763052 | 0.766129 | 0.764587 |
| 12 | B-MEDICATION     |   838 |   224 |    81 | 0.789077 | 0.911861 | 0.846037 |
| 13 | I-PHI            |   597 |   202 |   136 | 0.747184 | 0.814461 | 0.779373 |
| 14 | Macro-average    |  4533 |  1784 |  1600 | 0.620602 | 0.567477 | 0.592852 |
| 15 | Micro-average    |  4533 |  1784 |  1600 | 0.717588 | 0.739116 | 0.728193 |
```