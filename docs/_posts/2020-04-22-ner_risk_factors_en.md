---
layout: model
title: Detect Risk Factors
author: John Snow Labs
name: ner_risk_factors_en
date: 2020-04-22
tags: [ner, en, clinical, licensed]
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
Pretrained named entity recognition deep learning model for Heart Disease Risk Factors and Personal Health Information. The SparkNLP deep learning model (NerDL) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN. 

## Predicted Entities  
`CAD`, `DIABETES`, `FAMILY_HIST`, `HYPERLIPIDEMIA`, `HYPERTENSION`, `MEDICATION`, `OBESE`, `PHI`, `SMOKER`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_RISK_FACTORS/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_risk_factors_en_2.4.2_2.4_1587513300751.zip){:.button.button-orange.button-orange-trans.arr.button-icon}



## How to use
Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}


```python
...

clinical_ner = NerDLModel.pretrained("ner_risk_factors", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
...
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame(pd.DataFrame({"text": [
    """HISTORY OF PRESENT ILLNESS: The patient is a 40-year-old white male who presents with a chief complaint of "chest pain".

The patient is diabetic and has a prior history of coronary artery disease. The patient presents today stating that his chest pain started yesterday evening and has been somewhat intermittent. The severity of the pain has progressively increased. He describes the pain as a sharp and heavy pain which radiates to his neck & left arm. He ranks the pain a 7 on a scale of 1-10. He admits some shortness of breath & diaphoresis. He states that he has had nausea & 3 episodes of vomiting tonight. He denies any fever or chills. He admits prior episodes of similar pain prior to his PTCA in 1995. He states the pain is somewhat worse with walking and seems to be relieved with rest. There is no change in pain with positioning. He states that he took 3 nitroglycerin tablets sublingually over the past 1 hour, which he states has partially relieved his pain. The patient ranks his present pain a 4 on a scale of 1-10. The most recent episode of pain has lasted one-hour.

The patient denies any history of recent surgery, head trauma, recent stroke, abnormal bleeding such as blood in urine or stool or nosebleed.

REVIEW OF SYSTEMS: All other systems reviewed & are negative.

PAST MEDICAL HISTORY: Diabetes mellitus type II, hypertension, coronary artery disease, atrial fibrillation, status post PTCA in 1995 by Dr. ABC.

SOCIAL HISTORY: Denies alcohol or drugs. Smokes 2 packs of cigarettes per day. Works as a banker.

FAMILY HISTORY: Positive for coronary artery disease (father & brother)."""
]})))

```

```scala
...

val ner = NerDLModel.pretrained("ner_risk_factors", "en", "clinical/models")
  .setInputCols("sentence", "token", "embeddings")
  .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))

val result = pipeline.fit(Seq.empty["""HISTORY OF PRESENT ILLNESS: The patient is a 40-year-old white male who presents with a chief complaint of "chest pain".The patient is diabetic and has a prior history of coronary artery disease. The patient presents today stating that his chest pain started yesterday evening and has been somewhat intermittent. The severity of the pain has progressively increased. He describes the pain as a sharp and heavy pain which radiates to his neck & left arm. He ranks the pain a 7 on a scale of 1-10. He admits some shortness of breath & diaphoresis. He states that he has had nausea & 3 episodes of vomiting tonight. He denies any fever or chills. He admits prior episodes of similar pain prior to his PTCA in 1995. He states the pain is somewhat worse with walking and seems to be relieved with rest. There is no change in pain with positioning. He states that he took 3 nitroglycerin tablets sublingually over the past 1 hour, which he states has partially relieved his pain. The patient ranks his present pain a 4 on a scale of 1-10. The most recent episode of pain has lasted one-hour.The patient denies any history of recent surgery, head trauma, recent stroke, abnormal bleeding such as blood in urine or stool or nosebleed.REVIEW OF SYSTEMS: All other systems reviewed & are negative.PAST MEDICAL HISTORY: Diabetes mellitus type II, hypertension, coronary artery disease, atrial fibrillation, status post PTCA in 1995 by Dr. ABC.SOCIAL HISTORY: Denies alcohol or drugs. Smokes 2 packs of cigarettes per day. Works as a banker.FAMILY HISTORY: Positive for coronary artery disease (father & brother)."""].toDS.toDF("text")).transform(data)

```

</div>

{:.h2_title}
## Results
The output is a dataframe with a sentence per row and a ``"ner"`` column containing all of the entity labels in the sentence, entity character indices, and other metadata. To get only the tokens and entity labels, without the metadata, select ``"token.result"`` and ``"ner.result"`` from your output dataframe or add the ``"Finisher"`` to the end of your pipeline.

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
|Model Name:|ner_risk_factors_en_2.4.2_2.4|
|Type:|ner|
|Compatibility:|Spark NLP 2.4.2|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence,token, embeddings]|
|Output Labels:|[ner]|
|Language:|[en]|
|Case sensitive:|false|
| Dependencies:  | embeddings_clinical              |


{:.h2_title}
## Data Source
Trained on plain n2c2 2014: De-identification and Heart Disease Risk Factors Challenge datasets with ``embeddings_clinical``.
https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/

{:.h2_title}
## Benchmarking
```bash
|    | label            |    tp |    fp |    fn |     prec |      rec |       f1 |
|---:|-----------------:|------:|------:|------:|---------:|---------:|---------:|
|  0 | B-SMOKER         |    48 |    39 |    61 | 0.551724 | 0.440367 | 0.489796 |
|  1 | I-HYPERLIPIDEMIA |     7 |     7 |     7 | 0.5      | 0.5      | 0.5      |
|  2 | I-HYPERTENSION   |    12 |    12 |    89 | 0.5      | 0.118812 | 0.192    |
|  3 | B-CAD            |   104 |    52 |   101 | 0.666667 | 0.507317 | 0.576177 |
|  4 | I-DIABETES       |   127 |    67 |    92 | 0.654639 | 0.579909 | 0.615012 |
|  5 | B-FAMILY_HIST    |     0 |     0 |     1 | 0        | 0        | 0        |
|  6 | B-HYPERTENSION   |   173 |    52 |    64 | 0.768889 | 0.729958 | 0.748918 |
|  7 | B-OBESE          |    46 |    20 |     3 | 0.69697  | 0.938776 | 0.8      |
|  8 | B-PHI            |  1968 |   599 |   252 | 0.766654 | 0.886486 | 0.822227 |
|  9 | B-HYPERLIPIDEMIA |    71 |    17 |    14 | 0.806818 | 0.835294 | 0.820809 |
| 10 | I-CAD            |   122 |   111 |   343 | 0.523605 | 0.262366 | 0.34957  |
| 11 | I-SMOKER         |   116 |    73 |    94 | 0.613757 | 0.552381 | 0.581454 |
| 12 | I-OBESE          |     9 |     8 |     4 | 0.529412 | 0.692308 | 0.6      |
| 13 | I-FAMILY_HIST    |     5 |     0 |    10 | 1        | 0.333333 | 0.5      |
| 14 | B-DIABETES       |   190 |    59 |    58 | 0.763052 | 0.766129 | 0.764587 |
| 15 | B-MEDICATION     |   838 |   224 |    81 | 0.789077 | 0.911861 | 0.846037 |
| 16 | I-MEDICATION     |   100 |   242 |   190 | 0.292398 | 0.344828 | 0.316456 |
| 17 | I-PHI            |   597 |   202 |   136 | 0.747184 | 0.814461 | 0.779373 |
| 18 | Macro-average    |  4533 |  1784 |  1600 | 0.620602 | 0.567477 | 0.592852 |
| 19 | Micro-average    |  4533 |  1784 |  1600 | 0.717588 | 0.739116 | 0.728193 |
```