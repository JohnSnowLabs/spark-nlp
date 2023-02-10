---
layout: model
title: Detect PHI for Deidentification purposes (French)
author: John Snow Labs
name: ner_deid_subentity
date: 2022-02-14
tags: [deid, fr, licensed]
task: De-identification
language: fr
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Named Entity Recognition annotators allow for a generic model to be trained by using a Deep Learning architecture (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM, CNN.


Deidentification NER (French) is a Named Entity Recognition model that annotates text to find protected health information that may need to be de-identified. It detects 15 entities. This NER model is trained with a custom dataset internally annotated, the [French WikiNER dataset](https://metatext.io/datasets/wikiner), a [public dataset of French company names](https://www.data.gouv.fr/fr/datasets/entreprises-immatriculees-en-2017/), a [public dataset of French hospital names](https://salesdorado.com/fichiers-prospection/hopitaux/) and several data augmentation mechanisms.


## Predicted Entities


`PATIENT`, `HOSPITAL`, `DATE`, `ORGANIZATION`, `E-MAIL`, `USERNAME`, `ZIP`, `MEDICALRECORD`, `PROFESSION`, `PHONE`, `DOCTOR`, `AGE`, `STREET`, `CITY`, `COUNTRY`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_fr_3.4.1_3.0_1644838067533.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_fr_3.4.1_3.0_1644838067533.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use




<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d", "fr")\
	.setInputCols(["sentence", "token"])\
	.setOutputCol("word_embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_deid_subentity", "fr", "clinical/models")\
.setInputCols(["sentence","token", "word_embeddings"])\
.setOutputCol("ner")

nlpPipeline = Pipeline(stages=[
documentAssembler,
sentenceDetector,
tokenizer,
embeddings,
clinical_ner])

text = ["J'ai vu en consultation Michel Martinez (49 ans) adressé au Centre Hospitalier De Plaisir pour un diabète mal contrôlé avec des symptômes datant de Mars 2015."]

data = spark.createDataFrame([text]).toDF("text")

results = nlpPipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
.setInputCols("document")
.setOutputCol("sentence")

val tokenizer = new Tokenizer()
.setInputCols("sentence")
.setOutputCol("token")

val embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d", "fr")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_deid_subentity", "fr", "clinical/models")
.setInputCols(Array("sentence","token","embeddings"))
.setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, embeddings, clinical_ner))

val text = "J'ai vu en consultation Michel Martinez (49 ans) adressé au Centre Hospitalier De Plaisir pour un diabète mal contrôlé avec des symptômes datant de Mars 2015."

val data = Seq(text).toDF("text")

val results = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("fr.med_ner.deid_subentity").predict("""J'ai vu en consultation Michel Martinez (49 ans) adressé au Centre Hospitalier De Plaisir pour un diabète mal contrôlé avec des symptômes datant de Mars 2015.""")
```

</div>


## Results


```bash
+------------+----------+
|       token| ner_label|
+------------+----------+
|        J'ai|         O|
|          vu|         O|
|          en|         O|
|consultation|         O|
|      Michel| B-PATIENT|
|    Martinez| I-PATIENT|
|           (|         O|
|          49|     B-AGE|
|         ans|         O|
|           )|         O|
|     adressé|         O|
|          au|         O|
|      Centre|B-HOSPITAL|
| Hospitalier|I-HOSPITAL|
|          De|I-HOSPITAL|
|     Plaisir|I-HOSPITAL|
|        pour|         O|
|          un|         O|
|     diabète|         O|
|         mal|         O|
|    contrôlé|         O|
|        avec|         O|
|         des|         O|
|   symptômes|         O|
|      datant|         O|
|          de|         O|
|        Mars|    B-DATE|
|        2015|    I-DATE|
|           .|         O|
+------------+----------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_deid_subentity|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|fr|
|Size:|15.0 MB|


## References


- Internal JSL annotated corpus
- [French WikiNER dataset](https://metatext.io/datasets/wikiner)
- [Public dataset of French company names](https://www.data.gouv.fr/fr/datasets/entreprises-immatriculees-en-2017/)
- [Public dataset of French hospital names](https://salesdorado.com/fichiers-prospection/hopitaux/)


## Benchmarking


```bash
label      tp     fp     fn   total  precision  recall      f1
PATIENT  1966.0  124.0  135.0  2101.0     0.9407  0.9357  0.9382
HOSPITAL   315.0   23.0   19.0   334.0      0.932  0.9431  0.9375
DATE  2605.0   31.0   49.0  2654.0     0.9882  0.9815  0.9849
ORGANIZATION   503.0  142.0  159.0   662.0     0.7798  0.7598  0.7697
CITY  2296.0  370.0  351.0  2647.0     0.8612  0.8674  0.8643
MAIL    46.0    0.0    0.0    46.0        1.0     1.0     1.0
STREET    31.0    4.0    3.0    34.0     0.8857  0.9118  0.8986
USERNAME    91.0    1.0   14.0   105.0     0.9891  0.8667  0.9239
ZIP    33.0    0.0    0.0    33.0        1.0     1.0     1.0
MEDICALRECORD   100.0   11.0    2.0   102.0     0.9009  0.9804   0.939
PROFESSION   321.0   59.0   87.0   408.0     0.8447  0.7868  0.8147
PHONE   114.0    3.0    2.0   116.0     0.9744  0.9828  0.9785
COUNTRY   287.0   14.0   51.0   338.0     0.9535  0.8491  0.8983
DOCTOR   622.0    7.0    4.0   626.0     0.9889  0.9936  0.9912
AGE   370.0   52.0   71.0   441.0     0.8768   0.839  0.8575
macro       -      -      -       -          -       -  0.9197
micro       -      -      -       -          -       -  0.9154
```
