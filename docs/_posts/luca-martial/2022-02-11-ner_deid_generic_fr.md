---
layout: model
title: Detect PHI for Deidentification purposes (French, reduced entities)
author: John Snow Labs
name: ner_deid_generic
date: 2022-02-11
tags: [deid, fr, licensed]
task: De-identification
language: fr
edition: Healthcare NLP 3.4.1
spark_version: 2.4
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Named Entity Recognition annotators allow for a generic model to be trained by using a Deep Learning architecture (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM, CNN.


Deidentification NER (French) is a Named Entity Recognition model that annotates text to find protected health information that may need to be de-identified. It detects 7 entities. This NER model is trained with a custom dataset internally annotated, the [French WikiNER dataset](https://metatext.io/datasets/wikiner), a [public dataset of French company names](https://www.data.gouv.fr/fr/datasets/entreprises-immatriculees-en-2017/), a [public dataset of French hospital names](https://salesdorado.com/fichiers-prospection/hopitaux/) and several data augmentation mechanisms.


## Predicted Entities


`CONTACT`, `NAME`, `DATE`, `ID`, `LOCATION`, `PROFESSION`, `AGE`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_generic_fr_3.4.1_2.4_1644591444704.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


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


clinical_ner = MedicalNerModel.pretrained("ner_deid_generic", "fr", "clinical/models")\
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
.setInputCols(Array("document"))
.setOutputCol("sentence")


val tokenizer = new Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")


val embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d", "fr")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")


val clinical_ner = MedicalNerModel.pretrained("ner_deid_generic", "fr", "clinical/models")
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
nlu.load("fr.med_ner.deid_generic").predict("""J'ai vu en consultation Michel Martinez (49 ans) adressé au Centre Hospitalier De Plaisir pour un diabète mal contrôlé avec des symptômes datant de Mars 2015.""")
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
|      Michel|    B-NAME|
|    Martinez|    I-NAME|
|           (|         O|
|          49|     B-AGE|
|         ans|     I-AGE|
|           )|         O|
|     adressé|         O|
|          au|         O|
|      Centre|B-LOCATION|
| Hospitalier|I-LOCATION|
|          De|I-LOCATION|
|     Plaisir|I-LOCATION|
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
|Model Name:|ner_deid_generic|
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
CONTACT   159.0    0.0    1.0   160.0        1.0  0.9938  0.9969
NAME  2633.0  111.0  197.0  2830.0     0.9595  0.9304  0.9447
DATE  2612.0   32.0   42.0  2654.0     0.9879  0.9842   0.986
ID    95.0    8.0    7.0   102.0     0.9223  0.9314  0.9268
LOCATION  3450.0  480.0  522.0  3972.0     0.8779  0.8686  0.8732
PROFESSION   326.0   54.0   82.0   408.0     0.8579   0.799  0.8274
AGE   395.0   37.0   46.0   441.0     0.9144  0.8957  0.9049
macro       -      -      -       -          -       -  0.9229
micro       -      -      -       -          -       -  0.9226
```
