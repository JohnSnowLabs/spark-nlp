---
layout: model
title: Detect PHI for Deidentification purposes (Spanish)
author: John Snow Labs
name: ner_deid_subentity
date: 2022-01-18
tags: [deid, es, licensed]
task: De-identification
language: es
edition: Healthcare NLP 3.3.4
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Named Entity Recognition annotators allow for a generic model to be trained by using a Deep Learning architecture (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN. 


Deidentification NER (Spanish) is a Named Entity Recognition model that annotates text to find protected health information that may need to be de-identified. It detects 13 entities. This NER model is trained with a combination of custom datasets, Spanish 2002 conLL, MeddoProf dataset and several data augmentation mechanisms.


## Predicted Entities


`PATIENT`, `HOSPITAL`, `DATE`, `ORGANIZATION`, `E-MAIL`, `USERNAME`, `LOCATION`, `ZIP`, `MEDICALRECORD`, `PROFESSION`, `PHONE`, `DOCTOR`, `AGE`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_es_3.3.4_3.0_1642512189785.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_es_3.3.4_3.0_1642512189785.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")

embeddings = nlp.WordEmbeddingsModel.pretrained("embeddings_sciwiki_300d","es","clinical/models")\
	.setInputCols(["sentence","token"])\
	.setOutputCol("word_embeddings")

clinical_ner = medical.NerModel.pretrained("ner_deid_subentity", "es", "clinical/models")\
.setInputCols(["sentence","token","word_embeddings"])\
.setOutputCol("ner")

nlpPipeline = Pipeline(stages=[
documentAssembler,
sentenceDetector,
tokenizer,
embeddings,
clinical_ner])

text = ['''
Antonio Pérez Juan, nacido en Cadiz, España. Aún no estaba vacunado, se infectó con Covid-19 el dia 14/03/2020 y tuvo que ir al Hospital. Fue tratado con anticuerpos monoclonales en la Clinica San Carlos.
''']

data = spark.createDataFrame([text]).toDF("text")

results = nlpPipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","xx")
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = new Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")

val embeddings = WordEmbeddingsModel.pretrained("embeddings_sciwiki_300d","es","clinical/models")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_deid_subentity", "es", "clinical/models")
.setInputCols(Array("sentence","token","embeddings"))
.setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, embeddings, clinical_ner))

val text = """Antonio Pérez Juan, nacido en Cadiz, España. Aún no estaba vacunado, se infectó con Covid-19 el dia 14/03/2020 y tuvo que ir al Hospital. Fue tratado con anticuerpos monoclonales en la Clinica San Carlos."""

val data = Seq(text).toDS.toDF("text")

val results = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("es.med_ner.deid.subentity").predict("""
Antonio Pérez Juan, nacido en Cadiz, España. Aún no estaba vacunado, se infectó con Covid-19 el dia 14/03/2020 y tuvo que ir al Hospital. Fue tratado con anticuerpos monoclonales en la Clinica San Carlos.
""")
```

</div>


## Results


```bash
+------------+----------+
|       token| ner_label|
+------------+----------+
|     Antonio| B-PATIENT|
|       Pérez| I-PATIENT|
|        Juan| I-PATIENT|
|           ,|         O|
|      nacido|         O|
|          en|         O|
|       Cadiz|B-LOCATION|
|           ,|         O|
|      España|B-LOCATION|
|           .|         O|
|         Aún|         O|
|          no|         O|
|      estaba|         O|
|    vacunado|         O|
|           ,|         O|
|          se|         O|
|     infectó|         O|
|         con|         O|
|    Covid-19|         O|
|          el|         O|
|         dia|         O|
|          14|    B-DATE|
|          de|    I-DATE|
|       Marzo|    I-DATE|
|           y|         O|
|        tuvo|         O|
|         que|         O|
|          ir|         O|
|          al|         O|
|    Hospital|         O|
|         Fue|         O|
|     tratado|         O|
|         con|         O|
| anticuerpos|         O|
|monoclonales|         O|
|          en|         O|
|          la|         O|
|     Clinica|B-HOSPITAL|
|         San|I-HOSPITAL|
|      Carlos|I-HOSPITAL|
|           .|         O|
+------------+----------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_deid_subentity|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, word_embeddings]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|15.0 MB|
|Dependencies:|embeddings_sciwiki_300d|


## Data Source


- Internal JSL annotated corpus
- [Spanish conLL](https://www.clips.uantwerpen.be/conll2002/ner/data/)
- [MeddoProf](https://temu.bsc.es/meddoprof/data/)


## Benchmarking


```bash
label      tp     fp     fn   total  precision  recall      f1
PATIENT  2088.0  201.0  178.0  2266.0     0.9122  0.9214  0.9168
HOSPITAL   302.0   43.0   85.0   387.0     0.8754  0.7804  0.8251
DATE  1837.0   33.0   20.0  1857.0     0.9824  0.9892  0.9858
ORGANIZATION  2498.0  477.0  649.0  3147.0     0.8397  0.7938  0.8161
MAIL    58.0    0.0    0.0    58.0        1.0     1.0     1.0
USERNAME    90.0    0.0   15.0   105.0        1.0  0.8571  0.9231
LOCATION  1866.0  391.0  354.0  2220.0     0.8268  0.8405  0.8336
ZIP    20.0    1.0    2.0    22.0     0.9524  0.9091  0.9302
MEDICALRECORD   111.0    5.0   20.0   131.0     0.9569  0.8473  0.8988
PROFESSION   270.0   96.0  134.0   404.0     0.7377  0.6683  0.7013
PHONE   108.0   11.0    8.0   116.0     0.9076   0.931  0.9191
DOCTOR   659.0   40.0   40.0   699.0     0.9428  0.9428  0.9428
AGE   302.0   53.0   61.0   363.0     0.8507   0.832  0.8412
macro     -      -      -       -         -      -      0.8872247
micro     -      -      -       -         -      -      0.8741892
```
