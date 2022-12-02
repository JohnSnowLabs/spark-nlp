---
layout: model
title: Detect PHI for Deidentification purposes (Spanish, Roberta, augmented)
author: John Snow Labs
name: ner_deid_subentity_roberta_augmented
date: 2022-02-15
tags: [deid, es, licensed]
task: De-identification
language: es
edition: Healthcare NLP 3.3.4
spark_version: 2.4
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Named Entity Recognition annotators allow for a generic model to be trained by using a Deep Learning architecture (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN. 


Deidentification NER (Spanish) is a Named Entity Recognition model that annotates text to find protected health information that may need to be de-identified. It detects 17 entities, which is more than the previously released `ner_deid_subentity_roberta` model.


This NER model is trained with a combination of custom datasets, Spanish 2002 conLL, MeddoProf and MeddoCan datasets, and includes several data augmentation mechanisms.


This is a version that includes Roberta Clinical embeddings. You can find as well `ner_deid_subentity_augmented` that uses Sciwi 300d embeddings based instead of Roberta.


## Predicted Entities


`PATIENT`, `HOSPITAL`, `DATE`, `ORGANIZATION`, `CITY`, `ID`, `STREET`, `USERNAME`, `SEX`, `EMAIL`, `ZIP`, `MEDICALRECORD`, `PROFESSION`, `PHONE`, `COUNTRY`, `DOCTOR`, `AGE`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_roberta_augmented_es_3.3.4_2.4_1644927666923.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


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


roberta_embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_base_biomedical", "es")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")


clinical_ner = medical.NerModel.pretrained("ner_deid_subentity_roberta_augmented", "es", "clinical/models")\
.setInputCols(["sentence","token","embeddings"])\
.setOutputCol("ner")


nlpPipeline = Pipeline(stages=[
documentAssembler,
sentenceDetector,
tokenizer,
roberta_embeddings,
clinical_ner])


text = ['''
Antonio Miguel Martínez, varón de de 35 años de edad, de profesión auxiliar de enfermería y nacido en Cadiz, España. Aún no estaba vacunado, se infectó con Covid-19 el dia 14 de Marzo y tuvo que ir al Hospital. Fue tratado con anticuerpos monoclonales en la Clinica San Carlos.
''']


df = spark.createDataFrame([text]).toDF("text")


results = nlpPipeline.fit(df).transform(df)
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


val roberta_embeddings = RoBertaEmbeddings.pretrained("roberta_base_biomedical", "es")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")


val clinical_ner = MedicalNerModel.pretrained("ner_deid_subentity_roberta_augmented", "es", "clinical/models")
.setInputCols(Array("sentence","token","embeddings"))
.setOutputCol("ner")


val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, roberta_embeddings, clinical_ner))


val text = "Antonio Miguel Martínez, varón de de 35 años de edad, de profesión auxiliar de enfermería y nacido en Cadiz, España. Aún no estaba vacunado, se infectó con Covid-19 el dia 14 de Marzo y tuvo que ir al Hospital. Fue tratado con anticuerpos monoclonales en la Clinica San Carlos."


val df = Seq(text).toDF("text")


val results = pipeline.fit(df).transform(df)
```


{:.nlu-block}
```python
import nlu
nlu.load("es.med_ner.deid.subentity.roberta").predict("""
Antonio Miguel Martínez, varón de de 35 años de edad, de profesión auxiliar de enfermería y nacido en Cadiz, España. Aún no estaba vacunado, se infectó con Covid-19 el dia 14 de Marzo y tuvo que ir al Hospital. Fue tratado con anticuerpos monoclonales en la Clinica San Carlos.
""")
```

</div>


## Results


```bash
+------------+------------+
|       token|   ner_label|
+------------+------------+
|     Antonio|   B-PATIENT|
|      Miguel|   I-PATIENT|
|    Martínez|   I-PATIENT|
|           ,|           O|
|       varón|       B-SEX|
|          de|           O|
|          de|           O|
|          35|       B-AGE|
|        años|           O|
|          de|           O|
|        edad|           O|
|           ,|           O|
|          de|           O|
|   profesión|           O|
|    auxiliar|B-PROFESSION|
|          de|I-PROFESSION|
|  enfermería|I-PROFESSION|
|           y|           O|
|      nacido|           O|
|          en|           O|
|       Cadiz|      B-CITY|
|           ,|           O|
|      España|   B-COUNTRY|
|           .|           O|
|         Aún|           O|
|          no|           O|
|      estaba|           O|
|    vacunado|           O|
|           ,|           O|
|          se|           O|
|     infectó|           O|
|         con|           O|
|    Covid-19|           O|
|          el|           O|
|         dia|           O|
|          14|      B-DATE|
|          de|      I-DATE|
|       Marzo|      I-DATE|
|           y|           O|
|        tuvo|           O|
|         que|           O|
|          ir|           O|
|          al|           O|
|    Hospital|           O|
|         Fue|           O|
|     tratado|           O|
|         con|           O|
| anticuerpos|           O|
|monoclonales|           O|
|          en|           O|
|          la|           O|
|     Clinica|  B-HOSPITAL|
|         San|  I-HOSPITAL|
|      Carlos|  I-HOSPITAL|
|           .|           O|
+------------+------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_deid_subentity_roberta_augmented|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|16.3 MB|


## References


- Internal JSL annotated corpus
- [Spanish conLL](https://www.clips.uantwerpen.be/conll2002/ner/data/)
- [MeddoProf](https://temu.bsc.es/meddoprof/data/)
- [MeddoCan](https://temu.bsc.es/meddocan/)


## Benchmarking


```bash
label      tp     fp     fn   total  precision  recall      f1
PATIENT  1874.0  165.0  186.0  2060.0     0.9191  0.9097  0.9144
HOSPITAL   241.0   19.0   54.0   295.0     0.9269  0.8169  0.8685
DATE   954.0   17.0   15.0   969.0     0.9825  0.9845  0.9835
ORGANIZATION  2521.0  483.0  468.0  2989.0     0.8392  0.8434  0.8413
CITY  1464.0  369.0  289.0  1753.0     0.7987  0.8351  0.8165
ID    35.0    1.0    0.0    35.0     0.9722     1.0  0.9859
STREET   194.0    8.0    6.0   200.0     0.9604    0.97  0.9652
USERNAME     7.0    0.0    4.0    11.0        1.0  0.6364  0.7778
SEX   618.0    9.0    9.0   627.0     0.9856  0.9856  0.9856
EMAIL   134.0    0.0    0.0   134.0        1.0     1.0     1.0
ZIP   138.0    0.0    1.0   139.0        1.0  0.9928  0.9964
MEDICALRECORD    29.0   10.0    0.0    29.0     0.7436     1.0  0.8529
PROFESSION   231.0   20.0   30.0   261.0     0.9203  0.8851  0.9023
PHONE    44.0    0.0    6.0    50.0        1.0    0.88  0.9362
COUNTRY   458.0   96.0  103.0   561.0     0.8267  0.8164  0.8215
DOCTOR   432.0   38.0   48.0   480.0     0.9191     0.9  0.9095
AGE   509.0    9.0   10.0   519.0     0.9826  0.9807  0.9817
macro       -      -      -       -          -       -  0.9141
micro       -      -      -       -          -       -  0.8891
```
