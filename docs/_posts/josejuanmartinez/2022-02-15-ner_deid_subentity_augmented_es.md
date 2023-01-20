---
layout: model
title: Detect PHI for Deidentification purposes (Spanish, augmented)
author: John Snow Labs
name: ner_deid_subentity_augmented
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


Deidentification NER (Spanish) is a Named Entity Recognition model that annotates text to find protected health information that may need to be de-identified. It detects 17 entities, which is more than the previously released `ner_deid_subentity` model.


This NER model is trained with a combination of custom datasets, Spanish 2002 conLL, MeddoProf and MeddoCan datasets, and includes several data augmentation mechanisms.


## Predicted Entities


`PATIENT`, `HOSPITAL`, `DATE`, `ORGANIZATION`, `CITY`, `ID`, `STREET`, `USERNAME`, `SEX`, `EMAIL`, `ZIP`, `MEDICALRECORD`, `PROFESSION`, `PHONE`, `COUNTRY`, `DOCTOR`, `AGE`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_augmented_es_3.3.4_2.4_1644927080275.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_augmented_es_3.3.4_2.4_1644927080275.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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


clinical_ner = medical.NerModel.pretrained("ner_deid_subentity_augmented", "es", "clinical/models")\
        .setInputCols(["sentence","token","word_embeddings"])\
        .setOutputCol("ner")


nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
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


val embeddings = WordEmbeddingsModel.pretrained("embeddings_sciwiki_300d","es","clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")


val clinical_ner = MedicalNerModel.pretrained("ner_deid_subentity_augmented", "es", "clinical/models")
        .setInputCols(Array("sentence","token","embeddings"))
        .setOutputCol("ner")


val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, embeddings, clinical_ner))


val text = "Antonio Miguel Martínez, varón de de 35 años de edad, de profesión auxiliar de enfermería y nacido en Cadiz, España. Aún no estaba vacunado, se infectó con Covid-19 el dia 14 de Marzo y tuvo que ir al Hospital. Fue tratado con anticuerpos monoclonales en la Clinica San Carlos."


val df = Seq(text).toDF("text")


val results = pipeline.fit(df).transform(df)
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
|Model Name:|ner_deid_subentity_augmented|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, word_embeddings]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|15.0 MB|


## References


- Internal JSL annotated corpus
- [Spanish conLL](https://www.clips.uantwerpen.be/conll2002/ner/data/)
- [MeddoProf](https://temu.bsc.es/meddoprof/data/)
- [MeddoCan](https://temu.bsc.es/meddocan/)


## Benchmarking


```bash
        label      tp     fp     fn   total  precision  recall      f1
      PATIENT  2022.0  224.0  140.0  2162.0     0.9003  0.9352  0.9174
     HOSPITAL   259.0   35.0   50.0   309.0      0.881  0.8382   0.859
         DATE  1023.0   12.0   12.0  1035.0     0.9884  0.9884  0.9884
 ORGANIZATION  2624.0  516.0  544.0  3168.0     0.8357  0.8283   0.832
         CITY  1561.0  339.0  266.0  1827.0     0.8216  0.8544  0.8377
           ID    36.0    1.0    3.0    39.0      0.973  0.9231  0.9474
       STREET   197.0   14.0    9.0   206.0     0.9336  0.9563  0.9448
     USERNAME    10.0    6.0    1.0    11.0      0.625  0.9091  0.7407
          SEX   682.0   13.0   11.0   693.0     0.9813  0.9841  0.9827
        EMAIL   134.0    0.0    1.0   135.0        1.0  0.9926  0.9963
          ZIP   141.0    2.0    1.0   142.0      0.986   0.993  0.9895
MEDICALRECORD    29.0    5.0    0.0    29.0     0.8529     1.0  0.9206
   PROFESSION   252.0   27.0   25.0   277.0     0.9032  0.9097  0.9065
        PHONE    51.0   11.0    0.0    51.0     0.8226     1.0  0.9027
      COUNTRY   505.0   74.0   82.0   587.0     0.8722  0.8603  0.8662
       DOCTOR   444.0   26.0   48.0   492.0     0.9447  0.9024  0.9231
          AGE   549.0   15.0    7.0   556.0     0.9734  0.9874  0.9804
        macro       -      -      -       -          -       -  0.9138
        micro       -      -      -       -          -       -  0.8930
```
