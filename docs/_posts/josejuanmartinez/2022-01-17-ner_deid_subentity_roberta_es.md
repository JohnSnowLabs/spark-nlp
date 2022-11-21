---
layout: model
title: Detect PHI for Deidentification purposes (Spanish, Roberta embeddings)
author: John Snow Labs
name: ner_deid_subentity_roberta
date: 2022-01-17
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


Deidentification NER (Spanish) is a Named Entity Recognition model that annotates text to find protected health information that may need to be de-identified. It detects 13 entities. This NER model is trained with a combination of custom datasets, Spanish 2002 conLL, MeddoProf dataset and several data augmentation mechanisms. This model uses Roberta Clinical Embeddings.


## Predicted Entities


`PATIENT`, `HOSPITAL`, `DATE`, `ORGANIZATION`, `E-MAIL`, `USERNAME`, `LOCATION`, `ZIP`, `MEDICALRECORD`, `PROFESSION`, `PHONE`, `DOCTOR`, `AGE`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_roberta_es_3.3.4_3.0_1642428102794.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

# Feel free to experiment with multilingual or Spanish nlp.SentenceDetector instead
sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

roberta_embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_base_biomedical", "es")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = medical.NerModel.pretrained("ner_deid_subentity_roberta", "es", "clinical/models")\
        .setInputCols(["sentence","token","embeddings"])\
        .setOutputCol("ner")

nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        roberta_embeddings,
        clinical_ner])

text = ['''
Antonio Pérez Juan, nacido en Cadiz, España. Aún no estaba vacunado, se infectó con Covid-19 el dia 14 de Marzo y tuvo que ir al Hospital. Fue tratado con anticuerpos monoclonales en la Clinica San Carlos.
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

val clinical_ner = MedicalNerModel.pretrained("ner_deid_subentity_roberta", "es", "clinical/models")
        .setInputCols(Array("sentence","token","embeddings"))
        .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, roberta_embeddings, clinical_ner))

val text = """Antonio Pérez Juan, nacido en Cadiz, España. Aún no estaba vacunado, se infectó con Covid-19 el dia 14 de Marzo y tuvo que ir al Hospital. Fue tratado con anticuerpos monoclonales en la Clinica San Carlos."""

val df = Seq(text).toDS.toDF("text")

val results = pipeline.fit(df).transform(df)
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
|Model Name:|ner_deid_subentity_roberta|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|16.3 MB|
|Dependencies:|roberta_base_biomedical|


## Data Source


- Internal JSL annotated corpus
- [Spanish conLL](https://www.clips.uantwerpen.be/conll2002/ner/data/)
- [MeddoProf](https://temu.bsc.es/meddoprof/data/)


## Benchmarking


```bash
        label      tp     fp     fn   total  precision  recall      f1
      PATIENT  1946.0  157.0  213.0  2159.0     0.9253  0.9013  0.9132
     HOSPITAL   272.0   82.0   87.0   359.0     0.7684  0.7577   0.763
         DATE  1632.0   24.0   35.0  1667.0     0.9855   0.979  0.9822
 ORGANIZATION  2460.0  479.0  513.0  2973.0      0.837  0.8274  0.8322
         MAIL    58.0    0.0    0.0    58.0        1.0     1.0     1.0
     USERNAME    95.0    1.0   10.0   105.0     0.9896  0.9048  0.9453
     LOCATION  1734.0  416.0  381.0  2115.0     0.8065  0.8199  0.8131
          ZIP    13.0    0.0    4.0    17.0        1.0  0.7647  0.8667
MEDICALRECORD   111.0   11.0   10.0   121.0     0.9098  0.9174  0.9136
   PROFESSION   273.0   72.0  116.0   389.0     0.7913  0.7018  0.7439
        PHONE   108.0   12.0    8.0   116.0        0.9   0.931  0.9153
       DOCTOR   641.0   32.0   46.0   687.0     0.9525   0.933  0.9426
          AGE   284.0   37.0   64.0   348.0     0.8847  0.8161   0.849
        macro     -      -      -       -         -       -    0.88308
        micro     -      -      -       -         -       -    0.87258
```
