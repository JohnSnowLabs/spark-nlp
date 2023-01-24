---
layout: model
title: Detect PHI for Deidentification (Sub Entity)
author: John Snow Labs
name: ner_deid_subentity
date: 2022-01-06
tags: [de, deid, ner, licensed]
task: Named Entity Recognition
language: de
edition: Healthcare NLP 3.3.4
spark_version: 2.4
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM, CNN. Deidentification NER is a Named Entity Recognition model that annotates German text to find protected health information (PHI) that may need to be deidentified. It was trained with in-house annotations and detects 12 entities.


## Predicted Entities


`PATIENT`, `HOSPITAL`, `DATE`, `ORGANIZATION`, `CITY`, `STREET`, `USERNAME`, `PROFESSION`, `PHONE`, `COUNTRY`, `DOCTOR`, `AGE`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DEID_DE){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.1.Clinical_Deidentification_in_German.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_de_3.3.4_2.4_1641460993460.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_de_3.3.4_2.4_1641460993460.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
document_assembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentence_detector = SentenceDetector()\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","de","clinical/models")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")

deid_ner = MedicalNerModel.pretrained("ner_deid_subentity", "de", "clinical/models")\
.setInputCols(["sentence", "token", "embeddings"])\
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(["sentence", "token", "ner"])\
.setOutputCol("ner_deid_subentity_chunk")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, deid_ner, ner_converter])

data = spark.createDataFrame([["""Michael Berger wird am Morgen des 12 Dezember 2018 ins St. Elisabeth-Krankenhaus
in Bad Kissingen eingeliefert. Herr Berger ist 76 Jahre alt und hat zu viel Wasser in den Beinen."""]]).toDF("text")

result = nlpPipeline.fit(data).transform(data)
```
```scala
...
val document_assembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val sentence_detector = new SentenceDetector()
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = new Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d", "de", "clinical/models")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")

val deid_ner = MedicalNerModel.pretrained("ner_deid_subentity", "de", "clinical/models") 
.setInputCols(Array("sentence", "token", "embeddings")) 
.setOutputCol("ner")

val ner_converter = new NerConverter()
.setInputCols(Array("sentence", "token", "ner"))
.setOutputCol("ner_deid_subentity_chunk")

val nlpPipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, deid_ner, ner_converter))

val data = Seq("""Michael Berger wird am Morgen des 12 Dezember 2018 ins St. Elisabeth-Krankenhausin Bad Kissingen eingeliefert. Herr Berger ist 76 Jahre alt und hat zu viel Wasser in den Beinen.""").toDS.toDF("text")

val result = nlpPipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("de.med_ner.deid_subentity").predict("""Michael Berger wird am Morgen des 12 Dezember 2018 ins St. Elisabeth-Krankenhaus
in Bad Kissingen eingeliefert. Herr Berger ist 76 Jahre alt und hat zu viel Wasser in den Beinen.""")
```

</div>


## Results


```bash
+-------------------------+-------------------------+
|chunk                    |ner_deid_subentity_chunk |
+-------------------------+-------------------------+
|Michael Berger           |PATIENT                  |
|12 Dezember 2018         |DATE                     |
|St. Elisabeth-Krankenhaus|HOSPITAL                 |
|Bad Kissingen            |CITY                     |
|Berger                   |PATIENT                  |
|76                       |AGE                      |
+-------------------------+-------------------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_deid_subentity|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|de|
|Size:|15.0 MB|


## Data Source


In-house annotated dataset


## Benchmarking


```bash
label      tp     fp    fn   total  precision  recall      f1
PATIENT  2080.0   58.0  74.0  2154.0     0.9729  0.9656  0.9692
HOSPITAL  1598.0    4.0   0.0  1598.0     0.9975     1.0  0.9988
DATE  4047.0    7.0   2.0  4049.0     0.9983  0.9995  0.9989
ORGANIZATION  1288.0  108.0  67.0  1355.0     0.9226  0.9506  0.9364
CITY   196.0    1.0   4.0   200.0     0.9949    0.98  0.9874
STREET   124.0    1.0   4.0   128.0      0.992  0.9688  0.9802
USERNAME    45.0    0.0   0.0    45.0        1.0     1.0     1.0
PROFESSION   262.0    1.0   0.0   262.0     0.9962     1.0  0.9981
PHONE    71.0   10.0   9.0    80.0     0.8765  0.8875   0.882
COUNTRY   306.0    5.0   6.0   312.0     0.9839  0.9808  0.9823
DOCTOR  1414.0    9.0  39.0  1453.0     0.9937  0.9732  0.9833
AGE   473.0    3.0   3.0   476.0     0.9937  0.9937  0.9937
```
