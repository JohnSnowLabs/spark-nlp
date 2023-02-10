---
layout: model
title: Detect PHI for Deidentification (Generic)
author: John Snow Labs
name: ner_deid_generic
date: 2022-01-06
tags: [deid, ner, de, licensed]
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


Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM, CNN. Deidentification NER is a Named Entity Recognition model that annotates German text to find protected health information (PHI) that may need to be deidentified. It was trained with in-house annotations and detects 7 entities.


## Predicted Entities


`DATE`, `NAME`, `LOCATION`, `PROFESSION`, `AGE`, `ID`, `CONTACT`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DEID_DE){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.1.Clinical_Deidentification_in_German.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_generic_de_3.3.4_2.4_1641460977185.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deid_generic_de_3.3.4_2.4_1641460977185.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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

deid_ner = MedicalNerModel.pretrained("ner_deid_generic", "de", "clinical/models")\
.setInputCols(["sentence", "token", "embeddings"])\
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(["sentence", "token", "ner"])\
.setOutputCol("ner_deid_generic_chunk")

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

val deid_ner = MedicalNerModel.pretrained("ner_deid_generic", "de", "clinical/models") 
.setInputCols(Array("sentence", "token", "embeddings")) 
.setOutputCol("ner")

val ner_converter = new NerConverter()
.setInputCols(Array("sentence", "token", "ner"))
.setOutputCol("ner_deid_generic_chunk")

val nlpPipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, deid_ner, ner_converter))

val data = Seq("""Michael Berger wird am Morgen des 12 Dezember 2018 ins St. Elisabeth-Krankenhausin Bad Kissingen eingeliefert. Herr Berger ist 76 Jahre alt und hat zu viel Wasser in den Beinen.""").toDS.toDF("text"))

val result = nlpPipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("de.med_ner.deid_generic").predict("""Michael Berger wird am Morgen des 12 Dezember 2018 ins St. Elisabeth-Krankenhaus
in Bad Kissingen eingeliefert. Herr Berger ist 76 Jahre alt und hat zu viel Wasser in den Beinen.""")
```

</div>


## Results


```bash
+-------------------------+----------------------+
|chunk                    |ner_deid_generic_chunk|
+-------------------------+----------------------+
|Michael Berger           |NAME                  |
|12 Dezember 2018         |DATE                  |
|St. Elisabeth-Krankenhaus|LOCATION              |
|Bad Kissingen            |LOCATION              |
|Berger                   |NAME                  |
|76                       |AGE                   |
+-------------------------+----------------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_deid_generic|
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
label      tp     fp      fn   total  precision  recall      f1
CONTACT    68.0   25.0    12.0    80.0     0.7312    0.85  0.7861
NAME  3965.0  294.0   274.0  4239.0      0.931  0.9354  0.9332
DATE  4049.0    2.0     0.0  4049.0     0.9995     1.0  0.9998
ID   185.0   11.0    32.0   217.0     0.9439  0.8525  0.8959
LOCATION  5065.0  414.0  1021.0  6086.0     0.9244  0.8322  0.8759
PROFESSION   145.0    8.0   117.0   262.0     0.9477  0.5534  0.6988
AGE   458.0   13.0    18.0   476.0     0.9724  0.9622  0.9673
```
