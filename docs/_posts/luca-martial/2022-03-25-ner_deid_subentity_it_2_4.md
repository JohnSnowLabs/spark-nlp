---
layout: model
title: Detect PHI for Deidentification purposes (Italian)
author: John Snow Labs
name: ner_deid_subentity
date: 2022-03-25
tags: [deid, it, licensed]
task: Named Entity Recognition
language: it
edition: Healthcare NLP 3.4.2
spark_version: 2.4
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Named Entity Recognition annotators allow for a generic model to be trained by using a Deep Learning architecture (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM, CNN.


Deidentification NER (Italian) is a Named Entity Recognition model that annotates text to find protected health information that may need to be de-identified. It detects 19 entities. This NER model is trained with a custom dataset internally annotated, a COVID-19 Italian de-identification research dataset making up 15% of the total data [(Catelli et al.)](https://ieeexplore.ieee.org/document/9335570) and several data augmentation mechanisms.


## Predicted Entities


`DATE`, `AGE`, `SEX`, `PROFESSION`, `ORGANIZATION`, `PHONE`, `EMAIL`, `ZIP`, `STREET`, `CITY`, `COUNTRY`, `PATIENT`, `DOCTOR`, `HOSPITAL`, `MEDICALRECORD`, `SSN`, `IDNUM`, `USERNAME`, `URL`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_it_3.4.2_2.4_1648218077881.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_it_3.4.2_2.4_1648218077881.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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


embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d", "it")\
.setInputCols(["sentence", "token"])\
.setOutputCol("word_embeddings")


clinical_ner = MedicalNerModel.pretrained("ner_deid_subentity", "it", "clinical/models")\
.setInputCols(["sentence","token", "word_embeddings"])\
.setOutputCol("ner")


nlpPipeline = Pipeline(stages=[
documentAssembler,
sentenceDetector,
tokenizer,
embeddings,
clinical_ner])


text = ["Ho visto Gastone Montanariello (49 anni) riferito all' Ospedale San Camillo per diabete mal controllato con sintomi risalenti a marzo 2015."]


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


val embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d", "it")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")


val clinical_ner = MedicalNerModel.pretrained("ner_deid_subentity", "it", "clinical/models")
.setInputCols(Array("sentence","token","embeddings"))
.setOutputCol("ner")


val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, embeddings, clinical_ner))


val text = "Ho visto Gastone Montanariello (49 anni) riferito all' Ospedale San Camillo per diabete mal controllato con sintomi risalenti a marzo 2015."


val data = Seq(text).toDF("text")


val results = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("it.med_ner.deid_subentity").predict("""Ho visto Gastone Montanariello (49 anni) riferito all' Ospedale San Camillo per diabete mal controllato con sintomi risalenti a marzo 2015.""")
```

</div>


## Results


```bash
+-------------+----------+
|        token| ner_label|
+-------------+----------+
|           Ho|         O|
|        visto|         O|
|      Gastone| B-PATIENT|
|Montanariello| I-PATIENT|
|            (|         O|
|           49|     B-AGE|
|         anni|         O|
|            )|         O|
|     riferito|         O|
|          all|         O|
|            '|         O|
|     Ospedale|B-HOSPITAL|
|          San|I-HOSPITAL|
|      Camillo|I-HOSPITAL|
|          per|         O|
|      diabete|         O|
|          mal|         O|
|  controllato|         O|
|          con|         O|
|      sintomi|         O|
|    risalenti|         O|
|            a|         O|
|        marzo|    B-DATE|
|         2015|    I-DATE|
|            .|         O|
+-------------+----------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_deid_subentity|
|Compatibility:|Healthcare NLP 3.4.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|it|
|Size:|15.0 MB|


## References


- Internally annotated corpus
- [COVID-19 Italian de-identification dataset making up 15% of total data: R. Catelli, F. Gargiulo, V. Casola, G. De Pietro, H. Fujita and M. Esposito, "A Novel COVID-19 Data Set and an Effective Deep Learning Approach for the De-Identification of Italian Medical Records," in IEEE Access, vol. 9, pp. 19097-19110, 2021, doi: 10.1109/ACCESS.2021.3054479.](https://ieeexplore.ieee.org/document/9335570)


## Benchmarking


```bash
label      tp    fp    fn   total  precision  recall      f1
PATIENT   263.0  29.0  25.0   288.0     0.9007  0.9132  0.9069
HOSPITAL   365.0  36.0  48.0   413.0     0.9102  0.8838  0.8968
DATE  1164.0  13.0  26.0  1190.0      0.989  0.9782  0.9835
ORGANIZATION    72.0  25.0  26.0    98.0     0.7423  0.7347  0.7385
URL    41.0   0.0   0.0    41.0        1.0     1.0     1.0
CITY   421.0   9.0  19.0   440.0     0.9791  0.9568  0.9678
STREET   198.0   4.0   6.0   204.0     0.9802  0.9706  0.9754
USERNAME    20.0   2.0   2.0    22.0     0.9091  0.9091  0.9091
SEX   753.0  26.0  21.0   774.0     0.9666  0.9729  0.9697
IDNUM   113.0   3.0   7.0   120.0     0.9741  0.9417  0.9576
EMAIL   148.0   0.0   0.0   148.0        1.0     1.0     1.0
ZIP   148.0   3.0   1.0   149.0     0.9801  0.9933  0.9867
MEDICALRECORD    19.0   3.0   6.0    25.0     0.8636    0.76  0.8085
SSN    13.0   1.0   1.0    14.0     0.9286  0.9286  0.9286
PROFESSION   316.0  28.0  53.0   369.0     0.9186  0.8564  0.8864
PHONE    53.0   0.0   2.0    55.0        1.0  0.9636  0.9815
COUNTRY   182.0  14.0  15.0   197.0     0.9286  0.9239  0.9262
DOCTOR   769.0  77.0  62.0   831.0      0.909  0.9254  0.9171
AGE   763.0   8.0  18.0   781.0     0.9896   0.977  0.9832
macro       -     -     -       -          -       -  0.9328
micro       -     -     -       -          -       -  0.9494
```