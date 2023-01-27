---
layout: model
title: Detect PHI for Deidentification purposes (Italian, reduced entities)
author: John Snow Labs
name: ner_deid_generic
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


Deidentification NER (Italian) is a Named Entity Recognition model that annotates text to find protected health information that may need to be de-identified. It detects 8 entities. This NER model is trained with a custom dataset internally annotated, a COVID-19 Italian de-identification research dataset making up 15% of the total data [(Catelli et al.)](https://ieeexplore.ieee.org/document/9335570) and several data augmentation mechanisms.


## Predicted Entities


`CONTACT`, `NAME`, `DATE`, `ID`, `LOCATION`, `PROFESSION`, `AGE`, `SEX`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_generic_it_3.4.2_2.4_1648224320582.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deid_generic_it_3.4.2_2.4_1648224320582.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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


clinical_ner = MedicalNerModel.pretrained("ner_deid_generic", "it", "clinical/models")\
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
.setInputCols("document")
.setOutputCol("sentence")


val tokenizer = new Tokenizer()
.setInputCols("sentence")
.setOutputCol("token")


val embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d", "it")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")


val clinical_ner = MedicalNerModel.pretrained("ner_deid_generic", "it", "clinical/models")
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
nlu.load("it.med_ner.deid_generic").predict("""Ho visto Gastone Montanariello (49 anni) riferito all' Ospedale San Camillo per diabete mal controllato con sintomi risalenti a marzo 2015.""")
```

</div>


## Results


```bash
+-------------+----------+
|        token| ner_label|
+-------------+----------+
|           Ho|         O|
|        visto|         O|
|      Gastone|    B-NAME|
|Montanariello|    I-NAME|
|            (|         O|
|           49|     B-AGE|
|         anni|         O|
|            )|         O|
|     riferito|         O|
|          all|         O|
|            '|         O|
|     Ospedale|B-LOCATION|
|          San|I-LOCATION|
|      Camillo|I-LOCATION|
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
|Model Name:|ner_deid_generic|
|Compatibility:|Healthcare NLP 3.4.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|it|
|Size:|15.0 MB|


## References


- Internally annotated corpus
- [COVID-19 Italian de-identification dataset making up 15% of total data - R. Catelli, F. Gargiulo, V. Casola, G. De Pietro, H. Fujita and M. Esposito, "A Novel COVID-19 Data Set and an Effective Deep Learning Approach for the De-Identification of Italian Medical Records," in IEEE Access, vol. 9, pp. 19097-19110, 2021, doi: 10.1109/ACCESS.2021.3054479.](https://ieeexplore.ieee.org/document/9335570)


## Benchmarking


```bash
label      tp    fp     fn   total  precision  recall      f1
CONTACT   244.0   1.0    0.0   244.0     0.9959     1.0   0.998
NAME  1082.0  69.0   59.0  1141.0     0.9401  0.9483  0.9442
DATE  1173.0  26.0   17.0  1190.0     0.9783  0.9857   0.982
ID   138.0   2.0   21.0   159.0     0.9857  0.8679  0.9231
SEX   742.0  21.0   32.0   774.0     0.9725  0.9587  0.9655
LOCATION  1039.0  64.0  108.0  1147.0      0.942  0.9058  0.9236
PROFESSION   300.0  15.0   69.0   369.0     0.9524   0.813  0.8772
AGE   746.0   5.0   35.0   781.0     0.9933  0.9552  0.9739
macro       -     -      -       -          -       -  0.9484
micro       -     -      -       -          -       -  0.9521
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMzI5NTgzNzAyLC02MzUyMjYwNDddfQ==
-->