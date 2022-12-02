---
layout: model
title: Detect PHI for Deidentification (Generic - Augmented)
author: John Snow Labs
name: ner_deid_generic_augmented
date: 2021-06-01
tags: [clinical, deid, ner, en, licensed]
task: [Named Entity Recognition, De-identification]
language: en
edition: Healthcare NLP 3.1.0
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN. Deidentification NER (Generic  - Augmented) is a Named Entity Recognition model that annotates text to find protected health information that may need to be de-identified. It detects 7 entities. This ner model is trained with a combination of i2b2 train set and augmented version of i2b2 train set.


We sticked to official annotation guideline (AG) for 2014 i2b2 Deid challenge while annotating new datasets for this model. All the details regarding the nuances and explanations for AG can be found here [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4978170/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4978170/)


## Predicted Entities


`DATE`, `NAME`, `LOCATION`, `PROFESSION`, `CONTACT`, `AGE`, `ID`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DEMOGRAPHICS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_generic_augmented_en_3.0.3_2.4_1622538123966.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

deid_ner = MedicalNerModel.pretrained("ner_deid_generic_augmented", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk_generic")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, deid_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame(pd.DataFrame({"text": ["""A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25 -year-old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227."""]})))
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence_detector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val deid_ner = MedicalNerModel.pretrained("ner_deid_generic_augmented", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk_generic")

val nlpPipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, deid_ner, ner_converter))

val data = Seq("""A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25 -year-old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227.""").toDS.toDF("text")

val result = nlpPipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.deid.generic_augmented").predict("""A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25 -year-old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227.""")
```

</div>


## Results


```bash
+-----------------------------+---------+
|chunk                        |ner_label|
+-----------------------------+---------+
|2093-01-13                   |DATE     |
|David Hale                   |NAME     |
|Hendrickson, Ora             |NAME     |
|7194334                      |ID       |
|01/13/93                     |DATE     |
|Oliveira                     |NAME     |
|25                           |AGE      |
|1-11-2000                    |DATE     |
|Cocke County Baptist Hospital|LOCATION |
|0295 Keats Street            |LOCATION |
|(302) 786-5227               |CONTACT  |
+-----------------------------+---------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_deid_generic_augmented|
|Compatibility:|Healthcare NLP 3.0.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|


## Data Source


A custom data set which is created from the i2b2-PHI train and the augmented version of the i2b2-PHI train set is used.


## Benchmarking


```bash
entity          tp      fp     fn     total   precision  recall      f1
CONTACT        341.0   15.0   14.0    355.0     0.9579   0.9606    0.9592
NAME           5065.0  165.0  205.0   5270.0    0.9685   0.9611    0.9648
DATE           5532.0  53.0   110.0   5642.0    0.9905   0.9805    0.9855
ID             614.0   23.0   71.0    685.0     0.9639   0.8964    0.9289
LOCATION       2686.0  164.0  284.0   2970.0    0.9425   0.9044    0.923
PROFESSION     228.0   28.0   145.0   373.0     0.8906   0.6113    0.725
AGE            713.0   34.0   49.0    762.0     0.9545   0.9357    0.945
macro-avg        -      -      -       -         -         -       0.91876
micro-avg        -      -      -       -         -         -       0.95616
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE1MzMzNTUwNzNdfQ==
-->