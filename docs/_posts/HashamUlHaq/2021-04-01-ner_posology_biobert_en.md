---
layout: model
title: Detect posology entities (biobert)
author: John Snow Labs
name: ner_posology_biobert
date: 2021-04-01
tags: [ner, clinical, licensed, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.0.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Detect Drug, Dosage and administration instructions in text using pretraiend NER model.


## Predicted Entities


`FREQUENCY`, `DRUG`, `STRENGTH`, `FORM`, `DURATION`, `DOSAGE`, `ROUTE`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_POSOLOGY/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_posology_biobert_en_3.0.0_3.0_1617260806766.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
         
sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings_clinical = BertEmbeddings.pretrained("biobert_pubmed_base_cased")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_posology_biobert", "en", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = NerConverter()\
 	  .setInputCols(["sentence", "token", "ner"])\
 	  .setOutputCol("ner_chunk")
    
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical, clinical_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["EXAMPLE_TEXT"]]).toDF("text"))
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

val embeddings_clinical = BertEmbeddings.pretrained("biobert_pubmed_base_cased")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_posology_biobert", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
 	.setInputCols(Array("sentence", "token", "ner"))
 	.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.posology.biobert").predict("""Put your text here.""")
```

</div>


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_posology_biobert|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|


## Benchmarking


```bash
label  precision    recall  f1-score   support
B-DOSAGE       0.78      0.67      0.72       559
B-DRUG       0.93      0.94      0.94      3865
B-DURATION       0.79      0.81      0.80       331
B-FORM       0.90      0.87      0.88      1472
B-FREQUENCY       0.92      0.94      0.93      1577
B-ROUTE       0.94      0.85      0.89       772
B-STRENGTH       0.88      0.92      0.90      2519
I-DOSAGE       0.62      0.57      0.60       357
I-DRUG       0.81      0.89      0.85      1539
I-DURATION       0.80      0.89      0.84       796
I-FORM       0.58      0.54      0.56       142
I-FREQUENCY       0.86      0.93      0.89      2424
I-ROUTE       1.00      0.47      0.64        32
I-STRENGTH       0.85      0.91      0.88      2972
O       0.98      0.98      0.98    101134
accuracy       -         -         0.97    120491
macro-avg       0.84      0.81      0.82    120491
weighted-avg       0.97      0.97      0.97    120491
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTI4NTU1NDEwN119
-->