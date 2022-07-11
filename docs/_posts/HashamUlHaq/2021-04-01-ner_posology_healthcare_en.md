---
layout: model
title: Detect Posology concepts (ner_posology_healthcare)
author: John Snow Labs
name: ner_posology_healthcare
date: 2021-04-01
tags: [ner, clinical, licensed, en]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 3.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Detect Drug, Dosage and administration instructions in text using pretraiend NER model.


## Predicted Entities


`Drug`, `Duration`, `Strength`, `Form`, `Frequency`, `Dosage`, `Route`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_POSOLOGY/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_posology_healthcare_en_3.0.0_3.0_1617260847574.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_healthcare", "en", "clinical/models").setInputCols(["sentence", "token"]).setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_posology_healthcare", "en", "clinical/models").setInputCols(["sentence", "token", "embeddings"]).setOutputCol("ner")
...
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical, clinical_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["EXAMPLE_TEXT"]]).toDF("text"))
```
```scala
...
val embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_healthcare", "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_posology_healthcare", "en", "clinical/models")
  .setInputCols(Array("sentence", "token", "embeddings"))
  .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))

val result = pipeline.fit(Seq.empty[String]).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.posology.healthcare").predict("""Put your text here.""")
```

</div>


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_posology_healthcare|
|Compatibility:|Spark NLP for Healthcare 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|




## Benchmarking
```bash
   entity      tp     fp     fn   total  precision  recall      f1
 DURATION   995.0  463.0  132.0  1127.0     0.6824  0.8829  0.7698
     DRUG  4957.0  632.0  476.0  5433.0     0.8869  0.9124  0.8995
   DOSAGE   539.0  183.0  380.0   919.0     0.7465  0.5865  0.6569
    ROUTE   676.0   47.0  129.0   805.0      0.935  0.8398  0.8848
FREQUENCY  3688.0  675.0  313.0  4001.0     0.8453  0.9218  0.8819
     FORM  1328.0  261.0  294.0  1622.0     0.8357  0.8187  0.8272
 STRENGTH  5008.0  687.0  557.0  5565.0     0.8794  0.8999  0.8895
    macro     -      -      -       -         -       -     0.82994
    micro     -      -      -       -         -       -     0.86743
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTA3NjIzOTY1XX0=
-->