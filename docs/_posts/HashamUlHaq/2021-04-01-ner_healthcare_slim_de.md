---
layout: model
title: Detect clinical entities (ner_healthcare_slim)
author: John Snow Labs
name: ner_healthcare_slim
date: 2021-04-01
tags: [ner, clinical, licensed, de]
task: Named Entity Recognition
language: de
edition: Spark NLP for Healthcare 3.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Detect clinical entities in German text using pretrained NER model

## Predicted Entities

`TREATMENT`, `PERSON`, `BODY_PART`, `TIME_INFORMATION`, `MEDICAL_CONDITION`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_HEALTHCARE_DE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_healthcare_slim_de_3.0.0_3.0_1617260856273.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

...
embeddings_clinical = WordEmbeddingsModel.pretrained("w2v_cc_300d", "de", "clinical/models")  .setInputCols(["sentence", "token"])  .setOutputCol("embeddings")
clinical_ner = MedicalNerModel.pretrained("ner_healthcare_slim", "de", "clinical/models")   .setInputCols(["sentence", "token", "embeddings"])   .setOutputCol("ner")
...
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical, clinical_ner, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
results = model.transform(spark.createDataFrame([["EXAMPLE_TEXT"]]).toDF("text"))
```
```scala

...
val embeddings_clinical = WordEmbeddingsModel.pretrained("w2v_cc_300d", "de", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")
val ner = MedicalNerModel.pretrained("ner_healthcare_slim", "de", "clinical/models")
  .setInputCols(Array("sentence", "token", "embeddings"))
  .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))
val result = pipeline.fit(Seq.empty[String]).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("de.med_ner").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_healthcare_slim|
|Compatibility:|Spark NLP for Healthcare 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|de|