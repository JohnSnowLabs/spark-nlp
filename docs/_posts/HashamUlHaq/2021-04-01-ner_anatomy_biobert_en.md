---
layout: model
title: Detect Anatomical References (biobert)
author: John Snow Labs
name: ner_anatomy_biobert
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

Detect anatomical sites and references in medical text using pretrained NER model.

## Predicted Entities

`tissue_structure`, `Organism_substance`, `Developing_anatomical_structure`, `Cell`, `Cellular_component`, `Immaterial_anatomical_entity`, `Organ`, `Pathological_formation`, `Organism_subdivision`, `Anatomical_system`, `Tissue`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_ANATOMY/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_anatomy_biobert_en_3.0.0_3.0_1617260624773.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_anatomy_biobert_en_3.0.0_3.0_1617260624773.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

clinical_ner = MedicalNerModel.pretrained("ner_anatomy_biobert", "en", "clinical/models")\
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

val ner = MedicalNerModel.pretrained("ner_anatomy_biobert", "en", "clinical/models")
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
nlu.load("en.med_ner.anatomy.biobert").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_anatomy_biobert|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Benchmarking
```bash
+-------------------------------+-----+----+----+-----+---------+------+------+
|                         entity|   tp|  fp|  fn|total|precision|recall|    f1|
+-------------------------------+-----+----+----+-----+---------+------+------+
|                          Organ| 53.0|17.0|12.0| 65.0|   0.7571|0.8154|0.7852|
|         Pathological_formation| 83.0|23.0|14.0| 97.0|    0.783|0.8557|0.8177|
|             Organism_substance| 42.0| 1.0|14.0| 56.0|   0.9767|  0.75|0.8485|
|               tissue_structure|131.0|28.0|49.0|180.0|   0.8239|0.7278|0.7729|
|             Cellular_component| 17.0| 0.0|20.0| 37.0|      1.0|0.4595|0.6296|
|                         Tissue| 27.0| 4.0|16.0| 43.0|    0.871|0.6279|0.7297|
|              Anatomical_system| 15.0| 3.0| 8.0| 23.0|   0.8333|0.6522|0.7317|
|Developing_anatomical_structure|  2.0| 1.0| 3.0|  5.0|   0.6667|   0.4|   0.5|
|   Immaterial_anatomical_entity|  7.0| 2.0| 6.0| 13.0|   0.7778|0.5385|0.6364|
|                           Cell|180.0| 6.0|15.0|195.0|   0.9677|0.9231|0.9449|
|           Organism_subdivision| 11.0| 5.0|10.0| 21.0|   0.6875|0.5238|0.5946|
+-------------------------------+-----+----+----+-----+---------+------+------+

+------------------+
|             macro|
+------------------+
|0.7264701979913192|
+------------------+

+------------------+
|             micro|
+------------------+
|0.8108878300337679|
+------------------+
```