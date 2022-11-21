---
layout: model
title: Detect Anatomical Structures (Single Entity - biobert)
author: John Snow Labs
name: ner_anatomy_coarse_biobert
date: 2021-03-31
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

An NER model to extract all types of anatomical references in text using "biobert_pubmed_base_cased" embeddings. It is a single entity model and generalizes all anatomical references to a single entity.

## Predicted Entities

`Anatomy`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_ANATOMY/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_anatomy_coarse_biobert_en_3.0.0_3.0_1617209714335.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased", "en") \
    .setInputCols("sentence", "token") \
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_anatomy_coarse_biobert", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter()\
 	.setInputCols(["sentence", "token", "ner"])\
 	.setOutputCol("ner_chunk")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings, clinical_ner, ner_converter])

model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["content in the lung tissue"]], ["text"]))
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

val embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased", "en")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_anatomy_coarse_biobert", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
 	.setInputCols(Array("sentence", "token", "ner"))
 	.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings, ner, ner_converter))

val data = Seq("""content in the lung tissue""").toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.anatomy.coarse_biobert").predict("""content in the lung tissue""")
```

</div>

## Results

```bash
|    | ner_chunk         | entity    |
|---:|:------------------|:----------|
|  0 | lung tissue       | Anatomy   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_anatomy_coarse_biobert|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

Trained on a custom dataset using 'biobert_pubmed_base_cased'.

## Benchmarking

```bash
|    | label         |    tp |    fp |    fn |     prec |      rec |       f1 |
|---:|--------------:|------:|------:|------:|---------:|---------:|---------:|
|  0 | B-Anatomy     |  2499 |   155 |   162 | 0.941598 | 0.939121 | 0.940357 |
|  1 | I-Anatomy     |  1695 |   116 |   158 | 0.935947 | 0.914733 | 0.925218 |
|  2 | Macro-average | 4194  |  271  |   320 | 0.938772 | 0.926927 | 0.932812 |
|  3 | Micro-average | 4194  |  271  |   320 | 0.939306 | 0.929109 | 0.93418  |
```