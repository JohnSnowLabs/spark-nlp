---
layout: model
title: Detect Genes and Human Phenotypes
author: John Snow Labs
name: ner_human_phenotype_gene_clinical
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

This model detects mentions of genes and human phenotypes (hp) in medical text.

## Predicted Entities

: `GENE`, `HP`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_HUMAN_PHENOTYPE_GENE_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_human_phenotype_gene_clinical_en_3.0.0_3.0_1617209698053.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_human_phenotype_gene_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter()\
 	.setInputCols(["sentence", "token", "ner"])\
 	.setOutputCol("ner_chunk")
    
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])

model = nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text"))

results = model.transform(spark.createDataFrame([["Here we presented a case (BS type) of a 17 years old female presented with polyhydramnios, polyuria, nephrocalcinosis and hypokalemia, which was alleviated after treatment with celecoxib and vitamin D(3)."]]).toDF("text"))
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

val ner = MedicalNerModel.pretrained("ner_human_phenotype_gene_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
 	.setInputCols(Array("sentence", "token", "ner"))
 	.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))

val data = Seq("""Here we presented a case (BS type) of a 17 years old female presented with polyhydramnios, polyuria, nephrocalcinosis and hypokalemia, which was alleviated after treatment with celecoxib and vitamin D(3).""").toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.human_phenotype.gene_clinical").predict("""Here we presented a case (BS type) of a 17 years old female presented with polyhydramnios, polyuria, nephrocalcinosis and hypokalemia, which was alleviated after treatment with celecoxib and vitamin D(3).""")
```

</div>

## Results

```bash
+----+------------------+---------+-------+----------+
|    | chunk            |   begin |   end | entity   |
+====+==================+=========+=======+==========+
|  0 | BS type          |      29 |    32 | GENE     |
+----+------------------+---------+-------+----------+
|  1 | polyhydramnios   |      75 |    88 | HP       |
+----+------------------+---------+-------+----------+
|  2 | polyuria         |      91 |    98 | HP       |
+----+------------------+---------+-------+----------+
|  3 | nephrocalcinosis |     101 |   116 | HP       |
+----+------------------+---------+-------+----------+
|  4 | hypokalemia      |     122 |   132 | HP       |
+----+------------------+---------+-------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_human_phenotype_gene_clinical|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Benchmarking

```bash
|    | label         |    tp |   fp |   fn |     prec |      rec |       f1 |
|---:|--------------:|------:|-----:|-----:|---------:|---------:|---------:|
|  0 | I-HP          |   303 |   56 |   64 | 0.844011 | 0.825613 | 0.834711 |
|  1 | B-GENE        |  1176 |  158 |  252 | 0.881559 | 0.823529 | 0.851557 |
|  2 | B-HP          |  1078 |  133 |   96 | 0.890173 | 0.918228 | 0.903983 |
|  3 | Macro-average | 2557  | 347  |  412 | 0.871915 | 0.85579  | 0.863777 |
|  4 | Micro-average | 2557  | 347  |  412 | 0.88051  | 0.861233 | 0.870765 |
```