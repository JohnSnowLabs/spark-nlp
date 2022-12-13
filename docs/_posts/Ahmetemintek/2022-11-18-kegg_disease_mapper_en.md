---
layout: model
title: Mapping Diseases from the KEGG Database to Their Corresponding Categories, Descriptions and Clinical Vocabularies
author: John Snow Labs
name: kegg_disease_mapper
date: 2022-11-18
tags: [disease, category, description, icd10, icd11, mesh, brite, en, clinical, chunk_mapper, licensed]
task: Chunk Mapping
language: en
edition: Healthcare NLP 4.2.2
spark_version: 3.0
supported: true
annotator: ChunkMapperModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained model maps diseases with their corresponding `category`, `description`, `icd10_code`, `icd11_code`, `mesh_code`, and hierarchical `brite_code`. This model was trained with the data from the KEGG database.

## Predicted Entities

`category`, `description`, `icd10_code`, `icd11_code`, `mesh_code`, `brite_code`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/kegg_disease_mapper_en_4.2.2_3.0_1668794743905.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    .setInputCols("sentence")\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

ner = MedicalNerModel.pretrained("ner_diseases", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")\

chunkerMapper = ChunkMapperModel.pretrained("kegg_disease_mapper", "en", "clinical/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("mappings")\
    .setRels(["description", "category", "icd10_code", "icd11_code", "mesh_code", "brite_code"])\

pipeline = Pipeline().setStages([
    document_assembler,
    sentence_detector,
    tokenizer, 
    word_embeddings,
    ner, 
    converter, 
    chunkerMapper])


text= "A 55-year-old female with a history of myopia, kniest dysplasia and prostate cancer. She was on glipizide , and dapagliflozin for congenital nephrogenic diabetes insipidus."

data = spark.createDataFrame([[text]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence_detector = new SentenceDetector()
    .setInputCols(Array("document"))
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_diseases", "en", "clinical/models") 
    .setInputCols(Array("sentence", "token", "embeddings")) 
    .setOutputCol("ner")

val converter = new NerConverter() 
    .setInputCols(Array("sentence", "token", "ner")) 
    .setOutputCol("ner_chunk")

val chunkerMapper = ChunkMapperModel.pretrained("kegg_disease_mapper", "en", "clinical/models")
    .setInputCols("ner_chunk")
    .setOutputCol("mappings")
    .setRels(Array("description", "category", "icd10_code", "icd11_code", "mesh_code", "brite_code"))


val pipeline = new Pipeline().setStages(Array(
    document_assembler,
    sentence_detector,
    tokenizer, 
    word_embeddings,
    ner, 
    converter, 
    chunkerMapper))


val text= "A 55-year-old female with a history of myopia, kniest dysplasia and prostate cancer. She was on glipizide , and dapagliflozin for congenital nephrogenic diabetes insipidus."


val data = Seq(text).toDS.toDF("text")

val result= pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-----------------------------------------+--------------------------------------------------+-----------------------+----------+----------+---------+-----------------------+
|                                ner_chunk|                                       description|               category|icd10_code|icd11_code|mesh_code|             brite_code|
+-----------------------------------------+--------------------------------------------------+-----------------------+----------+----------+---------+-----------------------+
|                                   myopia|Myopia is the most common ocular disorder world...| Nervous system disease|     H52.1|    9D00.0|  D009216|            08402,08403|
|                         kniest dysplasia|Kniest dysplasia is an autosomal dominant chond...|Congenital malformation|     Q77.7|    LD24.3|  C537207|            08402,08403|
|                          prostate cancer|Prostate cancer constitutes a major health prob...|                 Cancer|       C61|      2C82|     NONE|08402,08403,08442,08441|
|congenital nephrogenic diabetes insipidus|Nephrogenic diabetes insipidus (NDI) is charact...| Urinary system disease|     N25.1|   GB90.4A|  D018500|            08402,08403|
+-----------------------------------------+--------------------------------------------------+-----------------------+----------+----------+---------+-----------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|kegg_disease_mapper|
|Compatibility:|Healthcare NLP 4.2.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|595.6 KB|