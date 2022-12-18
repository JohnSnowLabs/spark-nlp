---
layout: model
title: Mapping Drugs to Their Categories as well as Other Brand and Names
author: John Snow Labs
name: drug_category_mapper
date: 2022-12-18
tags: [category, chunk_mapper, drug, licensed, clinical, en]
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

This pretrained model maps drugs to their categories and other brands and names. It has two categories called main category and subcategory.

## Predicted Entities

`main_category`, `sub_category`, `other_name`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/drug_category_mapper_en_4.2.2_3.0_1671374094037.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    
ner = MedicalNerModel.pretrained("ner_posology", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")
    
converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")
    
chunkerMapper = ChunkMapperModel.pretrained("drug_category_mapper", "en", "clinical/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("mappings")\
    .setRels(["main_category", "sub_category", "other_name"])
    
pipeline = Pipeline().setStages([
    document_assembler,
    sentence_detector,
    tokenizer, 
    word_embeddings,
    ner, 
    converter, 
    chunkerMapper])

text= "She is given OxyContin, folic acid, levothyroxine, Norvasc, aspirin, Neurontin"

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
    
val ner = MedicalNerModel.pretrained("ner_posology", "en", "clinical/models") 
    .setInputCols(Array("sentence", "token", "embeddings")) 
    .setOutputCol("ner")
    
val converter = new NerConverter() 
    .setInputCols(Array("sentence", "token", "ner")) 
    .setOutputCol("ner_chunk")
    
val chunkerMapper = ChunkMapperModel.pretrained("drug_category_mapper", "en", "clinical/models")
    .setInputCols("ner_chunk")
    .setOutputCol("mappings")
    .setRels(Array(["main_category", "sub_category", "other_name"]))
    
val pipeline = new Pipeline().setStages(Array(
    document_assembler,
    sentence_detector,
    tokenizer, 
    word_embeddings,
    ner, 
    converter, 
    chunkerMapper))

val text= "She is given OxyContin, folic acid, levothyroxine, Norvasc, aspirin, Neurontin"

val data = Seq(text).toDS.toDF("text")

val result= pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-------------+---------------------+-----------------------------------+-----------+
|    ner_chunk|        main_category|                       sub_category|other_names|
+-------------+---------------------+-----------------------------------+-----------+
|    OxyContin|      Pain Management|                  Opioid Analgesics|     Oxaydo|
|   folic acid|         Nutritionals|            Vitamins, Water-Soluble|    Folvite|
|levothyroxine|Metabolic & Endocrine|                   Thyroid Products|     Levo T|
|      Norvasc|       Cardiovascular|                 Antianginal Agents|   Katerzia|
|      aspirin|       Cardiovascular|Antiplatelet Agents, Cardiovascular|        ASA|
|    Neurontin|          Neurologics|                       GABA Analogs|    Gralise|
+-------------+---------------------+-----------------------------------+-----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|drug_category_mapper|
|Compatibility:|Healthcare NLP 4.2.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|526.0 KB|
