---
layout: model
title: Mapping Abbreviations and Acronyms of Medical Regulatory Activities with Their Definitions and Categories
author: John Snow Labs
name: abbreviation_category_mapper
date: 2022-11-16
tags: [abbreviation, definition, category, licensed, en, clinical, chunk_mapper]
task: Chunk Mapping
language: en
edition: Healthcare NLP 4.2.1
spark_version: 3.0
supported: true
annotator: ChunkMapperModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained model maps abbreviations and acronyms of medical regulatory activities with their definitions and categories. Predicted categories: `general`, `problem`, `test`, `treatment`, `medical_condition`, `clinical_dept`, `drug`, `nursing`, `internal_organ_or_component`, `hospital_unit`, `drug_frequency`, `employment`, `procedure`

## Predicted Entities

`definition`, `category`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/abbreviation_category_mapper_en_4.2.1_3.0_1668594867892.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/abbreviation_category_mapper_en_4.2.1_3.0_1668594867892.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

abbr_ner = MedicalNerModel.pretrained("ner_abbreviation_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("abbr_ner")

abbr_converter = NerConverter() \
    .setInputCols(["sentence", "token", "abbr_ner"]) \
    .setOutputCol("abbr_ner_chunk")\

chunkerMapper = ChunkMapperModel.pretrained("abbreviation_category_mapper", "en", "clinical/models")\
    .setInputCols(["abbr_ner_chunk"])\
    .setOutputCol("mappings")\
    .setRels(["definition", "category"])\

pipeline = Pipeline().setStages([
    document_assembler,
    sentence_detector,
    tokenizer, 
    word_embeddings,
    abbr_ner, 
    abbr_converter, 
    chunkerMapper])


text = ["""Gravid with estimated fetal weight of 6-6/12 pounds.
           LABORATORY DATA: Laboratory tests include a CBC which is normal. 
           VDRL: Nonreactive
           HIV: Negative. One-Hour Glucose: 117. Group B strep has not been done as yet."""]

data = spark.createDataFrame([text]).toDF("text")

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

val abbr_ner = MedicalNerModel.pretrained("ner_abbreviation_clinical", "en", "clinical/models") 
    .setInputCols(Array("sentence", "token", "embeddings")) 
    .setOutputCol("abbr_ner")

val abbr_converter = new NerConverter() 
    .setInputCols(Array("sentence", "token", "abbr_ner")) 
    .setOutputCol("abbr_ner_chunk")

val chunkerMapper = ChunkMapperModel.pretrained("abbreviation_category_mapper", "en", "clinical/models")
    .setInputCols("abbr_ner_chunk")
    .setOutputCol("mappings")
    .setRels(Array("definition", "category"))


val pipeline = new Pipeline().setStages(Array(
    document_assembler,
    sentence_detector,
    tokenizer, 
    word_embeddings,
    abbr_ner, 
    abbr_converter, 
    chunkerMapper))


val sample_text = """Gravid with estimated fetal weight of 6-6/12 pounds.
           LABORATORY DATA: Laboratory tests include a CBC which is normal. 
           VDRL: Nonreactive
           HIV: Negative. One-Hour Glucose: 117. Group B strep has not been done as yet.""" 


val data = Seq(sample_text).toDS.toDF("text")

val result= pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
|    | chunk   | category          | definition                             |
|---:|:--------|:------------------|:---------------------------------------|
|  0 | CBC     | general           | complete blood count                   |
|  1 | VDRL    | clinical_dept     | Venereal Disease Research Laboratories |
|  2 | HIV     | medical_condition | Human immunodeficiency virus           |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|abbreviation_category_mapper|
|Compatibility:|Healthcare NLP 4.2.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[abbr_ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|128.2 KB|