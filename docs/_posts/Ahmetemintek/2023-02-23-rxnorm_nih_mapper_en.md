---
layout: model
title: Mapping Entities with Corresponding RxNorm Codes According to According to National Institute of Health (NIH) Database
author: John Snow Labs
name: rxnorm_nih_mapper
date: 2023-02-23
tags: [rxnorm, nih, chunk_mapping, clinical, en, licensed]
task: Chunk Mapping
language: en
edition: Healthcare NLP 4.3.0
spark_version: [3.0, 3.2]
supported: true
annotator: ChunkMapperModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained model maps entities with their corresponding RxNorm codes according to the National Institute of Health (NIH) database. It returns Rxnorm codes with their NIH Rxnorm Term Types within a parenthesis.

## Predicted Entities

`rxnorm_code`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/rxnorm_nih_mapper_en_4.3.0_3.2_1677156206111.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/rxnorm_nih_mapper_en_4.3.0_3.2_1677156206111.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
.setInputCol('text')\
.setOutputCol('document')

sentence_detector = SentenceDetector()\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer()\
.setInputCols("sentence")\
.setOutputCol("token")

word_embeddings = WordEmbeddingsModel\
.pretrained("embeddings_clinical", "en", "clinical/models")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")

posology_ner_model = MedicalNerModel\
.pretrained("ner_posology_greedy", "en", "clinical/models")\
.setInputCols(["sentence", "token", "embeddings"])\
.setOutputCol("posology_ner")

posology_ner_converter = NerConverterInternal()\
.setInputCols("sentence", "token", "posology_ner")\
.setOutputCol("ner_chunk")

chunkerMapper = ChunkMapperModel\
.pretrained("rxnorm_nih_mapper", "en", "clinical/models")\
.setInputCols(["ner_chunk"])\
.setOutputCol("mappings")\
.setRels(["rxnorm_code"])

mapper_pipeline = Pipeline().setStages([
document_assembler,
sentence_detector,
tokenizer, 
word_embeddings,
posology_ner_model, 
posology_ner_converter, 
chunkerMapper])


test_data = spark.createDataFrame([["The patient was given Adapin 10 MG Oral Capsule, acetohexamide and Parlodel"]]).toDF("text")

mapper_model = mapper_pipeline.fit(test_data)

result= mapper_model.transform(test_data)
```
```scala
val document_assembler = new DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

val sentence_detector = new SentenceDetector()\
.setInputCols(Array("document"))\
.setOutputCol("sentence")

val tokenizer = new Tokenizer()\
.setInputCols("sentence")\
.setOutputCol("token")

val word_embeddings = WordEmbeddingsModel
.pretrained("embeddings_clinical", "en", "clinical/models")\
.setInputCols(Array("sentence", "token"))\
.setOutputCol("embeddings")

val posology_ner_model = MedicalNerModel
.pretrained("ner_posology_greedy", "en", "clinical/models")\
.setInputCols(Array("sentence", "token", "embeddings"))\
.setOutputCol("posology_ner")

val posology_ner_converter = new NerConverterInternal()\
.setInputCols("sentence", "token", "posology_ner")\
.setOutputCol("ner_chunk")

val chunkerMapper = ChunkMapperModel
.pretrained("rxnorm_nih_mapper", "en", "clinical/models")\
.setInputCols(Array("ner_chunk"))\
.setOutputCol("mappings")\
.setRels(Array("rxnorm_code")) 

val mapper_pipeline = new Pipeline().setStages(Array(
document_assembler,
sentence_detector,
tokenizer, 
word_embeddings,
posology_ner_model, 
posology_ner_converter, 
chunkerMapper))


val data = Seq("The patient was given Adapin 10 MG Oral Capsule, acetohexamide and Parlodel").toDS.toDF("text")

val result = pipeline.fit(data).transform(data) 
```
</div>

## Results

```bash
+-------------------------+-------------+-----------+
|ner_chunk                |mappings     |relation   |
+-------------------------+-------------+-----------+
|Adapin 10 MG Oral Capsule|1911002 (SY) |rxnorm_code|
|acetohexamide            |12250421 (IN)|rxnorm_code|
|Parlodel                 |829 (BN)     |rxnorm_code|
+-------------------------+-------------+-----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|rxnorm_nih_mapper|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|10.3 MB|

## References

Trained on February 2023 with NIH data: 
 https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html