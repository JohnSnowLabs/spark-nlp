---
layout: model
title: Mapping Entities with Corresponding RxNorm Codes
author: John Snow Labs
name: rxnorm_mapper
date: 2022-06-07
tags: [en, rxnorm, licensed, chunk_mapper]
task: Chunk Mapping
language: en
edition: Healthcare NLP 3.5.0
spark_version: 3.0
supported: true
annotator: ChunkMapperModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained model maps entities with their corresponding RxNorm codes

## Predicted Entities

`rxnorm_code`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/rxnorm_mapper_en_3.5.0_3.0_1654614618628.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/rxnorm_mapper_en_3.5.0_3.0_1654614618628.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")

posology_ner_model = MedicalNerModel.pretrained("ner_posology_greedy", "en", "clinical/models")\
.setInputCols(["sentence", "token", "embeddings"])\
.setOutputCol("posology_ner")

posology_ner_converter = NerConverterInternal()\
.setInputCols("sentence", "token", "posology_ner")\
.setOutputCol("ner_chunk")

chunkerMapper = ChunkMapperModel.pretrained("rxnorm_mapper", "en", "clinical/models")\
.setInputCols(["ner_chunk"])\
.setOutputCol("mappings")\
.setRel("rxnorm_code") 

mapper_pipeline = Pipeline().setStages([
document_assembler,
sentence_detector,
tokenizer, 
word_embeddings,
posology_ner_model, 
posology_ner_converter, 
chunkerMapper])


data = spark.createDataFrame([["The patient was given Zyrtec 10 MG, Adapin 10 MG Oral Capsule, Septi-Soothe 0.5 Topical Spray"]]).toDF("text")
result = mapper_pipeline.fit(data).transform(data) 

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

val posology_ner_model = MedicalNerModel.pretrained("ner_posology_greedy", "en", "clinical/models")
.setInputCols(Array("sentence", "token", "embeddings"))
.setOutputCol("posology_ner")

val posology_ner_converter = new NerConverterInternal()
.setInputCols("sentence", "token", "posology_ner")
.setOutputCol("ner_chunk")

val chunkerMapper = ChunkMapperModel.pretrained("rxnorm_mapper", "en", "clinical/models")
.setInputCols(Array("ner_chunk"))
.setOutputCol("mappings")
.setRel("rxnorm_code") 

val mapper_pipeline = new Pipeline().setStages(Array(
document_assembler,
sentence_detector,
tokenizer, 
word_embeddings,
posology_ner_model, 
posology_ner_converter, 
chunkerMapper))

val senetence= "The patient was given Zyrtec 10 MG, Adapin 10 MG Oral Capsule, Septi-Soothe 0.5 Topical Spray"
val data = Seq(senetence).toDF("text")

val result = mapper_pipeline.fit(data).transform(data) 
```


{:.nlu-block}
```python
import nlu
nlu.load("en.map_entity.rxnorm_resolver").predict("""The patient was given Zyrtec 10 MG, Adapin 10 MG Oral Capsule, Septi-Soothe 0.5 Topical Spray""")
```

</div>

## Results

```bash
+------------------------------+---------------+
|chunk                         |rxnorm_mappings|
+------------------------------+---------------+
|Zyrtec 10 MG                  |1011483        |
|Adapin 10 MG Oral Capsule     |1000050        |
|Septi-Soothe 0.5 Topical Spray|1000046        |
+------------------------------+---------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|rxnorm_mapper|
|Compatibility:|Healthcare NLP 3.5.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[posology_ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|2.3 MB|