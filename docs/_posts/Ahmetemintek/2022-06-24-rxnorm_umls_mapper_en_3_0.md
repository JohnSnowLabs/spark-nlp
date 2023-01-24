---
layout: model
title: Mapping RxNorm Codes with Corresponding UMLS Codes
author: John Snow Labs
name: rxnorm_umls_mapper
date: 2022-06-24
tags: [rxnorm, umls, chunk_mapper, en, licensed]
task: Chunk Mapping
language: en
edition: Healthcare NLP 3.5.3
spark_version: 3.0
supported: true
annotator: ChunkMapperModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained model maps RxNorm codes with corresponding UMLS Codes.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/rxnorm_umls_mapper_en_3.5.3_3.0_1656088714126.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/rxnorm_umls_mapper_en_3.5.3_3.0_1656088714126.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings\
.pretrained("sbiobert_base_cased_mli", "en","clinical/models")\
.setInputCols(["ner_chunk"])\
.setOutputCol("sbert_embeddings")

rxnorm_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_rxnorm_augmented", "en", "clinical/models")\
.setInputCols(["ner_chunk", "sbert_embeddings"])\
.setOutputCol("rxnorm_code")\
.setDistanceFunction("EUCLIDEAN")

chunkerMapper = ChunkMapperModel.pretrained("rxnorm_umls_mapper", "en", "clinical/models")\
.setInputCols(["rxnorm_code"])\
.setOutputCol("mappings")\
.setRels(["umls_code"])


pipeline = Pipeline(
stages = [
documentAssembler,
sbert_embedder,
rxnorm_resolver,
chunkerMapper
])

model = pipeline.fit(spark.createDataFrame([['']]).toDF('text')) 

lp= LightPipeline(model)
result = lp.fullAnnotate(["amlodipine 5 MG", "hydrochlorothiazide 25 MG"])
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli", "en","clinical/models")
.setInputCols(Array("ner_chunk"))
.setOutputCol("sbert_embeddings")

val rxnorm_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_rxnorm_augmented", "en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("rxnorm_code")
.setDistanceFunction("EUCLIDEAN")

val chunkerMapper = ChunkMapperModel.pretrained("rxnorm_umls_mapper", "en", "clinical/models")
.setInputCols(Array("rxnorm_code"))
.setOutputCol("mappings")
.setRels(Array("umls_code"))

val pipeline = new Pipeline(stages = Array(documentAssembler,
sbert_embedder,
rxnorm_resolver,
chunkerMapper)

val model = pipeline.fit(Seq("").toDF("text"))

val lp= LightPipeline(model)
val result = lp.fullAnnotate(Seq("amlodipine 5 MG", "hydrochlorothiazide 25 MG"))
```


{:.nlu-block}
```python
import nlu
nlu.load("en.rxnorm_to_umls").predict("""hydrochlorothiazide 25 MG""")
```

</div>

## Results

```bash
|    | chunk                     |   rxnorm_code | umls_mappings   |
|---:|:--------------------------|--------------:|:----------------|
|  0 | amlodipine 5 MG           |        329528 | C1124796        |
|  1 | hydrochlorothiazide 25 MG |        310798 | C0977518        |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|rxnorm_umls_mapper|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[rxnorm_code]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|1.9 MB|