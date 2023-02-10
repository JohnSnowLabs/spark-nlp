---
layout: model
title: Mapping RxNorm and RxNorm Extension Codes with Corresponding Drug Brand Names
author: John Snow Labs
name: rxnorm_drug_brandname_mapper
date: 2023-02-09
tags: [chunk_mappig, rxnorm, drug_brand_name, rxnorm_extension, en, clinical, licensed]
task: Chunk Mapping
language: en
edition: Healthcare NLP 4.3.0
spark_version: 3.0
supported: true
annotator: ChunkMapperModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained model maps RxNorm and RxNorm Extension codes with their corresponding drug brand names. It returns 2 types of brand names for the corresponding RxNorm or RxNorm Extension code.

## Predicted Entities

`rxnorm_brandname`, `rxnorm_extension_brandname`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/rxnorm_drug_brandname_mapper_en_4.3.0_3.0_1675966478332.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/rxnorm_drug_brandname_mapper_en_4.3.0_3.0_1675966478332.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("chunk")

sbert_embedder = BertSentenceEmbeddings\
      .pretrained("sbiobert_base_cased_mli", "en","clinical/models")\
      .setInputCols(["chunk"])\
      .setOutputCol("sbert_embeddings")
    
rxnorm_resolver = SentenceEntityResolverModel\
      .pretrained("sbiobertresolve_rxnorm_augmented", "en", "clinical/models")\
      .setInputCols(["chunk", "sbert_embeddings"])\
      .setOutputCol("rxnorm_code")\
      .setDistanceFunction("EUCLIDEAN")

resolver2chunk = Resolution2Chunk()\
    .setInputCols(["rxnorm_code"]) \
    .setOutputCol("rxnorm_chunk")\

chunkerMapper = ChunkMapperModel.pretrained("rxnorm_drug_brandname_mapper", "en", "clinical/models")\
      .setInputCols(["rxnorm_chunk"])\
      .setOutputCol("mappings")\
      .setRels(["rxnorm_brandname", "rxnorm_extension_brandname"])


pipeline = Pipeline(
    stages = [
        documentAssembler,
        sbert_embedder,
        rxnorm_resolver,
        resolver2chunk,
        chunkerMapper
        ])

model = pipeline.fit(spark.createDataFrame([['']]).toDF('text')) 

pipeline = LightPipeline(model)

result = pipeline.fullAnnotate(['metformin', 'advil'])

```
```scala
val documentAssembler = new DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("chunk")

val sbert_embedder = BertSentenceEmbeddings\
      .pretrained("sbiobert_base_cased_mli", "en","clinical/models")\
      .setInputCols(["chunk"])\
      .setOutputCol("sbert_embeddings")
    
val rxnorm_resolver = SentenceEntityResolverModel\
      .pretrained("sbiobertresolve_rxnorm_augmented", "en", "clinical/models")\
      .setInputCols(["chunk", "sbert_embeddings"])\
      .setOutputCol("rxnorm_code")\
      .setDistanceFunction("EUCLIDEAN")

val resolver2chunk = new Resolution2Chunk()\
    .setInputCols(["rxnorm_code"]) \
    .setOutputCol("rxnorm_chunk")\

val chunkerMapper = ChunkMapperModel.pretrained("rxnorm_drug_brandname_mapper", "en", "clinical/models")\
      .setInputCols(["rxnorm_chunk"])\
      .setOutputCol("mappings")\
      .setRels(["rxnorm_brandname", "rxnorm_extension_brandname"])



val pipeline = new Pipeline(stages = Array(
documentAssembler,
sbert_embedder,
rxnorm_resolver,
resolver2chunk
chunkerMapper
))

val data = Seq(Array("metformin", "advil")).toDS.toDF("text")

val result= pipeline.fit(data).transform(data)

```
</div>

## Results

```bash
+--------------+-------------+--------------------------------------------------+--------------------------+
|     drug_name|rxnorm_result|                                    mapping_result|                 relation |
+--------------+-------------+--------------------------------------------------+--------------------------+
|     metformin|         6809|Actoplus Met (metformin):::Avandamet (metformin...|          rxnorm_brandname|
|     metformin|         6809|A FORMIN (metformin):::ABERIN MAX (metformin)::...|rxnorm_extension_brandname|
|         advil|       153010|                                     Advil (Advil)|          rxnorm_brandname|
|         advil|       153010|                                              NONE|rxnorm_extension_brandname|
+--------------+-------------+--------------------------------------------------+--------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|rxnorm_drug_brandname_mapper|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[rxnorm_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|4.0 MB|