---
layout: model
title: Mapping RxNorm Codes with Corresponding Actions and Treatments
author: John Snow Labs
name: rxnorm_action_treatment_mapper
date: 2022-06-27
tags: [chunk_mapper, action, treatment, clinical, licensed, en]
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

This pretrained model maps RxNorm and RxNorm Extension codes with their corresponding action and treatment. Action refers to the function of the drug in various body systems; treatment refers to which disease the drug is used to treat.

## Predicted Entities

`action`, `treatment`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/rxnorm_action_treatment_mapper_en_3.5.3_3.0_1656315389520.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/rxnorm_action_treatment_mapper_en_3.5.3_3.0_1656315389520.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

chunkerMapper_1 = ChunkMapperModel\
.pretrained("rxnorm_action_treatment_mapper", "en", "clinical/models")\
.setInputCols(["rxnorm_code"])\
.setOutputCol("action_mappings")\
.setRels(["action"])

chunkerMapper_2 = ChunkMapperModel\
.pretrained("rxnorm_action_treatment_mapper", "en", "clinical/models")\
.setInputCols(["rxnorm_code"])\
.setOutputCol("treatment_mappings")\
.setRels(["treatment"])

pipeline = Pipeline(stages = [
documentAssembler,
sbert_embedder,
rxnorm_resolver,
chunkerMapper_1,
chunkerMapper_2
])

model = pipeline.fit(spark.createDataFrame([['']]).toDF('text'))

pipeline = LightPipeline(model)

result = pipeline.fullAnnotate(['Sinequan 150 MG', 'Zonalon 50 mg'])
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")
.setInputCols("ner_chunk")
.setOutputCol("sbert_embeddings")

val rxnorm_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_rxnorm_augmented", "en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("rxnorm_code")
.setDistanceFunction("EUCLIDEAN")

val chunkerMapper_1 = ChunkMapperModel
.pretrained("rxnorm_action_treatment_mapper", "en", "clinical/models")
.setInputCols("rxnorm_code")
.setOutputCol("action_mappings")
.setRels(["action"])

val chunkerMapper_2 = ChunkMapperModel
.pretrained("rxnorm_action_treatment_mapper", "en", "clinical/models")
.setInputCols("rxnorm_code")
.setOutputCol("treatment_mappings")
.setRels(["treatment"])

val pipeline = new Pipeline(stages = Array(
documentAssembler,
sbert_embedder,
rxnorm_resolver,
chunkerMapper_1,
chunkerMapper_2
))

val data = Seq(Array("Sinequan 150 MG", "Zonalon 50 mg")).toDS.toDF("text")

val result= pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.map_entity.rxnorm_to_action_treatment").predict("""Sinequan 150 MG""")
```

</div>

## Results

```bash
|    | ner_chunk           | rxnorm_code   | Action                                                    | Treatment                                                              |
|---:|:--------------------|:--------------|:----------------------------------------------------------|:-----------------------------------------------------------------------|
|  0 | ['Sinequan 150 MG'] | ['1000067']   | ['Anxiolytic', 'Psychoanaleptics', 'Sedative']            | ['Depression', 'Neurosis', 'Anxiety&Panic Attacks', 'Psychosis']       |
|  1 | ['Zonalon 50 mg']   | ['103971']    | ['Analgesic (Opioid)', 'Analgetic', 'Opioid', 'Vitamins'] | ['Pain']                                                               |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|rxnorm_action_treatment_mapper|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[rxnorm_code]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|21.1 MB|
