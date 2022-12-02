---
layout: model
title: Sentence Entity Resolver for RxNorm (sbiobert_base_cased_mli - EntityChunkEmbeddings)
author: John Snow Labs
name: sbiobertresolve_rxnorm_augmented_re
date: 2022-02-09
tags: [rxnorm, licensed, en, clinical, entity_resolution]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.4.0
spark_version: 2.4
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities and concepts (like drugs/ingredients) to RxNorm codes without specifying the relations between the entities (relations are calculated on the fly inside the annotator) using sbiobert_base_cased_mli Sentence Bert Embeddings (EntityChunkEmbeddings). Embeddings used in this model are calculated with following weights : `{"DRUG": 0.8, "STRENGTH": 0.2, "ROUTE": 0.2, "FORM": 0.2}` . EntityChunkEmbeddings with those weights are required in the pipeline to get best result.

## Predicted Entities

`RxNorm Codes`, `Concept Classes`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_rxnorm_augmented_re_en_3.4.0_2.4_1644395696788.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documenter = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("documents")

sentence_detector = SentenceDetector() \
.setInputCols("documents") \
.setOutputCol("sentences")

tokenizer = Tokenizer() \
.setInputCols("sentences") \
.setOutputCol("tokens")

embeddings = WordEmbeddingsModel() \
.pretrained("embeddings_clinical", "en", "clinical/models")\
.setInputCols(["sentences", "tokens"])\
.setOutputCol("embeddings")

posology_ner_model = MedicalNerModel()\
.pretrained("ner_posology_large", "en", "clinical/models")\
.setInputCols(["sentences", "tokens", "embeddings"])\
.setOutputCol("ner")

ner_converter = NerConverterInternal()\
.setInputCols(["sentences", "tokens", "ner"])\
.setOutputCol("ner_chunks")

pos_tager = PerceptronModel()\
.pretrained("pos_clinical", "en", "clinical/models")\
.setInputCols(["sentences", "tokens"])\
.setOutputCol("pos_tags")

dependency_parser = DependencyParserModel()\
.pretrained("dependency_conllu", "en")\
.setInputCols(["sentences", "pos_tags", "tokens"])\
.setOutputCol("dependencies")

drug_chunk_embeddings = EntityChunkEmbeddings()\
.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
.setInputCols(["ner_chunks", "dependencies"])\
.setOutputCol("drug_chunk_embeddings")\
.setMaxSyntacticDistance(3)\
.setTargetEntities({"DRUG": ["STRENGTH", "ROUTE", "FORM"]})\
.setEntityWeights({"DRUG": 0.8, "STRENGTH": 0.2, "ROUTE": 0.2, "FORM": 0.2})

rxnorm_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_rxnorm_augmented_re", "en", "clinical/models")\
.setInputCols(["drug_chunk_embeddings"])\
.setOutputCol("rxnorm_code")\
.setDistanceFunction("EUCLIDEAN")

rxnorm_weighted_pipeline_re = Pipeline(
stages = [
documenter,
sentence_detector,
tokenizer,
embeddings,
posology_ner_model,
ner_converter,
pos_tager,
dependency_parser,
drug_chunk_embeddings,
rxnorm_resolver
])

sampleText = ["The patient was given metformin 500 mg, 2.5 mg of coumadin and then ibuprofen.",
"The patient was given metformin 400 mg, coumadin 5 mg, coumadin, amlodipine 10 MG"]

data_df = spark.createDataFrame(sample_df)

results = rxnorm_weighted_pipeline_re.fit(data_df).transform(data_df)

```
```scala
val documenter = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("documents")

val sentence_detector = SentenceDetector() 
.setInputCols("documents") 
.setOutputCol("sentences")

val tokenizer = Tokenizer() 
.setInputCols("sentences") 
.setOutputCol("tokens")

val embeddings = WordEmbeddingsModel() 
.pretrained("embeddings_clinical", "en", "clinical/models")
.setInputCols(Array("sentences", "tokens"))
.setOutputCol("embeddings")

val posology_ner_model = MedicalNerModel()
.pretrained("ner_posology_large", "en", "clinical/models")
.setInputCols(Array("sentences", "tokens", "embeddings"))
.setOutputCol("ner")

val ner_converter = NerConverterInternal()
.setInputCols(Array("sentences", "tokens", "ner"))
.setOutputCol("ner_chunks")

val pos_tager = PerceptronModel()
.pretrained("pos_clinical", "en", "clinical/models")
.setInputCols(Array("sentences", "tokens"))
.setOutputCol("pos_tags")

val dependency_parser = DependencyParserModel()
.pretrained("dependency_conllu", "en")
.setInputCols(Array("sentences", "pos_tags", "tokens"))
.setOutputCol("dependencies")

val drug_chunk_embeddings = EntityChunkEmbeddings()
.pretrained("sbiobert_base_cased_mli","en","clinical/models")
.setInputCols(Array("ner_chunks", "dependencies"))
.setOutputCol("drug_chunk_embeddings")
.setMaxSyntacticDistance(3)
.setTargetEntities({"DRUG": ["STRENGTH", "ROUTE", "FORM"]})
.setEntityWeights({"DRUG": 0.8, "STRENGTH": 0.2, "ROUTE": 0.2, "FORM": 0.2}

val rxnorm_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_rxnorm_augmented_re", "en", "clinical/models")
.setInputCols(Array("drug_chunk_embeddings"))
.setOutputCol("rxnorm_code")
.setDistanceFunction("EUCLIDEAN")

val rxnorm_weighted_pipeline_re = new PipelineModel().setStages(Array(documenter, sentence_detector, tokenizer, embeddings, posology_ner_model, 
ner_converter,  pos_tager, dependency_parser, drug_chunk_embeddings, rxnorm_re))

val light_model = LightPipeline(rxnorm_weighted_pipeline_re)

vat sampleText = Array("The patient was given metformin 500 mg, 2.5 mg of coumadin and then ibuprofen.",
"The patient was given metformin 400 mg, coumadin 5 mg, coumadin, amlodipine 10 MG")

val results = rxnorm_weighted_pipeline_re.fit(sampleText).transform(sampleText)

```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.rxnorm.augmented_re").predict("""The patient was given metformin 400 mg, coumadin 5 mg, coumadin, amlodipine 10 MG""")
```

</div>

## Results

```bash
+-----+----------------+--------------------------+--------------------------------------------------+
|index|           chunk|rxnorm_code_weighted_08_re|                                      Concept_Name|
+-----+----------------+--------------------------+--------------------------------------------------+
|    0|metformin 500 mg|                    860974|metformin hydrochloride 500 MG:::metformin 500 ...|
|    0| 2.5 mg coumadin|                    855313|warfarin sodium 2.5 MG [Coumadin]:::warfarin so...|
|    0|       ibuprofen|                   1747293|ibuprofen Injection:::ibuprofen Pill:::ibuprofe...|
|    1|metformin 400 mg|                    332809|metformin 400 MG:::metformin 250 MG Oral Tablet...|
|    1|   coumadin 5 mg|                    855333|warfarin sodium 5 MG [Coumadin]:::warfarin sodi...|
|    1|        coumadin|                    202421|Coumadin:::warfarin sodium 2 MG/ML Injectable S...|
|    1|amlodipine 10 MG|                    308135|amlodipine 10 MG Oral Tablet:::amlodipine 10 MG...|
+-----+----------------+--------------------------+--------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_rxnorm_augmented_re|
|Compatibility:|Healthcare NLP 3.4.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[rxnorm_code]|
|Language:|en|
|Size:|759.7 MB|
|Case sensitive:|false|
