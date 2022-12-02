---
layout: model
title: Relation extraction between Drugs and ADE - Conversational Text
author: John Snow Labs
name: re_ade_conversational
date: 2022-07-27
tags: [relation_extraction, licensed, clinical, en]
task: Relation Extraction
language: en
edition: Healthcare NLP 3.5.0
spark_version: 3.0
supported: true
annotator: RelationExtractionModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is capable of Relating Drugs and adverse reactions caused by them in conversational text.

## Predicted Entities

`is_related`, `not_related`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_ADE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_ade_conversational_en_3.5.0_3.0_1658956087191.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("sentences")

tokenizer = Tokenizer()\
    .setInputCols(["sentences"])\
    .setOutputCol("tokens")

words_embedder = WordEmbeddingsModel() \
    .pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"]) \
    .setOutputCol("embeddings")

ner_tagger = MedicalNerModel() \
    .pretrained("ner_ade_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens", "embeddings"]) \
    .setOutputCol("ner_tags")

ner_converter = NerConverter() \
    .setInputCols(["sentences", "tokens", "ner_tags"]) \
    .setOutputCol("ner_chunks")

pos_tagger = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("pos_tags")

dependency_parser = sparknlp.annotators.DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["sentences", "pos_tags", "tokens"])\
    .setOutputCol("dependencies")

re_model = RelationExtractionModel()\
    .pretrained("re_ade_conversational", "en", "clinical/models")\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setRelationPairs(["ade-drug", "drug-ade"]) # Possible relation pairs. Default: All Relations.

nlp_pipeline = Pipeline(stages=[documenter, tokenizer, words_embedder, pos_tagger, ner_tagger, ner_converter, dependency_parser, re_model])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text")))

text ="""19.32 day 20 rivaroxaban diary. still residual aches and pains; only had 4 paracetamol today."""

annotations = light_pipeline.fullAnnotate(text)
```
```scala
val documenter = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("sentences")

val tokenizer = new Tokenizer()
    .setInputCols("sentences")
    .setOutputCol("tokens")

val words_embedder = WordEmbeddingsModel()
    .pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("embeddings")

val ner_tagger = NerDLModel()
    .pretrained("ner_ade_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens", "embeddings"))
    .setOutputCol("ner_tags")

val ner_converter = new NerConverter()
    .setInputCols(Array("sentences", "tokens", "ner_tags"))
    .setOutputCol("ner_chunks")

val pos_tagger = PerceptronModel()
    .pretrained("pos_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("pos_tags")

val dependency_parser = DependencyParserModel()
    .pretrained("dependency_conllu", "en")
    .setInputCols(Array("sentences", "pos_tags", "tokens"))
    .setOutputCol("dependencies")

val re_model = RelationExtractionModel()
    .pretrained("re_ade_conversational", "en", "clinical/models")
    .setInputCols(Array("embeddings", "pos_tags", "ner_chunks", "dependencies"))
    .setOutputCol("relations")
    .setMaxSyntacticDistance(3) #default: 0 
    .setPredictionThreshold(0.5) #default: 0.5 
    .setRelationPairs(Array("drug-ade", "ade-drug")) # Possible relation pairs. Default: All Relations.

val nlpPipeline = new Pipeline().setStages(Array(documenter, tokenizer, words_embedder, pos_tagger, ner_tagger, ner_chunker, dependency_parser, re_model))

val data = Seq("""19.32 day 20 rivaroxaban diary. still residual aches and pains; only had 4 paracetamol today.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
|    | chunk1                        | entitiy1   | chunk2      | entity2 | relation    |
|----|-------------------------------|------------|-------------|---------|-------------|
| 0  | residual aches and pains      | ADE        | rivaroxaban | DRUG    | is_related  |
| 1  | residual aches and pains      | ADE        | paracetamol | DRUG    | not_related |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_ade_conversational|
|Type:|re|
|Compatibility:|Healthcare NLP 3.5.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[embeddings, pos_tags, train_ner_chunks, dependencies]|
|Output Labels:|[relations]|
|Language:|en|
|Size:|11.3 MB|

## References

Trained on SMM4H dataset - annotated manually. https://healthlanguageprocessing.org/smm4h-2022/

## Benchmarking

```bash
       label  precision    recall  f1-score   support
 not_related       0.81      0.88      0.85       528
  is_related       0.94      0.89      0.91      1019
    accuracy       -         -         0.89      1547
   macro-avg       0.87      0.89      0.88      1547
weighted-avg       0.89      0.89      0.89      1547
```
