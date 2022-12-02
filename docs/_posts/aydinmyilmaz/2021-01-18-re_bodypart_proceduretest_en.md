---
layout: model
title: Relation extraction between body parts and procedures
author: John Snow Labs
name: re_bodypart_proceduretest
date: 2021-01-18
task: Relation Extraction
language: en
edition: Spark NLP for Healthcare 2.7.1
spark_version: 2.4
tags: [en, relation_extraction, clinical, licensed]
supported: true
annotator: RelationExtractionModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Relation extraction between body parts entites ['Internal_organ_or_component','External_body_part_or_region'] and procedure and test entities

## Predicted Entities

`0`, `1`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_BODYPART_ENT/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb#scrollTo=D8TtVuN-Ee8s){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_bodypart_proceduretest_en_2.7.1_2.4_1610989267602.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

In the table below, `re_bodypart_proceduretest` RE model, its labels, optimal NER model, and meaningful relation pairs are illustrated.



|          RE MODEL         | RE MODEL LABES | NER MODEL | RE PAIRS |
|:-------------------------:|:--------------:|:---------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| re_bodypart_proceduretest |       0,1      |  ner_jsl  | [“external_body_part_or_region-test”,<br>“test-external_body_part_or_region”,<br>“internal_organ_or_component-test”,<br>“test-internal_organ_or_component”,<br>“external_body_part_or_region-procedure”,<br>“procedure-external_body_part_or_region”,<br>“procedure-internal_organ_or_component”,<br>“internal_organ_or_component-procedure”] |




Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, PerceptronModel, DependencyParserModel, WordEmbeddingsModel, NerDLModel, NerConverter, RelationExtractionModel.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentencer = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentences")

tokenizer = Tokenizer()\
    .setInputCols(["sentences"])\
    .setOutputCol("tokens")
  
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("embeddings")

pos_tagger = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("pos_tags")

ner_tagger = MedicalNerModel()\
    .pretrained("jsl_ner_wip_greedy_clinical","en","clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags") 

ner_chunker = NerConverterInternal()\
    .setInputCols(["sentences", "tokens", "ner_tags"])\
    .setOutputCol("ner_chunks")

dependency_parser = DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["sentences", "pos_tags", "tokens"])\
    .setOutputCol("dependencies")

re_model = RelationExtractionModel()\
    .pretrained("re_bodypart_proceduretest", "en", "clinical/models")\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(4)\
    .setPredictionThreshold(0.9)\
    .setRelationPairs(["external_body_part_or_region-test"]) # Possible relation pairs. Default: All Relations.

nlp_pipeline = Pipeline(stages=[documenter, sentencer,tokenizer, word_embeddings, pos_tagger, ner_tagger, ner_chunker, dependency_parser, re_model])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate('''TECHNIQUE IN DETAIL: After informed consent was obtained from the patient and his mother, the chest was scanned with portable ultrasound.''')
```

```scala

val documenter = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentencer = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentences")

val tokenizer = new Tokenizer()
    .setInputCols("sentences")
    .setOutputCol("tokens")
  
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("embeddings")

val pos_tagger = PerceptronModel()
    .pretrained("pos_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("pos_tags")

val ner_tagger = MedicalNerModel().pretrained("jsl_ner_wip_greedy_clinical","en","clinical/models")
    .setInputCols(Array("sentences", "tokens", "embeddings"))
    .setOutputCol("ner_tags")

val ner_chunker = new NerConverterInternal()
    .setInputCols(Array("sentences", "tokens", "ner_tags"))
    .setOutputCol("ner_chunks")

val dependency_parser = DependencyParserModel()
    .pretrained("dependency_conllu", "en")
    .setInputCols(Array("sentences", "pos_tags", "tokens"))
    .setOutputCol("dependencies")

val re_model = RelationExtractionModel().pretrained("re_bodypart_proceduretest", "en", "clinical/models")
    .setInputCols(Array("embeddings", "pos_tags", "ner_chunks", "dependencies"))
    .setOutputCol("relations")
    .setMaxSyntacticDistance(4) #default: 0
    .setPredictionThreshold(0.9) #default: 0.5
    .setRelationPairs(Array("external_body_part_or_region-test")) # Possible relation pairs. Default: All Relations.

val nlpPipeline = new Pipeline().setStages(Array(documenter, sentencer,tokenizer, word_embeddings, pos_tagger, ner_tagger, ner_chunker, dependency_parser, re_model))

val result = pipeline.fit(Seq.empty[String]).transform(data)

val annotations = light_pipeline.fullAnnotate("""TECHNIQUE IN DETAIL: After informed consent was obtained from the patient and his mother, the chest was scanned with portable ultrasound.""")
```

</div>

## Results

```bash
| index | relations | entity1                      | entity1_begin | entity1_end | chunk1 | entity2 | entity2_end | entity2_end | chunk2              | confidence |
|-------|-----------|------------------------------|---------------|-------------|--------|---------|-------------|-------------|---------------------|------------|
| 0     | 1         | External_body_part_or_region | 94            | 98          | chest  | Test    | 117         | 135         | portable ultrasound | 1.0        |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_bodypart_proceduretest|
|Type:|re|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[embeddings, pos_tags, train_ner_chunks, dependencies]|
|Output Labels:|[relations]|
|Language:|en|
|Dependencies:|embeddings_clinical|

## Data Source

Trained on data gathered and manually annotated by John Snow Labs

## Benchmarking

```bash
label  recall  precision  f1   
0      0.55    0.35       0.43 
1      0.73    0.86       0.79 

```

