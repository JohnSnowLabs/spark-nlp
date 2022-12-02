---
layout: model
title: Relation extraction between body parts and problem entities
author: John Snow Labs
name: re_bodypart_problem
date: 2021-01-18
task: Relation Extraction
language: en
edition: Spark NLP for Healthcare 2.7.1
spark_version: 2.4
tags: [en, clinical, relation_extraction, licensed]
supported: true
annotator: RelationExtractionModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Relation extraction between body parts and problem entities  in clinical texts.  `1` : Shows that there is a relation between the body part  entity and the entities labeled as problem ( diagnosis, symptom etc.), `0` : Shows that there no  relation between the body part entity and the entities labeled as problem ( diagnosis, symptom etc.).

## Predicted Entities

`0`, `1`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_BODYPART_ENT/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_bodypart_problem_en_2.7.1_2.4_1610959377894.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

In the table below, `re_bodypart_problem` RE model, its labels, optimal NER model, and meaningful relation pairs are illustrated.

|       RE MODEL      | RE MODELS LABES | NER MODEL | RE PAIRS                                                                         |
|:-------------------:|:---------------:|:---------:|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| re_bodypart_problem |       0,1       |  ner_jsl  | [“internal_organ_or_component-cerebrovascular_disease”, <br>“cerebrovascular_disease-internal_organ_or_component”,<br>“internal_organ_or_component-communicable_disease”, <br>“communicable_disease-internal_organ_or_component”,<br>“internal_organ_or_component-diabetes”, <br>“diabetes-internal_organ_or_component”,<br>“internal_organ_or_component-disease_syndrome_disorder”, <br>“disease_syndrome_disorder-internal_organ_or_component”,<br>“internal_organ_or_component-ekg_findings”, <br>“ekg_findings-internal_organ_or_component”,<br>“internal_organ_or_component-heart_disease”, <br>“heart_disease-internal_organ_or_component”,<br>“internal_organ_or_component-hyperlipidemia”, <br>“hyperlipidemia-internal_organ_or_component”,<br>“internal_organ_or_component-hypertension”, <br>“hypertension-internal_organ_or_component”,<br>“internal_organ_or_component-imagingfindings”, <br>“imagingfindings-internal_organ_or_component”,<br>“internal_organ_or_component-injury_or_poisoning”, <br>“injury_or_poisoning-internal_organ_or_component”,<br>“internal_organ_or_component-kidney_disease”, <br>“kidney_disease-internal_organ_or_component”,<br>“internal_organ_or_component-oncological”, <br>“oncological-internal_organ_or_component”,<br>“internal_organ_or_component-psychological_condition”, <br>“psychological_condition-internal_organ_or_component”,<br>“internal_organ_or_component-symptom”, <br>“symptom-internal_organ_or_component”,<br>“internal_organ_or_component-vs_finding”, <br>“vs_finding-internal_organ_or_component”,<br>“external_body_part_or_region-communicable_disease”, <br>“communicable_disease-external_body_part_or_region”,<br>“external_body_part_or_region-diabetes”,<br>“diabetes-external_body_part_or_region”,<br>“external_body_part_or_region-disease_syndrome_disorder”, <br>“disease_syndrome_disorder-external_body_part_or_region”,<br>“external_body_part_or_region-hypertension”, <br>“hypertension-external_body_part_or_region”,<br>“external_body_part_or_region-imagingfindings”, <br>“imagingfindings-external_body_part_or_region”,<br>“external_body_part_or_region-injury_or_poisoning”, <br>“injury_or_poisoning-external_body_part_or_region”,<br>“external_body_part_or_region-obesity”, <br>“obesity-external_body_part_or_region”,<br>“external_body_part_or_region-oncological”, <br>“oncological-external_body_part_or_region”,<br>“external_body_part_or_region-overweight”, <br>“overweight-external_body_part_or_region”,<br>“external_body_part_or_region-symptom”, <br>“symptom-external_body_part_or_region”,<br>“external_body_part_or_region-vs_finding”, <br>“vs_finding-external_body_part_or_region”] |



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

reModel = RelationExtractionModel.pretrained("re_bodypart_problem","en","clinical/models")\
    .setInputCols(["embeddings","ner_chunks","pos_tags","dependencies"])\
    .setOutputCol("relations") \
    .setRelationPairs(['symptom-external_body_part_or_region'])

pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, word_embeddings, pos_tagger, ner_tagger, ner_chunker, dependency_parser, reModel])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = LightPipeline(model).fullAnnotate('''No neurologic deficits other than some numbness in his left hand.''')
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

val ner_tagger = MedicalNerModel()
    .pretrained("jsl_ner_wip_greedy_clinical","en","clinical/models")
    .setInputCols(Array("sentences", "tokens", "embeddings"))
    .setOutputCol("ner_tags") 

val ner_chunker = new NerConverterInternal()
    .setInputCols(Array("sentences", "tokens", "ner_tags"))
    .setOutputCol("ner_chunks")

val dependency_parser = DependencyParserModel()
    .pretrained("dependency_conllu", "en")
    .setInputCols(Array("sentences", "pos_tags", "tokens"))
    .setOutputCol("dependencies")

val reModel = RelationExtractionModel().pretrained("re_bodypart_problem","en","clinical/models")
    .setInputCols(Array("embeddings","ner_chunks","pos_tags","dependencies"))
    .setOutput("relations")
    .setRelationPairs(Array("symptom-external_body_part_or_region"))

val nlpPipeline = new Pipeline().setStages(Array(documenter, sentencer, tokenizer, word_embeddings, pos_tagger, ner_tagger, ner_chunker, dependency_parser, reModel))

val result = pipeline.fit(Seq.empty[String]).transform(data)

val results = LightPipeline(model).fullAnnotate("""No neurologic deficits other than some numbness in his left hand.""")
```

</div>

## Results

```bash
| index | relations | entity1 | entity1_begin | entity1_end | chunk1              | entity2                      | entity2_end | entity2_end | chunk2 | confidence |
|-------|-----------|---------|---------------|-------------|---------------------|------------------------------|-------------|-------------|--------|------------|
| 0     | 0         | Symptom | 3             | 21          | neurologic deficits | external_body_part_or_region | 60          | 63          | hand   | 0.999998   |
| 1     | 1         | Symptom | 39            | 46          | numbness            | external_body_part_or_region | 60          | 63          | hand   | 1          |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_bodypart_problem|
|Type:|re|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[embeddings, pos_tags, train_ner_chunks, dependencies]|
|Output Labels:|[relations]|
|Language:|en|
|Dependencies:|embeddings_clinical|

## Data Source

Trained on custom datasets annotated internally

## Benchmarking

```bash
label  recall  precision
0      0.72    0.82     
1      0.94    0.91     
```
