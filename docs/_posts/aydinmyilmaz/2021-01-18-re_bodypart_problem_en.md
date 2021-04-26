---
layout: model
title: Relation extraction between body parts and problem entities
author: John Snow Labs
name: re_bodypart_problem
date: 2021-01-18
task: Relation Extraction
language: en
edition: Spark NLP for Healthcare 2.7.1
tags: [en, clinical, relation_extraction, licensed]
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Relation extraction between body parts and problem entities  in clinical texts

## Predicted Entities

 `1` : Shows that there is a relation between the body part  entity and the entities labeled as problem ( diognosis, symptom etc.)
 `0` : Shows that there no  relation between the body part entity and the entities labeled as problem ( diognosis, symptom etc.)

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_bodypart_problem_en_2.7.1_2.4_1610959377894.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

ner_tagger = sparknlp.annotators.NerDLModel()\
    .pretrained('jsl_ner_wip_greedy_clinical','en','clinical/models')\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags") 

reModel = RelationExtractionModel.pretrained("re_bodypart_problem","en","clinical/models")\
    .setInputCols(["word_embeddings","chunk","pos","dependency"])\
    .setOutput("relations")
    .setRelationPairs(['symptom-external_body_part_or_region'])

pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, words_embedder, pos_tagger, ner_tagger, ner_chunker, dependency_parser, reModel)

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = LightPipeline(model).fullAnnotate('''No neurologic deficits other than some numbness in his left hand.''')
```

```scala
...
val ner_tagger = sparknlp.annotators.NerDLModel()
    .pretrained('jsl_ner_wip_greedy_clinical','en','clinical/models')
    .setInputCols("sentences", "tokens", "embeddings")
    .setOutputCol("ner_tags") 

val reModel = RelationExtractionModel().pretrained("re_bodypart_problem","en","clinical/models")
    .setInputCols(Array("word_embeddings","chunk","pos","dependency"))
    .setOutput("relations")
    .setRelationPairs(Array('symptom-external_body_part_or_region'))

val nlpPipeline = new Pipeline().setStages(Array(documenter, sentencer, tokenizer, words_embedder, pos_tagger, ner_tagger, ner_chunker, dependency_parser, reModel))
val model = nlpPipeline.fit(Seq.empty[""].toDS.toDF("text"))

val results = LightPipeline(model).fullAnnotate('''No neurologic deficits other than some numbness in his left hand.''')
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
| relation | recall | precision |
|----------|--------|-----------|
| 0        | 0.72   | 0.82      |
| 1        | 0.94   | 0.91      |

```
