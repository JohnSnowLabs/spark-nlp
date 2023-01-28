---
layout: model
title: Athena Conditions Entity Resolver (Healthcare)
author: John Snow Labs
name: chunkresolve_athena_conditions_healthcare
class: ChunkEntityResolverModel
language: en
repository: clinical/models
date: 2020-09-16
task: Entity Resolution
edition: Healthcare NLP 2.6.0
spark_version: 2.4
tags: [clinical,licensed,entity_resolution,en]
deprecated: true
annotator: ChunkEntityResolverModel
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance.


## Predicted Entities
Athena Codes and their normalized definition.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_athena_conditions_healthcare_en_2.6.0_2.4_1600265258887.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/chunkresolve_athena_conditions_healthcare_en_2.6.0_2.4_1600265258887.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

{:.h2_title}
## How to use
This model requires `embeddings_healthcare_100d` and `ner_healthcare` in the pipeline you use.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...
athena_re_model = ChunkEntityResolverModel.pretrained("chunkresolve_athena_conditions_healthcare","en","clinical/models")\
	.setInputCols("token","chunk_embeddings")\
	.setOutputCol("entity")

pipeline_athena = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, word_embeddings, ner_model, ner_converter, chunk_embeddings, athena_re_model])

model = pipeline_athena.fit(spark.createDataFrame([["""The patient is a 5-month-old infant who presented initially on Monday with a cold, cough, and runny nose for 2 days. Mom states she had no fever. Her appetite was good but she was spitting up a lot. She had no difficulty breathing and her cough was described as dry and hacky. At that time, physical exam showed a right TM, which was red. Left TM was okay. She was fairly congested but looked happy and playful. She was started on Amoxil and Aldex and we told to recheck in 2 weeks to recheck her ear. Mom returned to clinic again today because she got much worse overnight. She was having difficulty breathing. She was much more congested and her appetite had decreased significantly today. She also spiked a temperature yesterday of 102.6 and always having trouble sleeping secondary to congestion."""]]).toDF("text"))

results = model.transform(data)
```

```scala
val athena_re_model = ChunkEntityResolverModel.pretrained("chunkresolve_athena_conditions_healthcare","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, word_embeddings, ner_model, ner_converter, chunk_embeddings, athena_re_model))

val data = Seq("The patient is a 5-month-old infant who presented initially on Monday with a cold, cough, and runny nose for 2 days. Mom states she had no fever. Her appetite was good but she was spitting up a lot. She had no difficulty breathing and her cough was described as dry and hacky. At that time, physical exam showed a right TM, which was red. Left TM was okay. She was fairly congested but looked happy and playful. She was started on Amoxil and Aldex and we told to recheck in 2 weeks to recheck her ear. Mom returned to clinic again today because she got much worse overnight. She was having difficulty breathing. She was much more congested and her appetite had decreased significantly today. She also spiked a temperature yesterday of 102.6 and always having trouble sleeping secondary to congestion.").toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

{:.h2_title}
## Results
```bash

    chunk                    entity                          athena_description  athena_code

0   a cold                  PROBLEM                          Intolerant of cold  4213725
1   cough                   PROBLEM                                       Cough  254761
2   runny nose              PROBLEM                                  O/E - nose  4156058
3   fever                   PROBLEM                                       Fever  437663
4   difficulty breathing    PROBLEM                        Difficulty breathing  4041664
5   her cough               PROBLEM                                  Does cough  4122567
6   dry                     PROBLEM                                    Dry eyes  4036620
7   hacky                   PROBLEM    Resolving infantile idiopathic scoliosis  44833868
8   physical exam              TEST                         Physical angioedema  37110554
9   a right TM              PROBLEM  Tuberculosis of thyroid gland, unspecified  44819346
10  fairly congested        PROBLEM                            Tonsil congested  4116401
11  Amoxil                TREATMENT                        Amoxycillin overdose  4173544
12  Aldex                 TREATMENT                                 Oral lesion  43530620
13  difficulty breathing    PROBLEM                        Difficulty breathing  4041664
14  more congested          PROBLEM                            Nasal congestion  4195085
15  a temperature              TEST  Tolerance of ambient temperature - finding  4271383
16  congestion              PROBLEM                            Nasal congestion  4195085
```

{:.model-param}
## Model Information

{:.table-model}
|----------------|-------------------------------------------|
| Name:           | chunkresolve_athena_conditions_healthcare |
| Type:    | ChunkEntityResolverModel                  |
| Compatibility:  | 2.6.0                                     |
| License:        | Licensed                                  |
|Edition:|Official|                                |
|Input labels:         | [token, chunk_embeddings]                   |
|Output labels:        | [entity]                                    |
| Language:       | en                                        |
| Case sensitive: | True                                      |
| Dependencies:  | embeddings_healthcare_100d                |

{:.h2_title}
## Data Source
Trained on Athena dataset.
