---
layout: model
title: Mapping RxNorm Codes with Corresponding Actions and Treatments
author: John Snow Labs
name: rxnorm_action_treatment_mapper
date: 2022-05-08
tags: [en, chunk_mapper, rxnorm, action, treatment, licensed, clinical]
task: Chunk Mapping
language: en
edition: Healthcare NLP 3.5.1
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/rxnorm_action_treatment_mapper_en_3.5.1_3.0_1652043181565.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use


<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
.setInputCol('text')\
.setOutputCol('ner_chunk')


sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
.setInputCols(["ner_chunk"])\
.setOutputCol("sentence_embeddings")\
.setCaseSensitive(False)

rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_augmented","en", "clinical/models") \
.setInputCols(["ner_chunk", "sentence_embeddings"]) \
.setOutputCol("rxnorm_code")\
.setDistanceFunction("EUCLIDEAN")


chunkerMapper_action = ChunkMapperModel.pretrained("rxnorm_action_treatment_mapper", "en", "clinical/models")\
.setInputCols(["rxnorm_code"])\
.setOutputCol("Action")\
.setRel("Action") 


chunkerMapper_treatment = ChunkMapperModel.pretrained("rxnorm_action_treatment_mapper", "en", "clinical/models")\
.setInputCols(["rxnorm_code"])\
.setOutputCol("Treatment")\
.setRel("Treatment") 


pipeline = Pipeline().setStages([document_assembler,
sbert_embedder,
rxnorm_resolver,
chunkerMapper_action,
chunkerMapper_treatment
])


model = pipeline.fit(spark.createDataFrame([['']]).toDF('text')) 

light_pipeline = LightPipeline(model)

result = light_pipeline.annotate(['Sinequan 150 MG', 'Zonalon 50 mg'])
```
```scala
val document_assembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("ner_chunk")


val sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en","clinical/models")
.setInputCols(Array("ner_chunk"))
.setOutputCol("sentence_embeddings")
.setCaseSensitive(False)

val rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_augmented","en", "clinical/models")
.setInputCols(Array("ner_chunk", "sentence_embeddings"))
.setOutputCol("rxnorm_code")
.setDistanceFunction("EUCLIDEAN")


val chunkerMapper_action = ChunkMapperModel.pretrained("rxnorm_action_treatment_mapper", "en", "clinical/models"))
.setInputCols("rxnorm_code")
.setOutputCol("Action")
.setRel("Action") 


val chunkerMapper_treatment = ChunkMapperModel.pretrained("rxnorm_action_treatment_mapper", "en", "clinical/models"))
.setInputCols("rxnorm_code")
.setOutputCol("Treatment")
.setRel("Treatment") 


val pipeline = new Pipeline().setStages(Array(document_assembler,
sbert_embedder,
rxnorm_resolver,
chunkerMapper_action,
chunkerMapper_treatment
))


val text_data = Seq("Sinequan 150 MG", "Zonalon 50 mg").toDS.toDF("text")
val res = pipeline.fit(text_data).transform(text_data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.map_entity.rxnorm_to_action_treatment").predict("""Sinequan 150 MG""")
```

</div>


## Results


```bash
|    | ner_chunk           | rxnorm_code   | Treatment                                                                      | Action                                                                 |
|---:|:--------------------|:--------------|:-------------------------------------------------------------------------------|:-----------------------------------------------------------------------|
|  0 | ['Sinequan 150 MG'] | ['1000067']   | ['Alcoholism', 'Depression', 'Neurosis', 'Anxiety&Panic Attacks', 'Psychosis'] | ['Antidepressant', 'Anxiolytic', 'Psychoanaleptics', 'Sedative']       |
|  1 | ['Zonalon 50 mg']   | ['103971']    | ['Pain']                                                                       | ['Analgesic', 'Analgesic (Opioid)', 'Analgetic', 'Opioid', 'Vitamins'] |


```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|rxnorm_action_treatment_mapper|
|Compatibility:|Healthcare NLP 3.5.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|19.3 MB|
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjA4MDQ2NDcyNywtMTE5NjM3NzMyOV19
-->