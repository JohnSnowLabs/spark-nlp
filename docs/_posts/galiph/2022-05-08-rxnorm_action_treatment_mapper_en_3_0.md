---
layout: model
title: rxnorm_action_treatment_mapper
author: John Snow Labs
name: rxnorm_action_treatment_mapper
date: 2022-05-08
tags: [en, chunk_mapper, rxnorm, action, treatment, licensed]
task: Chunk Mapping
language: en
edition: Spark NLP for Healthcare 3.5.1
spark_version: 3.0
supported: true
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
<button class="button button-orange" disabled>Open in Colab</button>
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

chunkerMapper_action = ChunkMapperModel.pretrained("rxnorm_action_treatment_mapper", "en", "clinical/models"))\
      .setInputCols(["rxnorm_code"])\
      .setOutputCol("Action")\
      .setRel("Action") 

chunkerMapper_treatment = ChunkMapperModel.pretrained("rxnorm_action_treatment_mapper", "en", "clinical/models"))\
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

lp = LightPipeline(model)

res = lp.annotate(['Sinequan 150 MG', 'Zonalon 50 mg'])

```
```scala
val document_assembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en","clinical/models")\
      .setInputCols(Array("ner_chunk"))\
      .setOutputCol("sentence_embeddings")\
      .setCaseSensitive(False)
    
val rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_augmented","en", "clinical/models") \
      .setInputCols(Array("ner_chunk", "sentence_embeddings")) \
      .setOutputCol("rxnorm_code")\
      .setDistanceFunction("EUCLIDEAN")

val chunkerMapper_action = ChunkMapperModel.pretrained("rxnorm_action_treatment_mapper", "en", "clinical/models"))\
      .setInputCols("rxnorm_code")\
      .setOutputCol("Action")\
      .setRel("Action") 

val chunkerMapper_treatment = ChunkMapperModel.pretrained("rxnorm_action_treatment_mapper", "en", "clinical/models"))\
      .setInputCols("rxnorm_code")\
      .setOutputCol("Treatment")\
      .setRel("Treatment") 

val pipeline = Pipeline().setStages(Array(document_assembler,
                                 sbert_embedder,
                                 rxnorm_resolver,
                                 chunkerMapper_action,
                                 chunkerMapper_treatment
                                 ))

 val text_data = Seq("Sinequan 150 MG", "Zonalon 50 mg").toDF("text")


 val res = pipeline.fit(text_data).transform(text_data)
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
|Compatibility:|Spark NLP for Healthcare 3.5.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|19.3 MB|
