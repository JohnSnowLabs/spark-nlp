---
layout: model
title: Sentence Entity Resolver for RxNorm (sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_rxnorm_augmented
date: 2021-11-09
tags: [rxnorm, licensed, en, clinical, entity_resolution]
task: Entity Resolution
language: en
edition: Spark NLP for Healthcare 3.3.1
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities and concepts (like drugs/ingredients) to RxNorm codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings. It trained on the augmented version of the dataset which is used in previous RxNorm resolver models. Additionally, this model returns concept classes of the drugs in `all_k_aux_labels` column.

## Predicted Entities

`RxNorm Codes`, `Concept Classes`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_rxnorm_augmented_en_3.3.1_2.4_1636472410154.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

rxnorm_pipeline = Pipeline(
    stages = [
        documentAssembler,
        sbert_embedder,
        rxnorm_resolver])

data = spark.createDataFrame([["She is given Fragmin 5000 units subcutaneously daily, Xenaderm to wounds topically b.i.d., OxyContin 30 mg p.o. q.12 h., folic acid 1 mg daily, levothyroxine 0.1 mg p.o. daily, Prevacid 30 mg daily, Avandia 4 mg daily, aspirin 81 mg daily, Neurontin 400 mg p.o. t.i.d., Percocet 5/325 mg 2 tablets q.4 h. p.r.n., magnesium citrate 1 bottle p.o. p.r.n., sliding scale coverage insulin, Wellbutrin 100 mg p.o. daily."]]).toDF("text")

results = rxnorm_pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings
      .pretrained("sbiobert_base_cased_mli", "en","clinical/models")
      .setInputCols(Array("ner_chunk"))
      .setOutputCol("sbert_embeddings")
    
val rxnorm_resolver = SentenceEntityResolverModel
      .pretrained("sbiobertresolve_rxnorm_augmented", "en", "clinical/models")
      .setInputCols(Array("ner_chunk", "sbert_embeddings"))
      .setOutputCol("rxnorm_code")
      .setDistanceFunction("EUCLIDEAN")

val rxnorm_pipeline = new Pipeline().setStages(Array(documentAssembler, sbert_embedder, rxnorm_resolver))

val data = Seq("She is given Fragmin 5000 units subcutaneously daily, Xenaderm to wounds topically b.i.d., OxyContin 30 mg p.o. q.12 h., folic acid 1 mg daily,  levothyroxine 0.1 mg p.o. daily, Prevacid 30 mg daily, Avandia 4 mg daily, aspirin 81 mg daily, Neurontin 400 mg p.o. t.i.d., Percocet 5/325 mg 2 tablets q.4 h. p.r.n., magnesium citrate 1 bottle p.o. p.r.n., sliding scale coverage insulin, Wellbutrin 100 mg p.o. daily.").toDF("text")

val result = rxnorm_pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-----------------+-----+---+---------------+----------+----------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|            chunk|begin|end|         entity|confidence|RxNormCode|                                         all_codes|                                       resolutions|                                     Concept Class|
+-----------------+-----+---+---------------+----------+----------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
|       folic acid|  121|130|Drug_Ingredient|   0.59705|      4511|4511:::1162058:::1162059:::62356:::1376005:::54...|folic acid:::folic acid oral product:::folic ac...|Ingredient:::Clinical Dose Group:::Clinical Dos...|
|    levothyroxine|  144|156|Drug_Ingredient|    0.6059|     10582|10582:::1868004:::40144:::1602753:::1602745:::2...|levothyroxine:::levothyroxine injection:::levot...|Ingredient:::Clinical Drug Form:::Precise Ingre...|
|          aspirin|  219|225|Drug_Ingredient|    0.9814|      1191|1191:::405403:::218266:::1154070:::215568:::202...|aspirin:::ysp aspirin:::med aspirin:::aspirin p...|Ingredient:::Brand Name:::Brand Name:::Clinical...|
|magnesium citrate|  313|329|Drug_Ingredient|   0.53295|     52356|52356:::29155:::1314220:::1006900:::52358:::291...|magnesium citrate:::magnesium carbonate:::magne...|Ingredient:::Ingredient:::Precise Ingredient:::...|
|          insulin|  376|382|Drug_Ingredient|    0.4832|    139825|139825:::1740938:::274783:::86009:::1605101:::5...|insulin detemir:::insulin argine:::insulin glar...|Ingredient:::Ingredient:::Ingredient:::Ingredie...|
+-----------------+-----+---+---------------+----------+----------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_rxnorm_augmented|
|Compatibility:|Spark NLP for Healthcare 3.3.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[rxnorm_code]|
|Language:|en|
|Case sensitive:|false|