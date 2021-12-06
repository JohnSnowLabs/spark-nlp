---
layout: model
title: Sentence Entity Resolver for UMLS CUI Codes (Drug & Substance)
author: John Snow Labs
name: sbiobertresolve_umls_drug_substance
date: 2021-12-06
tags: [entity_resolution, en, clinical, licensed]
task: Entity Resolution
language: en
edition: Spark NLP for Healthcare 3.3.3
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities to UMLS CUI codes. It is trained on `2021AB` UMLS dataset. The complete dataset has 127 different categories, and this model is trained on the `Clinical Drug`, `Pharmacologic Substance`, `Antibiotic`, `Hazardous or Poisonous Substance` categories using `sbiobert_base_cased_mli` embeddings.

## Predicted Entities

`Predicts UMLS codes for Drugs & Substances medical concepts`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_umls_drug_substance_en_3.3.3_3.0_1638802613409.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
     .pretrained("sbiobert_base_cased_mli",'en','clinical/models')\
     .setInputCols(["ner_chunk_doc"])\
     .setOutputCol("sbert_embeddings")

resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_umls_drug_substance","en", "clinical/models") \
     .setInputCols(["ner_chunk", "sbert_embeddings"]) \
     .setOutputCol("resolution")\
     .setDistanceFunction("EUCLIDEAN")

pipeline = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, resolver])

data = spark.createDataFrame([["""She was immediately given hydrogen peroxide 30 mg to treat the infection on her leg, and has been advised Neosporin Cream for 5 days. She has a history of taking magnesium hydroxide 100mg/1ml, metformin 1000 mg, Diluaid, Isobutyltheophylline, and Hydrocorticosterone."""]]).toDF("text")

model = pipeline.fit(data)

results = model.transform(data)
```
```scala
...
val chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")
val sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
      .setInputCols(["ner_chunk_doc"])\
      .setOutputCol("sbert_embeddings")
    
val resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_umls_drug_substance", "en", "clinical/models") \
      .setInputCols(["ner_chunk_doc", "sbert_embeddings"]) \
      .setOutputCol("resolution")\
      .setDistanceFunction("EUCLIDEAN")

val p_model = new PipelineModel().setStages(Array(documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, resolver))
    
val data = Seq("She was immediately given hydrogen peroxide 30 mg to treat the infection on her leg, and has been advised Neosporin Cream for 5 days. She has a history of taking magnesium hydroxide 100mg/1ml, metformin 1000 mg, Diluaid, Isobutyltheophylline, and Hydrocorticosterone.").toDF("text") 

val res = p_model.transform(data)
```
</div>

## Results

```bash
|  0 | hydrogen peroxide 30 mg       | C1126248 | hydrogen peroxide 30 MG/ML | ['C1126248', 'C0304655', 'C1605252', 'C0304656', 'C1154260'] | ['hydrogen peroxide 30 MG/ML', 'Hydrogen peroxide solution 30%', 'hydrogen peroxide 30 MG/ML [Proxacol]', 'Hydrogen peroxide 30 mg/mL cutaneous solution', 'benzoyl peroxide 30 MG/ML'] |
|  1 | Neosporin Cream               | C0132149 | Neosporin Cream            | ['C0132149', 'C0358174', 'C0357999', 'C0307085', 'C0698810'] | ['Neosporin Cream', 'Nystan cream', 'Nystadermal cream', 'Nupercainal Cream', 'Nystaform cream']                                                                                        |
|  2 | magnesium hydroxide 100mg/1ml | C1134402 | magnesium hydroxide 100 MG | ['C1134402', 'C1126785', 'C4317023', 'C4051486', 'C4047137'] | ['magnesium hydroxide 100 MG', 'magnesium hydroxide 100 MG/ML', 'Magnesium sulphate 100mg/mL injection', 'magnesium sulfate 100 MG', 'magnesium sulfate 100 MG/ML']                     |
|  3 | metformin 1000 mg             | C0987664 | metformin 1000 MG          | ['C0987664', 'C2719784', 'C0978482', 'C2719786', 'C4282269'] | ['metformin 1000 MG', 'metFORMIN hydrochloride 1000 MG', 'METFORMIN HCL 1000MG TAB', 'metFORMIN hydrochloride 1000 MG [Fortamet]', 'METFORMIN HCL 1000MG SA TAB']                       |
|  4 | diluaid                       | C0059854 | diludine                   | ['C0059854', 'C0174425', 'C0211286', 'C2937705', 'C1186842'] | ['diludine', 'dilapan', 'ditec', 'CanineAid', 'Diareze']                                                                                                                                |
|  5 | isobutyltheophylline          | C0000096 | Isobutyltheophylline       | ['C0000096', 'C3848545', 'C0059871', 'C0013360', 'C0039771'] | ['Isobutyltheophylline', 'bromotheophylline', 'oxyethyltheophylline', 'Dihydroxypropyltheophylline', 'THEOPHYLLINE']                                                                    |
|  6 | hydrocorticosterone           | C0063077 | Hydrocortisone acetate     | ['C0063077', 'C0000172', 'C0717842', 'C0360554', 'C0063076'] | ['Hydrocortisone acetate', '18 Hydrocorticosterone', 'hydrocortisone otic', 'Oral hydrocortisone', 'hydrocortisone probutate']                                                          |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_umls_drug_substance|
|Compatibility:|Spark NLP for Healthcare 3.3.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_chunk_embeddings]|
|Output Labels:|[output]|
|Language:|en|
|Case sensitive:|false|

## Data Source

Trained on `2021AB` UMLS dataset’s `Clinical Drug`, `Pharmacologic Substance`, `Antibiotic`, `Hazardous or Poisonous Substance` categories. https://www.nlm.nih.gov/research/umls/index.html