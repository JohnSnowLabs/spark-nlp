---
layout: model
title: Sentence Entity Resolver for UMLS CUI Codes (Drug & Substance)
author: John Snow Labs
name: sbiobertresolve_umls_drug_substance
date: 2021-12-06
tags: [entity_resolution, en, clinical, licensed]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.3.3
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
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
documentAssembler = DocumentAssembler()\
    .setInputCol('text')\
    .setOutputCol('document')

sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

stopwords = StopWordsCleaner.pretrained()\
    .setInputCols("token")\
    .setOutputCol("cleanTokens")\
    .setCaseSensitive(False)

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "cleanTokens"])\
    .setOutputCol("embeddings")


clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "cleanTokens", "ner"]) \
    .setOutputCol("ner_chunk")

chunk2doc = Chunk2Doc()\
    .setInputCols("ner_chunk")\
    .setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
    .pretrained("sbiobert_base_cased_mli",'en','clinical/models')\
    .setInputCols(["ner_chunk_doc"])\
    .setOutputCol("sbert_embeddings").setCaseSensitive(False)

resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_umls_drug_substance","en", "clinical/models") \
    .setInputCols(["ner_chunk", "sbert_embeddings"]) \
    .setOutputCol("resolution")\
    .setDistanceFunction("EUCLIDEAN")

pipeline = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, resolver])

data = spark.createDataFrame([[""]]).toDF("text")

model = LightPipeline(pipeline.fit(data))

results = model.fullAnnotate(['Dilaudid', 'Hydromorphone', 'Exalgo', 'Palladone', 'Hydrogen peroxide 30 mg', 'Neosporin Cream', 'Magnesium hydroxide 100mg/1ml', 'Metformin 1000 mg'])
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val stopwords = StopWordsCleaner.pretrained()
    .setInputCols("token")
    .setOutputCol("cleanTokens")
    .setCaseSensitive(False)

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "cleanTokens"))
    .setOutputCol("embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "cleanTokens", "ner"))
    .setOutputCol("ner_chunk")

val chunk2doc = new Chunk2Doc()
    .setInputCols("ner_chunk")
    .setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en","clinical/models")
    .setInputCols("ner_chunk_doc")
    .setOutputCol("sbert_embeddings")

val resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_umls_drug_substance", "en", "clinical/models")
    .setInputCols(Array("ner_chunk_doc", "sbert_embeddings"))
    .setOutputCol("resolution")
    .setDistanceFunction("EUCLIDEAN")

val p_model = new PipelineModel().setStages(Array(documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, resolver))

val data = Seq("""'Dilaudid', 'Hydromorphone', 'Exalgo', 'Palladone', 'Hydrogen peroxide 30 mg', 'Neosporin Cream', 'Magnesium hydroxide 100mg/1ml', 'Metformin 1000 mg'""").toDS().toDF("text") 

val res = p_model.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.umls_drug_substance").predict("""Magnesium hydroxide 100mg/1ml""")
```

</div>

## Results

```bash
|    | chunk                         | code     | code_description           | all_k_code_desc                                              | all_k_codes                                                                                                                                                                             |
|---:|:------------------------------|:---------|:---------------------------|:-------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | Dilaudid                      | C0728755 | dilaudid                   | ['C0728755', 'C0719907', 'C1448344', 'C0305924', 'C1569295'] | ['dilaudid', 'Dilaudid HP', 'Disthelm', 'Dilaudid Injection', 'Distaph']                                                                                                                |
|  1 | Hydromorphone                 | C0012306 | HYDROMORPHONE              | ['C0012306', 'C0700533', 'C1646274', 'C1170495', 'C0498841'] | ['HYDROMORPHONE', 'Hydromorphone HCl', 'Phl-HYDROmorphone', 'PMS HYDROmorphone', 'Hydromorphone injection']                                                                             |
|  2 | Exalgo                        | C2746500 | Exalgo                     | ['C2746500', 'C0604734', 'C1707065', 'C0070591', 'C3660437'] | ['Exalgo', 'exaltolide', 'Exelgyn', 'Extacol', 'exserohilone']                                                                                                                          |
|  3 | Palladone                     | C0730726 | palladone                  | ['C0730726', 'C0594402', 'C1655349', 'C0069952', 'C2742475'] | ['palladone', 'Palladone-SR', 'Palladone IR', 'palladiazo', 'palladia']                                                                                                                 |
|  4 | Hydrogen peroxide 30 mg       | C1126248 | hydrogen peroxide 30 MG/ML | ['C1126248', 'C0304655', 'C1605252', 'C0304656', 'C1154260'] | ['hydrogen peroxide 30 MG/ML', 'Hydrogen peroxide solution 30%', 'hydrogen peroxide 30 MG/ML [Proxacol]', 'Hydrogen peroxide 30 mg/mL cutaneous solution', 'benzoyl peroxide 30 MG/ML'] |
|  5 | Neosporin Cream               | C0132149 | Neosporin Cream            | ['C0132149', 'C0306959', 'C4722788', 'C0704071', 'C0698988'] | ['Neosporin Cream', 'Neosporin Ointment', 'Neomycin Sulfate Cream', 'Neosporin Topical Ointment', 'Naseptin cream']                                                                     |
|  6 | Magnesium hydroxide 100mg/1ml | C1134402 | magnesium hydroxide 100 MG | ['C1134402', 'C1126785', 'C4317023', 'C4051486', 'C4047137'] | ['magnesium hydroxide 100 MG', 'magnesium hydroxide 100 MG/ML', 'Magnesium sulphate 100mg/mL injection', 'magnesium sulfate 100 MG', 'magnesium sulfate 100 MG/ML']                     |
|  7 | Metformin 1000 mg             | C0987664 | metformin 1000 MG          | ['C0987664', 'C2719784', 'C0978482', 'C2719786', 'C4282269'] | ['metformin 1000 MG', 'metFORMIN hydrochloride 1000 MG', 'METFORMIN HCL 1000MG TAB', 'metFORMIN hydrochloride 1000 MG [Fortamet]', 'METFORMIN HCL 1000MG SA TAB']                       |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_umls_drug_substance|
|Compatibility:|Healthcare NLP 3.3.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_chunk_embeddings]|
|Output Labels:|[output]|
|Language:|en|
|Case sensitive:|false|

## Data Source

Trained on `2021AB` UMLS dataset’s `Clinical Drug`, `Pharmacologic Substance`, `Antibiotic`, `Hazardous or Poisonous Substance` categories. https://www.nlm.nih.gov/research/umls/index.html
