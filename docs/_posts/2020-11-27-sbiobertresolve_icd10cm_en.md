---
layout: model
title: Sentence Entity Resolver for ICD10-CM (using sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_icd10cm
language: en
repository: clinical/models
date: 2020-11-27
tags: [clinical,entity_resolution,en]
article_header:
    type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
This model maps extracted medical entities to ICD10-CM codes using chunk embeddings.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icd10cm_en_2.6.4_2.4_1606235759310.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")
 
sbert_embedder = BertSentenceEmbeddings\
     .pretrained("sbiobert_base_cased_mli",'en','clinical/models')\
     .setInputCols(["ner_chunk_doc"])\
     .setOutputCol("sbert_embeddings")
 
snomed_ct_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm","en", "clinical/models") \
     .setInputCols(["ner_chunk", "sbert_embeddings"]) \
     .setOutputCol("resolution")\
     .setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, snomed_ct_resolver])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

data = spark.createDataFrame([["A 50-year-old female with chest pain, respiratory insufficiency, and chronic lung disease with bronchospastic angina."]]).toDF("text")

results = model.transform(data)

```

{:.h2_title}
## Results

```bash
|    | ner_chunk         |  resolution |  resolution_description  |
|---:|:------------------|------------:|-------------------------:|
|  0 | respiratory insufficiency    |     J96.9   |   Respiratory Failure, Unspecified   |
|  1 | chronic lung disease        |     J98.4 |   Other Disorders Of Lung   |
|  2 | bronchospastic angina      |    I20.8 |  Other Forms Of Angina Pectoris  |


{:.model-param}
## Model Information

{:.table-model}
|---------------|---------------------|
| Name:         | sbiobertresolve_icd10cm        |
| Type:          | SentenceEntityResolverModel     |
| Compatibility: | Spark NLP 2.6.4 +               |
| License:       | Licensed            |
| Edition:       | Official          |
|Input labels:        | [ner_chunk, chunk_embeddings]     |
|Output labels:       | [resolution]                 |
| Language:      | en                  |
| Dependencies: | sbiobert_base_cased_mli |
