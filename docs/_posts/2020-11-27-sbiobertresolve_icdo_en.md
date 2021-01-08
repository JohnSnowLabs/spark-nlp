---
layout: model
title: Sentence Entity Resolver for ICD-O (using sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_icdo
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
This model maps extracted medical entities to ICD-O codes using chunk embeddings.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icdo_en_2.6.4_2.4_1606235766320.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
 
snomed_ct_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icdo","en", "clinical/models") \
     .setInputCols(["ner_chunk", "sbert_embeddings"]) \
     .setOutputCol("resolution")\
     .setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, snomed_ct_resolver])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

results = model.transform(data)

```

{:.h2_title}
## Results


{:.model-param}
## Model Information

{:.table-model}
|---------------|---------------------|
| Name:         | sbiobertresolve_icdo        |
| Type:          | SentenceEntityResolverModel     |
| Compatibility: | Spark NLP 2.6.4 +               |
| License:       | Licensed            |
| Edition:       | Official          |
|Input labels:        | [ner_chunk, chunk_embeddings]     |
|Output labels:       | [resolution]                 |
| Language:      | en                  |
| Dependencies: | sbiobert_base_cased_mli |
