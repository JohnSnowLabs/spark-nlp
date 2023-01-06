---
layout: model
title: Temporality / Certainty Assertion Status (sm)
author: John Snow Labs
name: legassertion_time
date: 2022-09-27
tags: [en, licensed]
task: Assertion Status
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: AssertionDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a small Assertion Status Model aimed to detect temporality (PRESENT, PAST, FUTURE) or Certainty (POSSIBLE) in your legal documents

## Predicted Entities

`PRESENT`, `PAST`, `FUTURE`, `POSSIBLE`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/legal/LEGASSERTION_TEMPORALITY){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legassertion_time_en_1.0.0_3.0_1664274039847.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
# YOUR NER HERE
# ...
embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

chunk_converter = nlp.ChunkConverter() \
    .setInputCols(["entity"]) \
    .setOutputCol("ner_chunk")

assertion = legal.AssertionDLModel.pretrained("legassertion_time", "en", "legal/models")\
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")
    
nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    tokenizer,
    embeddings,
    ner,
    chunk_converter,
    assertion
    ])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

lp = LightPipeline(model)

texts = ["The subsidiaries of Atlantic Inc will participate in a merging operation",
    "The Conditions and Warranties of this agreement might be modified"]

lp.annotate(texts)
```

</div>

## Results

```bash
chunk,begin,end,entity_type,assertion
Atlantic Inc,20,31,ORG,FUTURE

chunk,begin,end,entity_type,assertion
Conditions and Warranties,4,28,DOC,POSSIBLE
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legassertion_time|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, doc_chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|en|
|Size:|2.2 MB|

## References

In-house annotations on financial and legal corpora

## Benchmarking

```bash
label            tp      fp    fn    prec         rec         f1
PRESENT          201     11    16    0.9481132    0.9262672   0.937063
POSSIBLE         171     3     6     0.9827586    0.9661017   0.974359
FUTURE           119     6     4     0.952        0.9674796   0.959677
PAST             270     16    10    0.9440559    0.9642857   0.954063
Macro-average    761     36    36    0.9567319    0.9560336   0.9563826
Micro-average    761     36    36    0.9548306    0.9548306   0.9548306
```
