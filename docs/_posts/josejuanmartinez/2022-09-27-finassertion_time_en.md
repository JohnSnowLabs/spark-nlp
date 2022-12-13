---
layout: model
title: Temporality / Certainty Assertion Status
author: John Snow Labs
name: finassertion_time
date: 2022-09-27
tags: [en, licensed]
task: Assertion Status
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: AssertionDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is an Assertion Status Model aimed to detect temporality (PRESENT, PAST, FUTURE) or Certainty (POSSIBLE) in your financial documents

## Predicted Entities

`PRESENT`, `PAST`, `FUTURE`, `POSSIBLE`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FINASSERTION_TEMPORALITY){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finassertion_time_en_1.0.0_3.0_1664274273525.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finassertion_time_en_1.0.0_3.0_1664274273525.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
# YOUR NER HERE
# ...
embeddings = BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

chunk_converter = ChunkConverter() \
    .setInputCols(["entity"]) \
    .setOutputCol("ner_chunk")

assertion = finance.AssertionDLModel.pretrained("finassertion_time", "en", "finance/models")\
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

texts = ["Atlantic Inc headquarters could possibly be relocated to Delaware by the end of next year",
        "John Crawford will be hired by Atlantic Inc as CTO"]

lp.annotate(texts)
```

</div>

## Results

```bash
chunk,begin,end,entity_type,assertion
Atlantic Inc,0,11,ORG,POSSIBLE
Delaware,57,64,LOC,POSSIBLE

chunk,begin,end,entity_type,assertion
CTO,47,49,ROLE,FUTURE
Atlantic Inc,31,42,ORG,FUTURE
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finassertion_time|
|Compatibility:|Finance NLP 1.0.0+|
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
label            tp    fp   fn   prec         rec          f1
PRESENT          201   11   16   0.9481132    0.92626727   0.937063
POSSIBLE         171   3    6    0.98275864   0.9661017    0.9743589
FUTURE           119   6    4    0.952        0.96747965   0.95967746
PAST             270   16   10   0.9440559    0.96428573   0.9540636
Macro-average    761   36   36   0.9567319    0.9560336    0.95638263
Micro-average    761   36   36   0.9548306    0.9548306    0.9548306
```

