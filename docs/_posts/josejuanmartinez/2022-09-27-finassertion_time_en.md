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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finassertion_time_en_1.0.0_3.0_1664274273525.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["document"])\
    .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")

ner = finance.BertForTokenClassification.pretrained("finner_bert_roles","en","finance/models")\
  .setInputCols("token", "document")\
  .setOutputCol("ner")\
  .setCaseSensitive(True)  

chunk_converter = nlp.NerConverter() \
    .setInputCols(["document", "token", "ner"]) \
    .setOutputCol("ner_chunk")

assertion = finance.AssertionDLModel.pretrained("finassertion_time", "en", "finance/models")\
    .setInputCols(["document", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")
    
nlpPipeline = nlp.Pipeline(stages=[
    document_assembler, 
    tokenizer,
    embeddings,
    ner,
    chunk_converter,
    assertion
    ])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

lp = nlp.LightPipeline(model)

texts = ["John Crawford will be hired by Atlantic Inc as CTO"]

lp.annotate(texts)
```

</div>

## Results

```bash
chunk,begin,end,entity_type,assertion
CTO,47,49,ROLE,FUTURE
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

