---
layout: model
title: Temporality / Certainty Assertion Status (md)
author: John Snow Labs
name: finassertiondl_time_md
date: 2023-01-04
tags: [en, licensed]
task: Assertion Status
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a medium (md) Assertion Status Model aimed to detect temporality (PRESENT, PAST, FUTURE) or Certainty (POSSIBLE) in your financial documents, which may improve the results of the `finassertion_time` (small) model.

## Predicted Entities

`PRESENT`, `PAST`, `FUTURE`, `POSSIBLE`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FINASSERTION_TEMPORALITY){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finassertiondl_time_md_en_1.0.0_3.0_1672844660896.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

assertion = finance.AssertionDLModel.pretrained("finassertion_time_md", "en", "finance/models")\
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
|Model Name:|finassertiondl_time_md|
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
label	 tp	 fp	 fn	 prec	 rec	 f1
PRESENT	 115	 11	 5	 0.9126984	 0.9583333	 0.9349593
POSSIBLE	 79	 5	 4	 0.9404762	 0.9518072	 0.9461077
PAST	 54	 5	 11	 0.91525424	 0.83076924	 0.8709678
FUTURE	 77	 3	 4	 0.9625	 0.9506173	 0.95652175
Macro-average 325 24 24 0.9327322 0.9228818 0.92778087
Micro-average 325 24 24 0.9312321 0.9312321 0.9312321
```