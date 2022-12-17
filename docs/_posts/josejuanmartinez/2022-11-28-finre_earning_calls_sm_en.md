---
layout: model
title: Financial Relation Extraction on Earning Calls (Small)
author: John Snow Labs
name: finre_earning_calls_sm
date: 2022-11-28
tags: [earning, calls, en, licensed]
task: Relation Extraction
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: RelationExtractionDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model extracts relations between amounts, counts, percentages, dates and the financial entities extracted with any earning calls NER model, as `finner_earning_calls_sm` (shown in the example above). 

## Predicted Entities

`has_amount`, `has_amount_date`, `has_percentage_date`, `has_percentage`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finre_earning_calls_sm_en_1.0.0_3.0_1669649131686.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

sentencizer = nlp.SentenceDetectorDLModel\
        .pretrained("sentence_detector_dl", "en") \
        .setInputCols(["document"])\
        .setOutputCol("sentence")
                      
tokenizer = nlp.Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")\
        .setContextChars(['.', ',', ';', ':', '!', '?', '*', '-', '(', ')', '”', '’', '$','€'])

bert_embeddings= nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en")\
        .setInputCols(["sentence", "token"])\
        .setOutputCol("bert_embeddings")

ner_model = finance.NerModel.pretrained("finner_earning_calls_sm", "en", "finance/models")\
    .setInputCols(["sentence", "token", "bert_embeddings"])\
    .setOutputCol("ner")\

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

# ===========
# This is needed only to filter relation pairs using finance.RENerChunksFilter (see below)
# ===========
pos = nlp.PerceptronModel.pretrained("pos_anc", 'en')\
          .setInputCols("sentence", "token")\
          .setOutputCol("pos")

dependency_parser = nlp.DependencyParserModel.pretrained("dependency_conllu", "en") \
    .setInputCols(["sentence", "pos", "token"]) \
    .setOutputCol("dependencies")


ENTITIES = ['PROFIT', 'PROFIT_INCREASE', 'PROFIT_DECLINE', 'CF', 'CF_INCREASE', 'CF_DECREASE', 'LIABILITY', 'EXPENSE', 'EXPENSE_INCREASE', 'EXPENSE_DECREASE']

ENTITY_PAIRS = [f"{x}-AMOUNT" for x in ENTITIES]
ENTITY_PAIRS.extend([f"{x}-COUNT" for x in ENTITIES])
ENTITY_PAIRS.extend([f"{x}-PERCENTAGE" for x in ENTITIES])
ENTITY_PAIRS.append(f"AMOUNT-FISCAL_YEAR")
ENTITY_PAIRS.append(f"AMOUNT-DATE")
ENTITY_PAIRS.append(f"AMOUNT-CURRENCY")

re_ner_chunk_filter = finance.RENerChunksFilter() \
    .setInputCols(["ner_chunk", "dependencies"])\
    .setOutputCol("re_ner_chunk")\
    .setRelationPairs(ENTITY_PAIRS)\
    .setMaxSyntacticDistance(5)

# ===========

reDL = finance.RelationExtractionDLModel.pretrained('finre_financial_small', 'en', 'finance/models')\
    .setInputCols(["re_ner_chunk", "sentence"])\
    .setOutputCol("relations")

pipeline = Pipeline(stages=[
        documentAssembler,
        sentencizer,
        tokenizer,
        bert_embeddings,
        ner_model,
        ner_converter,
        pos,
        dependency_parser,
        re_ner_chunk_filter,
        reDL])

text = "In the third quarter of fiscal 2021, we received net proceeds of $342.7 million, after deducting underwriters discounts and commissions and offering costs of $31.8 million, including  the exercise of the underwriters option to purchase additional shares. "

data = spark.createDataFrame([[text]]).toDF("text")

model = pipeline.fit(data)

results = model.transform(data)
```

</div>

## Results

```bash
   relation   entity1 entity1_begin entity1_end                          chunk1 entity2 entity2_begin entity2_end         chunk2 confidence
 has_amount        CF            49          60                    net proceeds  AMOUNT            66          78  342.7 million  0.9999101
 has_amount  CURRENCY            65          65                               $  AMOUNT            66          78  342.7 million  0.9925425
 has_amount   EXPENSE           125         154  commissions and offering costs  AMOUNT           160         171   31.8 million  0.9997677
 has_amount  CURRENCY           159         159                               $  AMOUNT           160         171   31.8 million   0.998896
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finre_earning_calls_sm|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|402.6 MB|

## References

In-house annotations of scarpped Earning Call Transcripts.

## Benchmarking

```bash
Relation           Recall Precision        F1   Support
has_amount          0.973     0.973     0.973       183
has_amount_date     0.700     1.000     0.824        10
has_percentage      0.987     0.931     0.958       150
has_percentage_date     0.667     0.857     0.750         9
other               0.993     0.995     0.994      2048
Avg.                0.864     0.951     0.900 2048
Weighted-Avg.       0.988     0.988     0.988 2048
```
