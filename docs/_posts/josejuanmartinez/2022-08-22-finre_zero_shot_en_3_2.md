---
layout: model
title: Financial Zero-shot Relation Extraction
author: John Snow Labs
name: finre_zero_shot
date: 2022-08-22
tags: [en, finance, re, zero, shot, zero_shot, licensed]
task: Relation Extraction
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Zero-shot Relation Extraction Model, meaning that it does not require any training data, just few examples of of the relations types you are looking for, to output a proper result.

Make sure you keep the proper syntax of the relations you want to extract. For example:

```
re_model.setRelationalCategories({
    "DECREASE": ["{PROFIT_DECLINE} decrease {AMOUNT}", "{PROFIT_DECLINE}} decrease {PERCENTAGE}",
    "INCREASE": ["{PROFIT_INCREASE} increase {AMOUNT}", "{PROFIT_INCREASE}} increase {PERCENTAGE}"]
})
```


- The keys of the dictionary are the name of the relations (`DECREASE`, `INCREASE`)
- The values are list of sentences with similar examples of the relation
- The values in brackets are the NER labels extracted by an NER component before

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finre_zero_shot_en_1.0.0_3.2_1661179057628.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finre_zero_shot_en_1.0.0_3.2_1661179057628.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base", "en") \
  .setInputCols("sentence", "token") \
  .setOutputCol("embeddings")\
  .setMaxSentenceLength(512)

ner_model = finance.NerModel.pretrained("finner_10k", "en", "finance/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")\

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

re_model = finance.ZeroShotRelationExtractionModel.pretrained("finre_zero_shot", "en", "finance/models")\
    .setInputCols(["ner_chunk", "sentence"]) \
    .setOutputCol("relations")

# Remember it's 2 curly brackets instead of one if you are using Spark NLP < 4.0
re_model.setRelationalCategories({
    "DECREASE": ["{PROFIT_DECLINE} decrease {AMOUNT}", "{PROFIT_DECLINE} decrease {PERCENTAGE}"],
    "INCREASE": ["{PROFIT_INCREASE} increase {AMOUNT}", "{PROFIT_INCREASE} increase {PERCENTAGE}"]
})

pipeline = sparknlp.base.Pipeline() \
    .setStages([document_assembler,  
                sentence_detector,
                tokenizer, 
                embeddings,
                ner_model,
                ner_converter,
                re_model
               ])
               
sample_text = """License fees revenue decreased 40 %, or $ 0.5 million to $ 0.7 million for the year ended December 31, 2020 
compared to $ 1.2 million for the year ended December 31, 2019. Services revenue increased 4 %, or $ 1.1 million, to $ 25.6 million 
for the year ended December 31, 2020 from $ 24.5 million for the year ended December 31, 2019.
Costs of revenue, excluding depreciation and amortization increased by $ 0.1 million, or 2 %, to $ 8.8 million for the year ended December 31, 2020 
from $ 8.7 million for the year ended December 31, 2019. The increase was primarily related to increase in internal staff costs of $ 1.1 million as 
we increased delivery staff and work performed on internal projects, partially offset by a decrease in third party consultant costs of $ 0.6 million 
as these were converted to internal staff or terminated. Also, a decrease in travel costs of $ 0.4 million due to travel restrictions caused by the global pandemic. 
As a percentage of revenue, cost of revenue, excluding depreciation and amortization was 34 % for each of the years ended December 31, 2020 and 2019. 
Sales and marketing expenses decreased 20 %, or $ 1.5 million, to $ 6.0 million for the year ended December 31, 2020 from $ 7.5 million for the year ended December 31, 2019"
"""

data = spark.createDataFrame([[sample_text]]).toDF("text")
model = pipeline.fit(data)
results = model.transform(data)

# ner output
results.selectExpr("explode(ner_chunk) as ner").show(truncate=False)

# relations output
results.selectExpr("explode(relations) as relation").show(truncate=False)

```

</div>

## Results

```bash
+--------------------------------------------------------------------------------------------------------------------------+
|ner                                                                                                                       |
+--------------------------------------------------------------------------------------------------------------------------+
|[chunk, 0, 19, License fees revenue, [entity -> PROFIT_DECLINE, sentence -> 0, chunk -> 0, confidence -> 0.41060004], []] |
|[chunk, 31, 32, 40, [entity -> PERCENTAGE, sentence -> 0, chunk -> 1, confidence -> 0.9995], []]                          |
|[chunk, 40, 40, $, [entity -> CURRENCY, sentence -> 0, chunk -> 2, confidence -> 0.9995], []]                             |
|[chunk, 42, 52, 0.5 million, [entity -> AMOUNT, sentence -> 0, chunk -> 3, confidence -> 0.99995], []]                    |
|[chunk, 57, 57, $, [entity -> CURRENCY, sentence -> 0, chunk -> 4, confidence -> 0.9998], []]                             |
|[chunk, 59, 69, 0.7 million, [entity -> AMOUNT, sentence -> 0, chunk -> 5, confidence -> 0.99985003], []]                 |
|[chunk, 90, 106, December 31, 2020, [entity -> FISCAL_YEAR, sentence -> 0, chunk -> 6, confidence -> 0.977525], []]       |
|[chunk, 121, 121, $, [entity -> CURRENCY, sentence -> 0, chunk -> 7, confidence -> 0.9996], []]                           |
|[chunk, 123, 133, 1.2 million, [entity -> AMOUNT, sentence -> 0, chunk -> 8, confidence -> 0.99975], []]                  |
|[chunk, 154, 170, December 31, 2019, [entity -> FISCAL_YEAR, sentence -> 0, chunk -> 9, confidence -> 0.96227497], []]    |
|[chunk, 173, 188, Services revenue, [entity -> PROFIT_INCREASE, sentence -> 1, chunk -> 10, confidence -> 0.57490003], []]|
|[chunk, 200, 200, 4, [entity -> PERCENTAGE, sentence -> 1, chunk -> 11, confidence -> 0.9997], []]                        |
|[chunk, 208, 208, $, [entity -> CURRENCY, sentence -> 1, chunk -> 12, confidence -> 0.999], []]                           |
|[chunk, 210, 220, 1.1 million, [entity -> AMOUNT, sentence -> 1, chunk -> 13, confidence -> 0.99995], []]                 |
|[chunk, 226, 226, $, [entity -> CURRENCY, sentence -> 1, chunk -> 14, confidence -> 0.9982], []]                          |
|[chunk, 228, 239, 25.6 million, [entity -> AMOUNT, sentence -> 1, chunk -> 15, confidence -> 0.99975], []]                |
|[chunk, 261, 277, December 31, 2020, [entity -> FISCAL_YEAR, sentence -> 1, chunk -> 16, confidence -> 0.97915], []]      |
|[chunk, 284, 284, $, [entity -> CURRENCY, sentence -> 1, chunk -> 17, confidence -> 0.9991], []]                          |
|[chunk, 286, 297, 24.5 million, [entity -> AMOUNT, sentence -> 1, chunk -> 18, confidence -> 0.99965], []]                |
|[chunk, 318, 334, December 31, 2019, [entity -> FISCAL_YEAR, sentence -> 1, chunk -> 19, confidence -> 0.9588], []]       |
+--------------------------------------------------------------------------------------------------------------------------+

+--------+
|relation                                                                                                                                                 +--------+
|[category, 0, 217, DECREASE, [entity1_begin -> 0, relation -> DECREASE, hypothesis -> License fees revenue decrease 40, confidence -> 0.9931541, nli_prediction -> entail, entity1 -> PROFIT_DECLINE, syntactic_distance -> undefined, chunk2 -> 40, entity2_end -> 32, entity1_end -> 19, entity2_begin -> 31, entity2 -> PERCENTAGE, chunk1 -> License fees revenue, sentence -> 0], []]                  |
|[category, 672, 898, DECREASE, [entity1_begin -> 0, relation -> DECREASE, hypothesis -> License fees revenue decrease 1.2 million, confidence -> 0.7394818, nli_prediction -> entail, entity1 -> PROFIT_DECLINE, syntactic_distance -> undefined, chunk2 -> 1.2 million, entity2_end -> 133, entity1_end -> 19, entity2_begin -> 123, entity2 -> AMOUNT, chunk1 -> License fees revenue, sentence -> 0], []]|
|[category, 445, 671, DECREASE, [entity1_begin -> 0, relation -> DECREASE, hypothesis -> License fees revenue decrease 0.7 million, confidence -> 0.99002415, nli_prediction -> entail, entity1 -> PROFIT_DECLINE, syntactic_distance -> undefined, chunk2 -> 0.7 million, entity2_end -> 69, entity1_end -> 19, entity2_begin -> 59, entity2 -> AMOUNT, chunk1 -> License fees revenue, sentence -> 0], []] |
|[category, 218, 444, DECREASE, [entity1_begin -> 0, relation -> DECREASE, hypothesis -> License fees revenue decrease 0.5 million, confidence -> 0.99084955, nli_prediction -> entail, entity1 -> PROFIT_DECLINE, syntactic_distance -> undefined, chunk2 -> 0.5 million, entity2_end -> 52, entity1_end -> 19, entity2_begin -> 42, entity2 -> AMOUNT, chunk1 -> License fees revenue, sentence -> 0], []] |
+--------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finre_zero_shot|
|Type:|finance|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|406.4 MB|
|Case sensitive:|true|

## References

Bert Base (cased) trained on the GLUE MNLI dataset.
