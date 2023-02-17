---
layout: model
title: Understand Acquisitions in Context (Partial, Total, Other)
author: John Snow Labs
name: finassertiondl_acquisitions
date: 2023-02-17
tags: [en, finance, assertion, acquisition, licensed]
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

Description: This is an Assertion Status model, able to identify, giving ORG NER entities if there is a mention of an acquisition. If so, the assertion model will analyze the context and return either `TOTAL_ACQUISITION` or `PARTIAL_ACQUISITION`. If the ORGS are not mentioned in any acquisition context, the model will return `OTHER`.

## Predicted Entities

`PARTIAL_ACQUISITION`, `TOTAL_ACQUISITION`, `OTHER`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finassertiondl_acquisitions_en_1.0.0_3.0_1676667267978.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finassertiondl_acquisitions_en_1.0.0_3.0_1676667267978.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenizer = nlp.Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token") \

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_model = finance.NerModel.pretrained("finner_org_per_role_date", "en", "finance/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")\
    .setWhiteList(["ORG"])

fin_assertion = finance.AssertionDLModel.pretrained("finassertiondl_acquisitions", "en", "finance/models")\
    .setInputCols(["sentence", "ner_chunk", "embeddings"])\
    .setOutputCol("assertion")\
    .setScopeWindow([10, 10])

nlpPipeline = nlp.Pipeline(stages=[
        document_assembler,
        sentence_detector,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter,
        fin_assertion
])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

text = """On April 10, 2018, EQT Partners partially acquired a majority stake in Spirit Communications with the intent to combine Spirit."""

data = spark.createDataFrame([[text]]).toDF("text")

result = model.transform(data)
```

</div>

## Results

```bash
+---------------------+-------------------+
|            ner_chunk|          assertion|
+---------------------+-------------------+
|         EQT Partners|PARTIAL_ACQUISITION|
|Spirit Communications|PARTIAL_ACQUISITION|
+---------------------+-------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finassertiondl_acquisitions|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|en|
|Size:|1.7 MB|

## References

Wikidata

## Benchmarking

```bash
label                precision  recall  f1-score  support 
PARTIAL_ACQUISITION  0.99       1.00    0.99      521     
TOTAL_ACQUISITION    0.87       0.88    0.88      237     
OTHER                0.90       0.88    0.89      248     
accuracy             -          -       0.94      1006    
macro-avg            0.92       0.92    0.92      1006    
weighted-avg         0.94       0.94    0.94      1006  
```
