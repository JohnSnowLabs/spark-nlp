---
layout: model
title: Financial ORG, PER, ROLE, DATE NER
author: John Snow Labs
name: finner_org_per_role_date
date: 2022-08-30
tags: [en, finance, ner, licensed]
task: Named Entity Recognition
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is an NER model trained on SEC 10K documents, aimed to extract the following entities:
- ORG
- PERSON
- ROLE
- DATE

## Predicted Entities

`ORG`, `PER`, `DATE`, `ROLE`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FINPIPE_ORG_PER_DATE_ROLES/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_org_per_role_date_en_1.0.0_3.2_1661862246514.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finner_org_per_role_date_en_1.0.0_3.2_1661862246514.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
        
sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_model = finance.NerModel.pretrained("finner_org_per_role_date", "en", "finance/models")\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")
        
ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter,
])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

text = ["""Jeffrey Preston Bezos is an American entrepreneur, founder and CEO of Amazon"""]

result = model.transform(spark.createDataFrame([text]).toDF("text"))

```

</div>

## Results

```bash
+-----+---+---------------------+------+
|begin|end|                chunk|entity|
+-----+---+---------------------+------+
|    0| 20|Jeffrey Preston Bezos|PERSON|
|   37| 48|         entrepreneur|  ROLE|
|   51| 57|              founder|  ROLE|
|   63| 65|                  CEO|  ROLE|
|   70| 75|               Amazon|   ORG|
+-----+---+---------------------+------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_org_per_role_date|
|Type:|finance|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.3 MB|

## References

SEC 10-K filings with in-house annotations

## Benchmarking

```bash
label              tp     fp    fn        prec           rec            f1
B-PERSON           254    20    56        0.9270073      0.81935483     0.86986303
I-ORG              1161   133   231       0.8972179      0.8340517      0.8644826
B-DATE             202    15    14        0.9308756      0.9351852      0.9330255
I-DATE             302    29    12        0.9123867      0.96178347     0.93643415
B-ROLE             219    21    47        0.9125         0.8233083      0.8656126
B-ORG              674    92    163       0.87989557     0.80525684     0.84092325
I-ROLE             260    26    68        0.90909094     0.79268295     0.8469055
I-PERSON           501    34	94        0.9364486      0.8420168      0.88672566
Macro-average      3573   370   685       0.91317785     0.851705       0.8813709
Micro-average      3573   370   685       0.9061628      0.83912635     0.8713572
```
