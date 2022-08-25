---
layout: model
title: Financial Job Titles NER
author: John Snow Labs
name: finner_roles
date: 2022-08-25
tags: [en, finance, ner, job, titles, jobs, roles, licensed]
task: Named Entity Recognition
language: en
edition: Spark NLP for Finance 1.0.0
spark_version: 3.2
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This NER model is aimed to extract Job Titles / Roles of people in Companies, and was trained using Resumes, Wikipedia Articles, Financial and Legal documents, annotated in-house.

## Predicted Entities

`ROLE`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FINNER_ROLES/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_roles_en_1.0.0_3.2_1661419024728.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
        
sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

embeddings = BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_model = FinanceNerModel().pretrained("finner_roles", "en", "finance/models")\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

text = ["""Jeffrey Preston Bezos is an American entrepreneur, founder and CEO of Amazon"""]

result = model.transform(spark.createDataFrame([text]).toDF("text"))

result_df = result.select(F.explode(F.arrays_zip(result.token.result,result.ner.result, result.ner.metadata)).alias("cols"))\
                  .select(F.expr("cols['0']").alias("token"),
                          F.expr("cols['1']").alias("ner_label"),
                          F.expr("cols['2']['confidence']").alias("confidence"))

result_df.show(50, truncate=100)
```

</div>

## Results

```bash
+------------+---------+----------+
|       token|ner_label|confidence|
+------------+---------+----------+
|     Jeffrey|        O|    0.9984|
|     Preston|        O|    0.9878|
|       Bezos|        O|    0.9939|
|          is|        O|     0.999|
|          an|        O|    0.9988|
|    American|   B-ROLE|    0.8294|
|entrepreneur|   I-ROLE|    0.9358|
|           ,|        O|    0.9979|
|     founder|   B-ROLE|    0.8645|
|         and|        O|     0.857|
|         CEO|   B-ROLE|      0.72|
|          of|        O|     0.995|
|      Amazon|        O|    0.9428|
+------------+---------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_roles|
|Type:|finance|
|Compatibility:|Spark NLP for Finance 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|15.1 MB|

## References

In-house annotations on Wikidata, CUAD dataset, Financial 10-K documents and Resumes

## Benchmarking

```bash
Total test loss: 29.2454	Avg test loss: 0.5518
label	 tp	 fp	 fn	 prec	 rec	 f1
B-ROLE	 3553	 174	 262	 0.95331365	 0.9313237	 0.9421904
I-ROLE	 4868	 250	 243	 0.9511528	 0.95245546	 0.9518037
tp: 8421 fp: 424 fn: 505 labels: 2
Macro-average	 prec: 0.9522332, rec: 0.9418896, f1: 0.9470331
Micro-average	 prec: 0.9520633, rec: 0.9434237, f1: 0.94772375

```