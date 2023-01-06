---
layout: model
title: Legal NER from Sigma Absa Dataset (PER+Pronouns)
author: John Snow Labs
name: legner_sigma_absa_people
date: 2022-12-16
tags: [sigma, absa, people, pronouns, en, licensed]
task: Named Entity Recognition
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Legal NER model trained on the Sigma Absa Dataset for legal sentiment analysis on legal parties, including correference pronouns (he, him, their...). This is the first component which extracts those people names and pronouns and NER.

You have the second component, which does Assertion Status to retrieve sentiment, on `legassertion_sigma_absa_sentiment`

## Predicted Entities

`PER`, `O`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_sigma_absa_people_en_1.0.0_3.0_1671202164090.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = nlp.SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")

ner = legal.NerDLModel.pretrained("legner_sigma_absa_people", "en", "legal/models")\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("label")

pipe = nlp.Pipeline(stages = [ document_assembler, sentence_detector, tokenizer, embeddings, ner])

text = "Petitioner Jae Lee moved to the United States from South Korea with his parents when he was 13."

sdf = spark.createDataFrame([[text]]).toDF("text")
res = pipe.fit(sdf).transform(sdf)

import pyspark.sql.functions as F
res.select(F.explode(F.arrays_zip(res.token.result, 
                                     res.label.result, 
                                     res.label.metadata)).alias("cols"))\
                  .select(F.expr("cols['0']").alias("token"),
                          F.expr("cols['1']").alias("ner_label"),
                          F.expr("cols['2']['confidence']").alias("confidence")).show(200, truncate=100)
```

</div>

## Results

```bash
+----------+---------+----------+
|     token|ner_label|confidence|
+----------+---------+----------+
|Petitioner|    B-PER|    0.9997|
|       Jae|    I-PER|    0.9952|
|       Lee|    I-PER|    0.9951|
|     moved|        O|       1.0|
|        to|        O|       1.0|
|       the|        O|       1.0|
|    United|        O|       1.0|
|    States|        O|       1.0|
|      from|        O|       1.0|
|     South|        O|       1.0|
|     Korea|        O|       1.0|
|      with|        O|       1.0|
|       his|    B-PER|       1.0|
|   parents|        O|    0.9998|
|      when|        O|       1.0|
|        he|    B-PER|       1.0|
|       was|        O|       1.0|
|        13|        O|       1.0|
|         .|        O|       1.0|
+----------+---------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_sigma_absa_people|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.1 MB|

## References

https://metatext.io/datasets/sigmalaw-absa

## Benchmarking

```bash

 label          tp   fp  fn  prec        rec         f1         
 I-PER          43   2   0   0.95555556  1.0         0.97727275 
 B-PER          777  11  15  0.9860406   0.9810606   0.98354435 
 Macro-average  820  13  15  0.9707981   0.9905303   0.9805649  
 Micro-average  820  13  15  0.9843938   0.98203593  0.9832135  
 
```