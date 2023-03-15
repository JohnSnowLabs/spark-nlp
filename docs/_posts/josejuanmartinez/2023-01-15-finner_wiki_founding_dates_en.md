---
layout: model
title: Detect Founding / Listing dates in texts (small)
author: John Snow Labs
name: finner_wiki_founding_dates
date: 2023-01-15
tags: [listing, founding, establishment, dates, en, licensed]
task: Named Entity Recognition
language: en
nav_key: models
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is an NER model, aimed to detect Establishment (Founding) and Listing dates of Companies. It was trained with wikipedia texts about companies.

## Predicted Entities

`FOUNDING_DATE`, `LISTING_DATE`, `O`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_wiki_founding_dates_en_1.0.0_3.0_1673798045941.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
text = "The Toro Company, formerly known as the Toro Motor Company, is an American company founded in 1980. It was listed on the NASDAQ Global Market in August 2000. It design and operates lawn mowers and snow blowers and irrigation system supplies."
df = spark.createDataFrame([[text]]).toDF("text")

documenter = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentencizer = nlp.SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")
    
embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base", "en") \
    .setInputCols("sentence", "token") \
    .setOutputCol("embeddings")\
    .setMaxSentenceLength(512)

chunks = finance.NerConverterInternal()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

ner = finance.NerModel().pretrained("finner_wiki_founding_dates", "en", "finance/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

 pipe = nlp.Pipeline(stages=[documenter, sentencizer, tokenizer, embeddings, ner, chunks])
 model = pipe.fit(df)
 res = model.transform(df)


res.select(F.explode(F.arrays_zip(res.ner_chunk.result, res.ner_chunk.begin, res.ner_chunk.end, res.ner_chunk.metadata)).alias("cols")) \
       .select(F.expr("cols['3']['sentence']").alias("sentence_id"),
               F.expr("cols['0']").alias("chunk"),
               F.expr("cols['2']").alias("end"),
               F.expr("cols['3']['entity']").alias("ner_label"))\
       .filter("ner_label!='O'")\
       .show(truncate=False)
```

</div>

## Results

```bash
+-----------+-----------+---+-------------+
|sentence_id|chunk      |end|ner_label    |
+-----------+-----------+---+-------------+
|0          |1980       |97 |FOUNDING_DATE|
|1          |August 2000|155|LISTING_DATE |
+-----------+-----------+---+-------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_wiki_founding_dates|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|1.1 MB|

## References

Wikipedia

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
B-LISTING_DATE	 10	 0	 4	 1.0	 0.71428573	 0.8333334
B-FOUNDING_DATE	 18	 3	 2	 0.85714287	 0.9	 0.87804884
I-LISTING_DATE	 8	 0	 1	 1.0	 0.8888889	 0.94117653
Macro-average 36 4 9 4 0.9 0.8 0.8470588
Micro-average 36 4 9 4 0.9 0.8 0.8470588
```