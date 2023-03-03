---
layout: model
title: Detect Nationality / Company Founding Places in texts
author: John Snow Labs
name: finner_wiki_nationality
date: 2023-01-15
tags: [en, licensed]
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

This is an NER model, aimed to detect Nationalities, more specifically when talking about countries founding place. It was trained with wikipedia texts about companies.

## Predicted Entities

`NATIONALITY`, `O`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_wiki_nationality_en_1.0.0_3.0_1673797584937.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
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

ner = finance.NerModel().pretrained("finner_wiki_nationality", "en", "finance/models")\
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
+-----------+--------+---+-----------+
|sentence_id|chunk   |end|ner_label  |
+-----------+--------+---+-----------+
|0          |American|73 |NATIONALITY|
+-----------+--------+---+-----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_wiki_nationality|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|1.2 MB|

## References

Wikipedia

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
B-NATIONALITY	 57	 7	 1	 0.890625	 0.98275864	 0.93442625
Macro-average 57 7 1 0.890625 0.98275864	 0.93442625
Micro-average 57 7 1 0.890625 0.98275864	 0.93442625
```