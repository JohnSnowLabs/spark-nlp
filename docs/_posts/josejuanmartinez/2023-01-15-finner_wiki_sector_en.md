---
layout: model
title: Detect Company Sectors in texts (small)
author: John Snow Labs
name: finner_wiki_sector
date: 2023-01-15
tags: [en, licensed]
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

This is an NER model, aimed to detect Company Sectors. It was trained with wikipedia texts about companies.

## Predicted Entities

`SECTOR`, `O`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_wiki_sector_en_1.0.0_3.0_1673797361843.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner = finance.NerModel().pretrained("finner_wiki_sector", "en", "finance/models")\
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
+-----------+-----------------+---+---------+
|sentence_id|chunk            |end|ner_label|
+-----------+-----------------+---+---------+
|1          |lawn mowers      |175|SECTOR   |
|1          |snow blowers     |192|SECTOR   |
|1          |irrigation system|214|SECTOR   |
+-----------+-----------------+---+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_wiki_sector|
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
B-SECTOR	 70	 17	 23	 0.8045977	 0.75268817	 0.7777778
I-SECTOR	 24	 11	 9	 0.6857143	 0.72727275	 0.70588243
Macro-average 94 28 32 0.745156 0.73998046 0.7425592
Micro-average 94 32 0.7704918 0.74603176 0.7580645
```