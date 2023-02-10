---
layout: model
title: Detect former names of companies in texts (small)
author: John Snow Labs
name: finner_wiki_formername
date: 2023-01-15
tags: [former, name, en, licensed]
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

This is an NER model, aimed to detect Former Names of companies. It was trained with wikipedia texts about companies.

## Predicted Entities

`FORMER_NAME`, `O`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_wiki_formername_en_1.0.0_3.0_1673797854205.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner = finance.NerModel().pretrained("finner_wiki_formername", "en", "finance/models")\
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
+-----------+------------------+---+-----------+
|sentence_id|chunk             |end|ner_label  |
+-----------+------------------+---+-----------+
|0          |Toro Motor Company|57 |FORMER_NAME|
+-----------+------------------+---+-----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_wiki_formername|
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
I-FORMER_NAME	 29	 20	 13	 0.59183675	 0.6904762	 0.63736266
B-FORMER_NAME	 19	 5	 8	 0.7916667	 0.7037037	 0.7450981
Macro-average 48 25 21 0.6917517 0.6970899 0.69441056
Micro-average 48 25 21 0.65753424 0.6956522 0.6760564
```
