---
layout: model
title: Extract Dates in Financial Documents
author: John Snow Labs
name: finner_sec_dates
date: 2022-11-01
tags: [date, en, licensed]
task: Named Entity Recognition
language: en
edition: Spark NLP for Finance 1.0.0
spark_version: 3.0
supported: true
annotator: FinanceNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This NER models uses `bert_embeddings_sec_bert_base` embeddings, trained on SEC documents and empowered with OntoNotes 2022, to extract DATES. This model is light but very accurate.

## Predicted Entities

`DATE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_sec_dates_en_1.0.0_3.0_1667305896514.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finner_sec_dates_en_1.0.0_3.0_1667305896514.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence = nlp.SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokens = nlp.Tokenizer()\
    .setInputCols("sentence")\
    .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base", "en") \
    .setInputCols("sentence", "token") \
    .setOutputCol("embeddings")

ner = finance.NerModel.pretrained("finner_sec_dates", "en", "finance/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("label")

text  = "For the fiscal year ended December 31, 2021, Amazon reported a profit of ..."
df = spark.createDataFrame([text], StringType()).toDF('text')

pipeline = nlp.Pipeline(stages = [document, sentence, tokens, embeddings, ner])
fit_model = pipeline.fit(df)

res = fit_model.transform(df)

res.select(F.explode(F.arrays_zip(res.token.result, 
                                            res.label.result)).alias("cols")) \
             .select(F.expr("cols['0']").alias("token"),
                     F.expr("cols['1']").alias("label")).show(truncate=50)
```

</div>

## Results

```bash
+--------+------+
|   token| label|
+--------+------+
|     For|     O|
|     the|B-DATE|
|  fiscal|I-DATE|
|    year|I-DATE|
|   ended|I-DATE|
|December|I-DATE|
|      31|I-DATE|
|       ,|I-DATE|
|    2021|I-DATE|
|       ,|     O|
|  Amazon|     O|
|reported|     O|
|       a|     O|
|  profit|     O|
|      of|     O|
|       .|     O|
|       .|     O|
|       .|     O|
+--------+------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_sec_dates|
|Compatibility:|Spark NLP for Finance 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.6 MB|

## References

In-house annotations on SEC 10K filings, Ontonotes 2012

## Benchmarking

```bash
label	  tp	 fp	     fn	     prec	     rec	     f1
B-DATE	  3572	 278	 252	 0.9277922	 0.9341004	 0.9309356
I-DATE	  4300	 339	 245	 0.92692393	 0.94609463	 0.9364112
macro-avg 7872   617     497     0.92735803  0.9400975   0.93368435
micro-avg 7872   617     497     0.9273177   0.94061416  0.93391865
```