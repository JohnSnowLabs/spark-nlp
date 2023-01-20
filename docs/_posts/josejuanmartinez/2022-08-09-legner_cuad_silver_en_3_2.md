---
layout: model
title: NER on Legal Texts (CUAD, Silver corpus)
author: John Snow Labs
name: legner_cuad_silver
date: 2022-08-09
tags: [en, legal, ner, cuad, licensed]
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

This is a Legal Name Entity Recognition model, trained on a Silver version of the CUAD dataset. We say a corpus is on its "Silver" version when we use automatic labelling algorithms, rules, vocabularies, patterns and some predefined annotations. 

The entities included are:
"PERSON": Person
"LAW": Mentioned law
"PARTY": A party signing the agreement
"EFFDATE": Date of the agreement
"LOC": A mentioned location
"DATE": Another date, not EFFDATE
"DOC": Type of the document
"ORDINAL": And ordinal number
"ROLE": A role of a person or party
"PERCENT": A percentage
"ORG": An generic tag for detecting organizations

You can several models trained on Golden versions of this dataset (annotated by our JSL in-house domain experts) in Models Hub, looking in the Legal library.

## Predicted Entities

`PERSON`, `LAW`, `PARTY`, `EFFDATE`, `LOC`, `DATE`, `DOC`, `ORDINAL`, `ROLE`, `PERCENT`, `ORG`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_cuad_silver_en_1.0.0_3.2_1660041713538.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_cuad_silver_en_1.0.0_3.2_1660041713538.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

sentencizer = nlp.SentenceDetector()\
        .setInputCols(["document"])\
        .setOutputCol("sentences")\
        .setExplodeSentences(True)

tokenizer = nlp.Tokenizer()\
  .setInputCols(["sentences"])\
  .setOutputCol("token")
        
embeddings = nlp.WordEmbeddingsModel.pretrained("w2v_cc_300d", "en")\
    .setInputCols(["sentences", "token"])\
    .setOutputCol("embeddings")

jsl_ner = legal.NerModel.pretrained("legner_cuad_silver", "en", "legal/models")\
		.setInputCols(["sentences", "token", "embeddings"]) \
		.setOutputCol("jsl_ner")

jsl_ner_converter = nlp.NerConverter() \
		.setInputCols(["sentences", "token", "jsl_ner"]) \
		.setOutputCol("ner_chunk")
        
jsl_ner_pipeline = Pipeline().setStages([
				documentAssembler,
				sentencizer,
				tokenizer,
				embeddings,
				jsl_ner,
				jsl_ner_converter])

text = """December 2007 SUBORDINATED LOAN AGREEMENT. THIS LOAN AGREEMENT is made on 7th December, 2007 BETWEEN: (1) SILICIUM DE PROVENCE S.A.S., a private company with limited liability, incorporated under the laws of France, whose registered office is situated at Usine de Saint Auban, France, represented by Mr.Frank Wouters, hereinafter referred to as the "Borrower", and ( 2 ) EVERGREEN SOLAR INC., a company incorporated in Delaware, U.S.A., with registered number 2426798, whose registered office is situated at Bartlett Street, Marlboro, Massachusetts, U.S.A. represented by Richard Chleboski, hereinafter referred to as "Lender"."""

df = spark.createDataFrame([[text]]).toDF("text")

model = jsl_ner_pipeline.fit(df)
res = model.transform(df)

```

</div>

## Results

```bash
+------------+---------+----------+
|       token|ner_label|confidence|
+------------+---------+----------+
|    December|   B-DATE|    0.4111|
|        2007|   B-DATE|    0.7867|
|SUBORDINATED|        O|    0.5373|
|        LOAN|    B-DOC|    0.9998|
|   AGREEMENT|    I-DOC|    0.8615|
|           .|        O|    0.9695|
|        THIS|        O|    0.9977|
|        LOAN|    B-DOC|    0.9995|
|   AGREEMENT|    I-DOC|    0.9982|
|          is|        O|    0.8592|
|        made|        O|    0.9975|
|          on|        O|    0.9906|
|         7th|   B-DATE|    0.7804|
|    December|   B-DATE|    0.6701|
|           ,|   B-DATE|    0.5395|
|        2007|   B-DATE|    0.5327|
|     BETWEEN|        O|    0.9771|
|           :|        O|    0.9497|
|           (|        O|    0.7493|
|           1|        O|    0.9081|
|           )|        O|    0.4178|
|    SILICIUM|    B-ORG|    0.6731|
|          DE|    B-ORG|    0.3681|
|    PROVENCE|    B-ORG|    0.5065|
|       S.A.S|    B-ORG|    0.8924|
|          .,|        O|    0.7006|
|           a|        O|    0.9722|
|     private|        O|    0.9938|
|     company|        O|    0.9982|
|        with|        O|    0.9958|
|     limited|        O|     0.981|
|   liability|        O|    0.9994|
|           ,|        O|    0.9933|
|incorporated|        O|    0.9997|
|       under|        O|    0.9597|
|         the|        O|    0.9833|
|        laws|        O|    0.9969|
|          of|        O|    0.7129|
|      France|    B-LOC|    0.8789|
+------------+---------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_cuad_silver|
|Type:|legal|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|15.0 MB|

## References

Manual rules, patterns, weak-labelling, preannotations from in-house models and from CUAD dataset

## Benchmarking

```bash
label             tp       fp    fn    prec        rec          f1
B-PERSON          89       11    11    0.89        0.89         0.89
B-LAW             759      111   148   0.8724138   0.8368247    0.8542487
I-PARTY           8632     47    23    0.9945846   0.9973426    0.9959617
B-EFFDATE         9        1     4     0.9         0.6923077    0.7826087
B-LOC             372      76    61    0.83035713  0.8591224    0.8444949
B-DATE            1020     104   102   0.9074733   0.90909094   0.9082814
B-DOC             1370     36    12    0.97439545  0.9913169    0.9827834
I-EFFDATE         14       0     0     1.0         1.0          1.0
I-DOC             2227     49    0     0.978471    1.0          0.98911834
B-ORDINAL         99       11    15    0.9         0.8684211    0.8839286
B-ROLE            228      6     0     0.974359    1.0          0.987013
B-PERCENT         34       4     0     0.8947368   1.0          0.9444445
B-ORG          	  1992     478   624   0.8064777   0.7614679    0.7833268
B-PARTY        	  2275     39    82    0.9831461   0.96521      0.97409546
Macro-average     19120    973   1082  0.92188674  0.9122217    0.9170287
Micro-average     19120    973   1082  0.95157516  0.94644094   0.9490011
```

