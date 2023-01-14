---
layout: model
title: Legal Warranty NER (md)
author: John Snow Labs
name: legner_warranty_md
date: 2022-12-01
tags: [warranty, en, licensed]
task: Named Entity Recognition
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: LegalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

IMPORTANT: Don't run this model on the whole legal agreement. Instead:
- Split by paragraphs. You can use [notebook 1](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings_JSL) in Finance or Legal as inspiration;
- Use the `legclf_warranty_clause` Text Classifier to select only these paragraphs; 

This is a Legal Named Entity Recognition Model to identify the Subject (who), Action (what), Object(the indemnification) and Indirect Object (to whom) from Warranty clauses.

This is a `md` (medium version) of the classifier, trained with more data and being more resistent to false positives outside the specific section, which may help to run it at whole document level (although not recommended).

## Predicted Entities

`WARRANTY`, `WARRANTY_ACTION`, `WARRANTY_SUBJECT`, `WARRANTY_INDIRECT_OBJECT`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_warranty_md_en_1.0.0_3.0_1669893390077.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_model = legal.NerModel.pretrained('legner_warranty_md', 'en', 'legal/models')\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = nlp.Pipeline(stages=[documentAssembler,sentenceDetector,tokenizer,embeddings,ner_model,ner_converter])

data = spark.createDataFrame([["""8 . Representations and Warranties SONY hereby makes the following representations and warranties to PURCHASER , each of which shall be true and correct as of the date hereof and as of the Closing Date , and shall be unaffected by any investigation heretofore or hereafter made : 8.1 Power and Authority SONY has the right and power to enter into this IP Agreement and to transfer the Transferred Patents and to grant the license set forth in Section 3.1 ."""]]).toDF("text")

result = nlpPipeline.fit(data).transform(data)
```

</div>

## Results

```bash
+--------------------------------------------------------------------------+------------------------+
|chunk                                                                     |entity                  |
+--------------------------------------------------------------------------+------------------------+
|SONY                                                                      |WARRANTY_SUBJECT        |
|makes the following representations and warranties                        |WARRANTY_ACTION         |
|PURCHASER                                                                 |WARRANTY_INDIRECT_OBJECT|
|shall be true and correct as of the date hereof and as of the Closing Date|WARRANTY                |
|shall be unaffected by any investigation                                  |WARRANTY                |
|SONY                                                                      |WARRANTY_SUBJECT        |
|has the right and power to enter into this IP Agreement                   |WARRANTY                |
+--------------------------------------------------------------------------+------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_warranty_md|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.1 MB|

## References

In-house annotated examples from CUAD legal dataset

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
I-WARRANTY_SUBJECT	 23	 9	 19	 0.71875	 0.54761904	 0.62162155
B-WARRANTY	 111	 36	 34	 0.75510204	 0.76551723	 0.760274
B-WARRANTY_SUBJECT	 55	 31	 33	 0.6395349	 0.625	 0.6321839
I-WARRANTY_INDIRECT_OBJECT	 18	 6	 3	 0.75	 0.85714287	 0.79999995
I-WARRANTY_ACTION	 77	 8	 14	 0.90588236	 0.84615386	 0.875
B-WARRANTY_ACTION	 36	 4	 4	 0.9	 0.9	 0.9
I-WARRANTY	 1686	 487	 313	 0.7758859	 0.8434217	 0.8082455
B-WARRANTY_INDIRECT_OBJECT	 34	 12	 6	 0.73913044	 0.85	 0.79069775
Macro-average	 2040 593 426 0.7730357 0.7793569 0.7761834
Micro-average	 2040 593 426 0.77478164 0.8272506 0.80015695
```
