---
layout: model
title: Legal Merger Agreement Classification (MAUD)
author: John Snow Labs
name: legclf_bert_maud
date: 2023-01-31
tags: [legal, en, classification, bert, licensed, tensorflow]
task: Text Classification
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: LegalBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Bert-based model, which can be used to classify texts into 7 classes. This is a Multiclass model, meaning only one label will be returned as an output.
This dataset includes 152 merger agreements with 39,000 multiple-choice reading comprehension samples that have been manually tagged by lawyers.

## Predicted Entities

`Conditions to Closing`, `Deal Protection and Related Provisions`, `General Information`, `Knowledge`, `Material Adverse Effect`, `Operating and Efforts Covenant`, `Remedies`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_bert_maud_en_1.0.0_3.0_1675177515992.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_bert_maud_en_1.0.0_3.0_1675177515992.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = nlp.Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier = legal.BertForSequenceClassification.pretrained("legclf_bert_maud", "en", "legal/models")\
  .setInputCols(["document",'token'])\
  .setOutputCol("class")
  
pipeline = nlp.Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier
])

# couple of simple examples
example = spark.createDataFrame([["""(i)            Conversion of Company Common Stock. Each Share (including each Restricted Share) issued and outstanding immediately prior to the Effective Time, other than Excluded Shares, shall be cancelled and extinguished and automatically converted into the right to receive $70 in cash, without interest, subject to deduction for any required withholding Tax required to be withheld therefrom under applicable Law, in accordance with Section 2.05 (the “Merger Consideration”), and all of such Shares shall cease to be outstanding, shall cease to exist, and each certificate representing a Share (a “Certificate”) or a non-certificated Share represented by book-entry (“Book-Entry Shares”) that formerly represented any of the Shares (other than Excluded Shares) shall thereafter be cancelled and cease to have any rights with respect thereto, except the right to receive the Merger Consideration without interest thereon, subject to deduction for any required withholding Tax required to be withheld therefrom under applicable Law, in accordance with Section 2.05.  (Page 4)

SECTION 1.1      Increase of Merger Consideration. Section 2.01(a)(i) of the Merger Agreement is hereby amended by replacing the reference therein to “$70” as the Merger Consideration with “$75” as the Merger Consideration.  (Page 1)"""]]).toDF("text")

result = pipeline.fit(example).transform(example)

# result is a DataFrame
result.select("text", "class.result").show()

```

</div>

## Results

```bash
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
|text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |result               |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+
|(i)            Conversion of Company Common Stock. Each Share (including each Restricted Share) issued and outstanding immediately prior to the Effective Time, other than Excluded Shares, shall be cancelled and extinguished and automatically converted into the right to receive $70 in cash, without interest, subject to deduction for any required withholding Tax required to be withheld therefrom under applicable Law, in accordance with Section 2.05 (the “Merger Consideration”), and all of such Shares shall cease to be outstanding, shall cease to exist, and each certificate representing a Share (a “Certificate”) or a non-certificated Share represented by book-entry (“Book-Entry Shares”) that formerly represented any of the Shares (other than Excluded Shares) shall thereafter be cancelled and cease to have any rights with respect thereto, except the right to receive the Merger Consideration without interest thereon, subject to deduction for any required withholding Tax required to be withheld therefrom under applicable Law, in accordance with Section 2.05.  (Page 4)

SECTION 1.1      Increase of Merger Consideration. Section 2.01(a)(i) of the Merger Agreement is hereby amended by replacing the reference therein to “$70” as the Merger Consideration with “$75” as the Merger Consideration.  (Page 1)|[General Information]|
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_bert_maud|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|402.5 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

MAUD dataset and in-house data augmentation

## Benchmarking

```bash
label                                     precision    recall  f1-score   support
                 Conditions to Closing       1.00      1.00      1.00       204
Deal Protection and Related Provisions       1.00      0.89      0.94       210
                   General Information       1.00      1.00      1.00        66
                             Knowledge       0.83      0.99      0.90       112
               Material Adverse Effect       0.99      1.00      1.00       171
        Operating and Efforts Covenant       1.00      1.00      1.00       209
                              Remedies       1.00      1.00      1.00        54
                              accuracy         -        -        0.98      1026
                             macro-avg       0.97      0.98      0.98      1026
                          weighted-avg       0.98      0.98      0.98      1026
```