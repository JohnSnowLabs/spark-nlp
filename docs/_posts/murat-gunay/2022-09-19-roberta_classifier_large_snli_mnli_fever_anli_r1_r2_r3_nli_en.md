---
layout: model
title: English RoBertaForSequenceClassification Large Cased model (from ynie)
author: John Snow Labs
name: roberta_classifier_large_snli_mnli_fever_anli_r1_r2_r3_nli
date: 2022-09-19
tags: [en, open_source, roberta, sequence_classification, classification]
task: Text Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli` is a English model originally trained by `ynie`.

## Predicted Entities

`neutral`, `entailment`, `contradiction`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_large_snli_mnli_fever_anli_r1_r2_r3_nli_en_4.1.0_3.0_1663617247539.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_large_snli_mnli_fever_anli_r1_r2_r3_nli","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, seq_classifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCols(Array("text")) 
      .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_large_snli_mnli_fever_anli_r1_r2_r3_nli","en") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_classifier_large_snli_mnli_fever_anli_r1_r2_r3_nli|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|847.3 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli
- https://nlp.stanford.edu/projects/snli/
- https://cims.nyu.edu/~sbowman/multinli/
- https://github.com/easonnie/combine-FEVER-NSMN/blob/master/other_resources/nli_fever.md
- https://github.com/facebookresearch/anli
- https://easonnie.github.io