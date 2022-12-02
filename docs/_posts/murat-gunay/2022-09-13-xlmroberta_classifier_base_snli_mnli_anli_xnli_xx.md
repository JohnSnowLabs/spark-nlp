---
layout: model
title: Multilingual XlmRobertaForSequenceClassification Base Cased model (from symanto)
author: John Snow Labs
name: xlmroberta_classifier_base_snli_mnli_anli_xnli
date: 2022-09-13
tags: [en, de, fr, es, ru, ar, tr, zh, el, th, bg, ur, open_source, xlm_roberta, sequence_classification, classification, xx]
task: Text Classification
language: xx
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: XlmRoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRobertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `xlm-roberta-base-snli-mnli-anli-xnli` is a Multilingual model originally trained by `symanto`.

## Predicted Entities

`ENTAILMENT`, `NEUTRAL`, `CONTRADICTION`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_classifier_base_snli_mnli_anli_xnli_xx_4.1.0_3.0_1663063902373.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

seq_classifier = XlmRoBertaForSequenceClassification.pretrained("xlmroberta_classifier_base_snli_mnli_anli_xnli","xx") \
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
 
val seq_classifier = XlmRoBertaForSequenceClassification.pretrained("xlmroberta_classifier_base_snli_mnli_anli_xnli","xx") 
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
|Model Name:|xlmroberta_classifier_base_snli_mnli_anli_xnli|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|xx|
|Size:|900.4 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/symanto/xlm-roberta-base-snli-mnli-anli-xnli
- https://nlp.stanford.edu/projects/snli/
- https://cims.nyu.edu/~sbowman/multinli/
- https://github.com/facebookresearch/anli
- https://github.com/facebookresearch/XNLI