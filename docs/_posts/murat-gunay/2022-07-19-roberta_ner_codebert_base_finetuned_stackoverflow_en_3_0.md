---
layout: model
title: English RobertaForTokenClassification Base Cased model (from mrm8488)
author: John Snow Labs
name: roberta_ner_codebert_base_finetuned_stackoverflow
date: 2022-07-19
tags: [open_source, roberta, ner, stackoverflow, codebert, en]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: RoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBERTa NER model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `codebert_finetuned_stackoverflow` is a English model originally trained by `mrm8488`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_ner_codebert_base_finetuned_stackoverflow_en_4.0.0_3.0_1658212367861.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")
  
ner = RoBertaForTokenClassification.pretrained("roberta_ner_codebert_base_finetuned_stackoverflow","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, ner])

data = spark.createDataFrame([["PUT YOUR STRING HERE."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCol("text") 
      .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
    .setInputCols(Array("document"))
    .setOutputCol("sentence")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

val ner = RoBertaForTokenClassification.pretrained("roberta_ner_codebert_base_finetuned_stackoverflow","en") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, ner))

val data = Seq("PUT YOUR STRING HERE.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_ner_codebert_base_finetuned_stackoverflow|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|467.0 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

https://huggingface.co/mrm8488/codebert-base-finetuned-stackoverflow-ner