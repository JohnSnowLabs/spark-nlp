---
layout: model
title: Spanish RobertaForTokenClassification Cased model (from mrm8488)
author: John Snow Labs
name: roberta_ner_finetuned_bioclinical
date: 2022-07-18
tags: [open_source, roberta, ner, bioclinical, es]
task: Named Entity Recognition
language: es
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: RoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBERTa NER model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bioclinical-roberta-es-finenuned-clinical-ner` is a Spanish model originally trained by `mrm8488`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_ner_finetuned_bioclinical_es_4.0.0_3.0_1658155068450.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_ner_finetuned_bioclinical_es_4.0.0_3.0_1658155068450.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
  
ner = RoBertaForTokenClassification.pretrained("roberta_ner_finetuned_bioclinical","es") \
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

val ner = RoBertaForTokenClassification.pretrained("roberta_ner_finetuned_bioclinical","es") 
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
|Model Name:|roberta_ner_finetuned_bioclinical|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|441.1 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

https://huggingface.co/mrm8488/bioclinical-roberta-es-finenuned-clinical-ner
