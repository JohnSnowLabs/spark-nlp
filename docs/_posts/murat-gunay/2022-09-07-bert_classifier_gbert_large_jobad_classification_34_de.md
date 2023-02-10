---
layout: model
title: German BertForSequenceClassification Large Cased model (from dbb)
author: John Snow Labs
name: bert_classifier_gbert_large_jobad_classification_34
date: 2022-09-07
tags: [de, open_source, bert, sequence_classification, classification]
task: Text Classification
language: de
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `gbert-large-jobad-classification-34` is a German model originally trained by `dbb`.

## Predicted Entities

`hr/recruiting`, `chem. pharm. ausbildung`, `vertrieb/kundenbetreuung`, `kaufm. studium`, `beschaffung/supply chain`, `med. tech. beruf`, `it studium`, `log. ausbildung`, `it ausbildung`, `it`, `med. ausbildung`, `pflege/therapie`, `tech. ausbildung`, `logistik/transport`, `gastro. touri. ausbildung`, `bildung/soziales`, `administration/sekretariat`, `kaufm. ausbildung`, `controlling/finanzen`, `labor/forschung`, `indust. produktion`, `chem. pharm. beruf`, `rettungsdienst/sicherheit`, `recht/justiz`, `med. verwaltung`, `quali. kontr./-management`, `marketing/kommunikation`, `tech. studium`, `gastro./tourismus`, `indust. konstruk./ingenieur`, `mechaniker/techniker/elektriker`, `arzt`, `hausverw./-bewirt.`, `baugewerbe/-ingenieur`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_gbert_large_jobad_classification_34_de_4.1.0_3.0_1662513179471.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_gbert_large_jobad_classification_34_de_4.1.0_3.0_1662513179471.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_gbert_large_jobad_classification_34","de") \
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
 
val seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_gbert_large_jobad_classification_34","de") 
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
|Model Name:|bert_classifier_gbert_large_jobad_classification_34|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|de|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/dbb/gbert-large-jobad-classification-34