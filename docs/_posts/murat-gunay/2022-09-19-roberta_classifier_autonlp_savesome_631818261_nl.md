---
layout: model
title: Dutch RoBertaForSequenceClassification Cased model (from test1345)
author: John Snow Labs
name: roberta_classifier_autonlp_savesome_631818261
date: 2022-09-19
tags: [nl, open_source, roberta, sequence_classification, classification]
task: Text Classification
language: nl
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `autonlp-savesome-631818261` is a Dutch model originally trained by `test1345`.

## Predicted Entities

`Koeken of Chocolade of Snoep`, `Niet-voeding`, `Dranken`, `Bereidingen of Charcuterie of Vis of Veggie`, `Wijn`, `Onderhoud of Huishouden`, `Zuivel`, `Dieetvoeding of Voedingssupplementen`, `Baby`, `Diepvries`, `Groenten en fruit`, `Lichaamsverzorging of Parfumerie`, `Conserven`, `Huisdieren`, `Kruidenierswaren of Droge voeding`, `Colruyt-beenhouwerij`, `Chips of Borrelhapjes`, `Brood of Ontbijt`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_autonlp_savesome_631818261_nl_4.1.0_3.0_1663606218511.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_classifier_autonlp_savesome_631818261_nl_4.1.0_3.0_1663606218511.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_autonlp_savesome_631818261","nl") \
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
 
val seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_autonlp_savesome_631818261","nl") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("nl.classify.roberta").predict("""PUT YOUR STRING HERE""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_classifier_autonlp_savesome_631818261|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|nl|
|Size:|438.7 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/test1345/autonlp-savesome-631818261