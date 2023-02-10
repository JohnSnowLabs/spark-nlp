---
layout: model
title: English RoBertaForSequenceClassification Cased model (from niksmer)
author: John Snow Labs
name: roberta_classifier_manibert
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

Pretrained RoBertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `ManiBERT` is a English model originally trained by `niksmer`.

## Predicted Entities

`Multiculturalism: Negative`, `Labour Groups: Positive`, `Nationalisation`, `European Community/Union or Latin America Integration: Positive`, `Economic Growth: Positive`, `Anti-Growth Economy and Sustainability`, `Education Limitation`, `Agriculture and Farmers`, `Technology and Infrastructure: Positive`, `Incentives: Positive`, `Governmental and Administrative Efficiency`, `Anti-Imperialism`, `Free Market Economy`, `Traditional Morality: Positive`, `Labour Groups: Negative`, `Constitutionalism: Negative`, `Peace`, `Welfare State Limitation`, `Traditional Morality: Negative`, `Centralisation: Positive`, `Environmental Protection`, `Economic Goals`, `Internationalism: Negative`, `Protectionism: Negative`, `Foreign Special Relationships: Negative`, `Welfare State Expansion`, `Controlled Economy`, `Market Regulation`, `Education Expansion`, `Culture: Positive`, `Law and Order`, `Protectionism: Positive`, `Corporatism/ Mixed Economy`, `Non-economic Demographic Groups`, `Constitutionalism: Positive`, `National Way of Life: Negative`, `Military: Positive`, `Freedom and Human Rights`, `European Community/Union or Latin America Integration: Negative`, `Decentralisation: Positive`, `Multiculturalism: Positive`, `Democracy`, `Economic Planning`, `Equality: Positive`, `Underprivileged Minority Groups`, `Foreign Special Relationships: Positive`, `Political Authority`, `Economic Orthodoxy`, `Military: Negative`, `Political Corruption`, `Keynesian Demand Management`, `Marxist Analysis: Positive`, `Civic Mindedness: Positive`, `Internationalism: Positive`, `Middle Class and Professional Groups`, `National Way of Life: Positive`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_manibert_en_4.1.0_3.0_1663603671374.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_classifier_manibert_en_4.1.0_3.0_1663603671374.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_manibert","en") \
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
 
val seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_manibert","en") 
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
|Model Name:|roberta_classifier_manibert|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|458.3 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/niksmer/ManiBERT
- https://manifesto-project.wzb.eu/
- https://manifesto-project.wzb.eu/datasets
- https://manifesto-project.wzb.eu/down/tutorials/main-dataset.html#measuring-parties-left-right-positions