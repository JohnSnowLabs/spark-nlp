---
layout: model
title: English BertForSequenceClassification Cased model (from jakelever)
author: John Snow Labs
name: bert_sequence_classifier_coronabert
date: 2023-03-17
tags: [en, open_source, bert, sequence_classification, ner, tensorflow]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.3.1
spark_version: 3.0
supported: true
engine: tensorflow
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `coronabert` is a English model originally trained by `jakelever`.

## Predicted Entities

`Non-human`, `Misinformation`, `Prevalence`, `Vaccines`, `News`, `Health Policy`, `Immunology`, `Inequality`, `Meta-analysis`, `Imaging`, `Infection Reports`, `Effect on Medical Specialties`, `Drug Targets`, `Transmission`, `Prevention`, `Education`, `Pediatrics`, `Medical Devices`, `Clinical Reports`, `Therapeutics`, `Communication`, `Non-medical`, `Long Haul`, `Review`, `Molecular Biology`, `Psychology`, `Diagnostics`, `Recommendations`, `Risk Factors`, `Comment/Editorial`, `Surveillance`, `Contact Tracing`, `Forecasting & Modelling`, `Healthcare Workers`, `Model Systems & Tools`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_coronabert_en_4.3.1_3.0_1679068933148.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_coronabert_en_4.3.1_3.0_1679068933148.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_coronabert","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, sequenceClassifier])

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
 
val sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_coronabert","en") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("ner")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, sequenceClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_coronabert|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|411.0 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/jakelever/coronabert
- https://coronacentral.ai
- https://github.com/jakelever/corona-ml
- https://github.com/jakelever/corona-ml/blob/master/stepByStep.md
- https://doi.org/10.1101/2020.12.21.423860
- https://github.com/jakelever/corona-ml/blob/master/machineLearningDetails.md
- https://colab.research.google.com/drive/1cBNgKd4o6FNWwjKXXQQsC_SaX1kOXDa4?usp=sharing
- https://colab.research.google.com/drive/1h7oJa2NDjnBEoox0D5vwXrxiCHj3B1kU?usp=sharing
- https://github.com/jakelever/corona-ml/tree/master/category_prediction
- https://github.com/jakelever/corona-ml/blob/master/category_prediction/annotated_documents.json.gz
- https://github.com/jakelever/corona-ml/blob/master/stepByStep.md