---
layout: model
title: Sentiment Analysis for Food Reviews
author: John Snow Labs
name: distilbert_base_sequence_classifier_food
date: 2022-02-14
tags: [food, amazon, distilbert, sequence_classification, en, open_source]
task: Text Classification
language: en
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: true
annotator: DistilBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model was imported from `Hugging Face` ([link](https://huggingface.co/Tejas3/distillbert_base_uncased_amazon_food_review_300)) and it's been trained on Amazon Food Review dataset, leveraging `Distil-BERT` embeddings and `DistilBertForSequenceClassification` for text classification purposes. The model classifies `Positive` or `Negative` sentiments of texts related to food reviews.

## Predicted Entities

`Positive`, `Negative`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_sequence_classifier_food_en_3.4.0_3.0_1644846756022.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_base_sequence_classifier_food_en_3.4.0_3.0_1644846756022.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
     .setInputCol('text') \
     .setOutputCol('document')

 tokenizer = Tokenizer() \
     .setInputCols(['document']) \
     .setOutputCol('token')

 sequenceClassifier = DistilBertForSequenceClassification.pretrained("distilbert_base_sequence_classifier_food", "en")\
   .setInputCols(["document",'token'])\
   .setOutputCol("class")

 pipeline = Pipeline(stages=[document_assembler, tokenizer, sequenceClassifier])

 light_pipeline = LightPipeline(pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

result = light_pipeline.annotate("The first time I ever used them was about 9 months ago. The food that came was just left at my front doorstep all over the place. Bread was smashed, bananas nearly rotten, and containers crushed. Given the weather, I decided to give them another try. This time my order was cancelled 6 hours before delivery. When cancelled, they didn't even give an explanation as to why it was cancelled. Amazon just needs to close up this portion of their shop.")
```
```scala
val document_assembler = DocumentAssembler()
     .setInputCol("text")
     .setOutputCol("document")

val tokenizer = Tokenizer()
     .setInputCols(Array("document"))
     .setOutputCol("token")

 val sequenceClassifier = DistilBertForSequenceClassification.pretrained("distilbert_base_sequence_classifier_food", "en")
   .setInputCols(Array("document", "token"))
   .setOutputCol("class")

 val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

 val example = Seq.empty["The first time I ever used them was about 9 months ago. The food that came was just left at my front doorstep all over the place. Bread was smashed, bananas nearly rotten, and containers crushed. Given the weather, I decided to give them another try. This time my order was cancelled 6 hours before delivery. When cancelled, they didn't even give an explanation as to why it was cancelled. Amazon just needs to close up this portion of their shop."].toDS.toDF("text")

val result = pipeline.fit(example1).transform(example)
```
</div>

## Results

```bash
['Negative']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_base_sequence_classifier_food|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|249.8 MB|
|Case sensitive:|true|
|Max sentence length:|256|