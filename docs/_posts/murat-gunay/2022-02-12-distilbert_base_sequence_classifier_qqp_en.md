---
layout: model
title: Duplicate Question Detection
author: John Snow Labs
name: distilbert_base_sequence_classifier_qqp
date: 2022-02-12
tags: [en, open_source]
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

This model was imported from `Hugging Face` ([link](https://huggingface.co/assemblyai/distilbert-base-uncased-qqp)) and it's been trained on Quora Question Pairs dataset, leveraging `Distil-BERT` embeddings and `DistilBertForSequenceClassification` for text classification purposes. As an input, it requires two questions separated by a space.

## Predicted Entities

`non_duplicated`, `duplicated`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_sequence_classifier_qqp_en_3.4.0_3.0_1644663826044.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_base_sequence_classifier_qqp_en_3.4.0_3.0_1644663826044.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

 sequenceClassifier = DistilBertForSequenceClassification.pretrained("distilbert_base_sequence_classifier_qqp", "en")\
   .setInputCols(["document",'token'])\
   .setOutputCol("class")

 pipeline = Pipeline(stages=[document_assembler, tokenizer, sequenceClassifier])

 light_pipeline = LightPipeline(pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

 result1 = light_pipeline.annotate("Do we have to go there? Are you a doctor?")
 result2 = light_pipeline.annotate("Do you want to eat something? Are you hungry?")
```
```scala
val document_assembler = DocumentAssembler()
     .setInputCol("text")
     .setOutputCol("document")

 val tokenizer = Tokenizer()
     .setInputCols(Array("document"))
     .setOutputCol("token")

 val sequenceClassifier = DistilBertForSequenceClassification.pretrained("distilbert_base_sequence_classifier_qqp", "en")
   .setInputCols(Array("document", "token"))
   .setOutputCol("class")

 val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

 val example1 = Seq.empty["Do we have to go there? Are you a doctor?"].toDS.toDF("text")
 val example2 = Seq.empty["Do you want to eat something? Are you hungry?"].toDS.toDF("text")
 val result1 = pipeline.fit(example1).transform(example1)
 val result2 = pipeline.fit(example2).transform(example2)
```
</div>

## Results

```bash
['non_duplicated']
['duplicated']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_base_sequence_classifier_qqp|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|249.8 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

[https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)