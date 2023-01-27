---
layout: model
title: Zero-Shot Named Entity Recognition (RoBERTa)
author: John Snow Labs
name: zero_shot_ner_roberta
date: 2022-08-29
tags: [ner, zero_shot, licensed, clinical, en, roberta]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.0.2
spark_version: 3.0
supported: true
recommended: true
annotator: ZeroShotNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is trained with Zero-Shot Named Entity Recognition (NER) approach and it can detect any kind of defined entities with no training dataset, just pretrained RoBERTa embeddings (included in the model).

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/zero_shot_ner_roberta_en_4.0.2_3.0_1661769801401.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/zero_shot_ner_roberta_en_4.0.2_3.0_1661769801401.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")
    
zero_shot_ner = ZeroShotNerModel.pretrained("zero_shot_ner_roberta", "en", "clincial/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("zero_shot_ner")\
    .setEntityDefinitions(
        {
            "NAME": ["What is his name?", "What is my name?", "What is her name?"],
            "CITY": ["Which city?", "Which is the city?"]
        })

ner_converter = NerConverter()\
    .setInputCols(["sentence", "token", "zero_shot_ner"])\
    .setOutputCol("ner_chunk")\

pipeline = Pipeline(stages = [
    documentAssembler, 
    sentenceDetector, 
    tokenizer, 
    zero_shot_ner, 
    ner_converter])

zero_shot_ner_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

data = spark.createDataFrame(["Hellen works in London, Paris and Berlin. My name is Clara, I live in New York and Hellen lives in Paris.",
                              "John is a man who works in London, London and London."], StringType()).toDF("text")

result = zero_shot_ner_model.transform(data)
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
    
val zero_shot_ner = ZeroShotNerModel.pretrained("zero_shot_ner_roberta", "en", "clincial/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("zero_shot_ner")
    .setEntityDefinitions(Map(
            "NAME"-> Array("What is his name?", "What is my name?", "What is her name?"),
            "CITY"-> Array("Which city?", "Which is the city?")
    ))

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "zero_shot_ner"))
    .setOutputCol("ner_chunk")

val pipeline = new .setStages(Array(
    documentAssembler, 
    sentenceDetector, 
    tokenizer, 
    zero_shot_ner, 
    ner_converter))

val data = Seq(Array("Hellen works in London, Paris and Berlin. My name is Clara, I live in New York and Hellen lives in Paris.",
                                     "John is a man who works in London, London and London.")toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+------+---------+--------+-----+---+----------+
| token|ner_label|sentence|begin|end|confidence|
+------+---------+--------+-----+---+----------+
|Hellen|   B-NAME|       0|    0|  5|0.13306311|
| works|        O|       0|    7| 11|      null|
|    in|        O|       0|   13| 14|      null|
|London|   B-CITY|       0|   16| 21| 0.4064213|
|     ,|        O|       0|   22| 22|      null|
| Paris|   B-CITY|       0|   24| 28|0.04597357|
|   and|        O|       0|   30| 32|      null|
|Berlin|   B-CITY|       0|   34| 39|0.16265489|
|     .|        O|       0|   40| 40|      null|
|    My|        O|       1|   42| 43|      null|
|  name|        O|       1|   45| 48|      null|
|    is|        O|       1|   50| 51|      null|
| Clara|   B-NAME|       1|   53| 57| 0.9274031|
|     ,|        O|       1|   58| 58|      null|
|     I|        O|       1|   60| 60|      null|
|  live|        O|       1|   62| 65|      null|
|    in|        O|       1|   67| 68|      null|
|   New|   B-CITY|       1|   70| 72|0.82799006|
|  York|   I-CITY|       1|   74| 77|0.82799006|
|   and|        O|       1|   79| 81|      null|
|Hellen|   B-NAME|       1|   83| 88|0.40429682|
| lives|        O|       1|   90| 94|      null|
|    in|        O|       1|   96| 97|      null|
| Paris|   B-CITY|       1|   99|103|0.49216735|
|     .|        O|       1|  104|104|      null|
|  John|   B-NAME|       0|    0|  3|0.14063153|
|    is|        O|       0|    5|  6|      null|
|     a|        O|       0|    8|  8|      null|
|   man|        O|       0|   10| 12|      null|
|   who|        O|       0|   14| 16|      null|
| works|        O|       0|   18| 22|      null|
|    in|        O|       0|   24| 25|      null|
|London|   B-CITY|       0|   27| 32|0.15521188|
|     ,|        O|       0|   33| 33|      null|
|London|   B-CITY|       0|   35| 40|0.12151082|
|   and|        O|       0|   42| 44|      null|
|London|   B-CITY|       0|   46| 51| 0.2650951|
|     .|        O|       0|   52| 52|      null|
+------+---------+--------+-----+---+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|zero_shot_ner_roberta|
|Compatibility:|Healthcare NLP 4.0.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|460.2 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

As it is a Zero-Shot NER, no training dataset is necessary.
