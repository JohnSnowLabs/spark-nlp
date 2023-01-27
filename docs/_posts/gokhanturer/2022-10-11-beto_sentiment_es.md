---
layout: model
title: Sentiment Analysis in Spanish
author: John Snow Labs
name: beto_sentiment
date: 2022-10-11
tags: [beto, sentiment, bert, es, open_source]
task: Text Classification
language: es
edition: Spark NLP 4.2.0
spark_version: [3.2, 3.0]
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Model trained with TASS 2020 corpus (around ~5k tweets) of several dialects of Spanish. Base model is BETO, a BERT model trained in Spanish.

## Predicted Entities

`POS`, `NEG`, `NEU`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/beto_sentiment_es_4.2.0_3.2_1665504729919.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/beto_sentiment_es_4.2.0_3.2_1665504729919.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier = BertForSequenceClassification.pretrained("beto_sentiment", "en")\
  .setInputCols(["document",'token'])\
  .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier   
])

# couple of simple examples
example = spark.createDataFrame([["Te quiero. Te amo."]]).toDF("text")

result = pipeline.fit(example).transform(example)

# result is a DataFrame
result.select("text", "class.result").show()
```

</div>

## Results

```bash
+------------------+------+
|              text|result|
+------------------+------+
|Te quiero. Te amo.| [POS]|
+------------------+------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|beto_sentiment|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|es|
|Size:|412.4 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

https://github.com/finiteautomata/pysentimiento/