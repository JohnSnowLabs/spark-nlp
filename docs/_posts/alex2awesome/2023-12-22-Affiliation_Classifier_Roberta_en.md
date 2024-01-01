---
layout: model
title: Affiliation Classifier
author: alex2awesome
name: Affiliation_Classifier_Roberta
date: 2023-12-22
tags: [en, open_source, tensorflow]
task: Text Classification
language: en
edition: Spark NLP 5.2.0
spark_version: 3.2
supported: false
engine: tensorflow
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Predicts the affiliation, if any, of the information in a paragraph.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/alex2awesome/Affiliation_Classifier_Roberta_en_5.2.0_3.2_1703264189300.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://community.johnsnowlabs.com/alex2awesome/Affiliation_Classifier_Roberta_en_5.2.0_3.2_1703264189300.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.annotator import *
from sparknlp.base import *

document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequence_classifier = RoBertaForSequenceClassification.load(MODEL_NAME)
  .setInputCols(["document",'token'])\
  .setOutputCol("class")  

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequence_classifier
])

# couple of simple examples
example = spark.createDataFrame([["I love you!"], ['I feel lucky to be here.']]).toDF("text")

result = pipeline.fit(example).transform(example)

# result is a DataFrame
result.select("text", "class.result").show()
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|Affiliation_Classifier_Roberta|
|Compatibility:|Spark NLP 5.2.0+|
|License:|Open Source|
|Edition:|Community|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|441.4 MB|
|Case sensitive:|true|
|Max sentence length:|128|
|Dependencies:|None|