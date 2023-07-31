---
layout: model
title: twitter_xlm_roberta_base_sentiment_pdc(cardiff)
author: veerdhwaj
name: twitter_xlm_roberta_base_sentiment_pdc
date: 2023-07-31
tags: [en, open_source, tensorflow]
task: Text Classification
language: en
edition: Spark NLP 5.0.0
spark_version: 3.2
supported: false
engine: tensorflow
annotator: XlmRoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Huggingface model: https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/veerdhwaj/twitter_xlm_roberta_base_sentiment_pdc_en_5.0.0_3.2_1690779049644.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://community.johnsnowlabs.com/veerdhwaj/twitter_xlm_roberta_base_sentiment_pdc_en_5.0.0_3.2_1690779049644.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from pyspark.ml import Pipeline

document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier = XlmRoBertaForSequenceClassification.pretrained('twitter_xlm_roberta_base_sentiment_pdc')\
  .setInputCols(["document",'token'])\
  .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

# couple of simple examples
example = spark.createDataFrame([['사랑해!'], ["T'estimo! ❤️"], ["I love you!"], ['Mahal kita!']]).toDF("text")

result = pipeline.fit(example).transform(example)

# result is a DataFrame
result.select("text", "class.result").show()
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|twitter_xlm_roberta_base_sentiment_pdc|
|Compatibility:|Spark NLP 5.0.0+|
|License:|Open Source|
|Edition:|Community|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|1.0 GB|
|Case sensitive:|true|
|Max sentence length:|512|