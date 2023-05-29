---
layout: model
title: Catalan RobertaForSequenceClassification Base Cased model (from JonatanGk)
author: John Snow Labs
name: roberta_classifier_base_ca_finetuned_cyberbullying_catalan
date: 2022-09-09
tags: [ca, open_source, roberta, sequence_classification, classification]
task: Text Classification
language: ca
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `roberta-base-ca-finetuned-cyberbullying-catalan` is a Catalan model originally trained by `JonatanGk`.

## Predicted Entities

`Not_bullying`, `Bullying`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_base_ca_finetuned_cyberbullying_catalan_ca_4.1.0_3.0_1662765604401.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_classifier_base_ca_finetuned_cyberbullying_catalan_ca_4.1.0_3.0_1662765604401.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_base_ca_finetuned_cyberbullying_catalan","ca") \
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
 
val seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_base_ca_finetuned_cyberbullying_catalan","ca") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("ca.classify.roberta.base_finetuned").predict("""PUT YOUR STRING HERE""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_classifier_base_ca_finetuned_cyberbullying_catalan|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|ca|
|Size:|469.7 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/JonatanGk/roberta-base-ca-finetuned-cyberbullying-catalan
- https://colab.research.google.com/github/JonatanGk/Shared-Colab/blob/master/Cyberbullying_detection_(CATALAN).ipynb
- https://JonatanGk.github.io
- https://www.linkedin.com/in/JonatanGk/