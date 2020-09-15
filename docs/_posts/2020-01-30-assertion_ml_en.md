---
layout: model
title: Assertion ML
author: John Snow Labs
name: assertion_ml_en
date: 2020-01-30
tags: [licensed, ner, en]
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Logistic regression based named entity recognition model for assertions. 

## Included Assertions

 Hypothetical, Present, Absent, Possible, Conditional, Associated_with_someone_else 

{:.btn-box}
[Live Demo](){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_dl_large_en_2.5.0_2.4_1590022282256.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel, NerConverter, AssertionLogRegModel.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}


```python

clinical_assertion_ml = AssertionLogRegModel.pretrained("assertion_ml", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")
    
nlpPipeline = Pipeline(stages=[clinical_assertion_ml])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

```

```scala

val clinical_assertion_ml = AssertionLogRegModel.pretrained("assertion_ml", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")

val pipeline = new Pipeline().setStages(Array(clinical_assertion_ml))

val result = pipeline.fit(Seq.empty[String].toDS.toDF("text")).transform(data)
```

</div>

{:.model-param}
## Model Parameters

{:.table-model}
|---|---|
|Model Name:|assertion_ml_en_2.4.0_2.4|
|Type:|ner|
|Compatibility:|Spark NLP 2.4.0|
|Edition:|Healthcare|
|License:|Licensed|
|Input Labels:|[sentence, ner_chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|[en]|
|Case sensitive:|false|

{:.h2_title}
## Dataset used for training
Trained on 2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text with 'embeddings_clinical'.
https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

{:.h2_title}
## Results
The output is a dataframe with a sentence per row and an "assertion" column containing all of the assertion labels in the sentence. The assertion column also contains assertion character indices, and other metadata. To get only the entity chunks and assertion labels, without the metadata, select "ner_chunk.result" and "assertion.result" from your output dataframe.

![image](/assets/images/assertiondl.png) 
