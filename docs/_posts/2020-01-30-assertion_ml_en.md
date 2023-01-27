---
layout: model
title: Detect Assertion Status (assertion_ml_en)
author: John Snow Labs
name: assertion_ml_en
date: 2020-01-30
task: Assertion Status
language: en
edition: Healthcare NLP 2.4.0
spark_version: 2.4
tags: [clinical, licensed, ner, en]
supported: true
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
 
Logistic regression based named entity recognition model for assertions. 

## Predicted Labels

 Hypothetical, Present, Absent, Possible, Conditional, Associated_with_someone_else 

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_dl_large_en_2.5.0_2.4_1590022282256.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/assertion_dl_large_en_2.5.0_2.4_1590022282256.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel, NerConverter, AssertionLogRegModel.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}


```python
...
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")
clinical_ner = NerDLModel.pretrained("ner_clinical", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
ner_converter = NerConverter() \
  .setInputCols(["sentence", "token", "ner"]) \
  .setOutputCol("ner_chunk")
clinical_assertion = AssertionDLModel.pretrained("assertion_ml", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")
    
nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, clinical_assertion])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

light_result = LightPipeline(model).fullAnnotate('Patient has a headache for the last 2 weeks and appears anxious when she walks fast. No alopecia noted. She denies pain')[0]

```

```scala
...
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")
val clinical_ner = NerDLModel.pretrained("ner_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token", "embeddings")) 
  .setOutputCol("ner")
val ner_converter = NerConverter()
  .setInputCols(Array("sentence", "token", "ner"))
  .setOutputCol("ner_chunk")
val clinical_assertion_ml = AssertionLogRegModel.pretrained("assertion_ml", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, clinical_assertion_ml))
val data = Seq("Patient has a headache for the last 2 weeks and appears anxious when she walks fast. No alopecia noted. She denies pain").toDF("text")
val result = pipeline.fit(data).transform(data)
```

</div>
{:.h2_title}
## Results
The output is a dataframe with a sentence per row and an "assertion" column containing all of the assertion labels in the sentence. The assertion column also contains assertion character indices, and other metadata. To get only the entity chunks and assertion labels, without the metadata, select "ner_chunk.result" and "assertion.result" from your output dataframe.

```bash
|   | chunks     | entities | assertion   |
|---|------------|----------|-------------|
| 0 | a headache | PROBLEM  | present     |
| 1 | anxious    | PROBLEM  | conditional |
| 2 | alopecia   | PROBLEM  | absent      |
| 3 | pain       | PROBLEM  | absent      |
```


{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|assertion_ml_en_2.4.0_2.4|
|Type:|ner|
|Compatibility:|Spark NLP 2.4.0+|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence, ner_chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|[en]|
|Case sensitive:|false|

{:.h2_title}
## Data Source
Trained on 2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text with 'embeddings_clinical'.
https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/