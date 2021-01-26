---
layout: model
title: Detect Assertion Status (assertion_dl_en)
author: John Snow Labs
name: assertion_dl_en
date: 2020-01-30
tags: [clinical, licensed, ner, en]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---
 
## Description

Deep learning named entity recognition model for assertions. The SparkNLP deep learning model (NerDL) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN.

{:.h2_title}
## Included Assertions
``hypothetical``, ``present``, ``absent``, ``possible``, ``conditional``, ``associated_with_someone_else``.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_dl_en_2.4.0_2.4_1580237286004.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel, NerConverter, AssertionDLModel.

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
clinical_assertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models") \
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
val clinical_assertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models")
    .setInputCols(Array("sentence", "ner_chunk", "embeddings"))
    .setOutputCol("assertion")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, clinical_assertion))

val result = pipeline.fit(Seq.empty["Patient has a headache for the last 2 weeks and appears anxious when she walks fast. No alopecia noted. She denies pain"].toDS.toDF("text")).transform(data)
```

</div>

{:.h2_title}
## Results
The output is a dataframe with a sentence per row and an ``"assertion"`` column containing all of the assertion labels in the sentence. The assertion column also contains assertion character indices, and other metadata. To get only the entity chunks and assertion labels, without the metadata, select ``"ner_chunk.result"`` and ``"assertion.result"`` from your output dataframe.

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
|Model Name:|assertion_dl_en|
|Type:|ner|
|Compatibility:|Spark NLP 2.4.0|
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

{:.h2_title}
## Benchmarking
```bash
|    | label                        | prec | rec  | f1   |
|---:|-----------------------------:|-----:|-----:|-----:|
| 0 | absent                        | 0.94 | 0.87 | 0.91 |
| 1 | associated_with_someone_else  | 0.81 | 0.73 | 0.76 |
| 2 | conditional                   | 0.78 | 0.24 | 0.37 |
| 3 | hypothetical                  | 0.89 | 0.75 | 0.81 |
| 4 | possible                      | 0.70 | 0.52 | 0.60 |
| 5 | present                       | 0.91 | 0.97 | 0.94 |
| 6 | Macro-average                 | 0.84 | 0.68 | 0.73 |
| 7 | Micro-average                 | 0.91 | 0.91 | 0.91 |
```