---
layout: model
title: Detect Assertion Status (assertion_dl_biobert) - supports confidence scores
author: John Snow Labs
name: assertion_dl_biobert
date: 2021-01-26
task: Assertion Status
language: en
edition: Healthcare NLP 2.7.2
spark_version: 2.4
tags: [assertion, en, licensed, clinical, biobert]
supported: true
annotator: AssertionDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Assign assertion status to clinical entities extracted by NER based on their context in the text.

## Predicted Entities

`absent`, `present`, `conditional`, `associated_with_someone_else`, `hypothetical`, `possible`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ASSERTION/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_dl_biobert_en_2.7.2_2.4_1611647486798.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/assertion_dl_biobert_en_2.7.2_2.4_1611647486798.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

word_embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased")\
    .setInputCols(["sentence",'token'])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_clinical_biobert", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")

clinical_assertion = AssertionDLModel.pretrained("assertion_dl_biobert", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")

nlpPipeline = Pipeline(stages=[
      documentAssembler, 
      sentenceDetector,
      tokenizer,
      word_embeddings,
      clinical_ner,
      ner_converter,
      clinical_assertion
])

data = spark.createDataFrame([["""Patient has a headache for the last 2 weeks, needs to get a head CT, and appears anxious when she walks fast. No alopecia noted. She denies pain"""]]).toDF("text")

result = nlpPipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
    .setInputCols("document") 
    .setOutputCol("sentence") 

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val word_embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_clinical_biobert", "en", "clinical/models") 
    .setInputCols(Array("sentence", "token", "embeddings")) 
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence","token","ner"))
    .setOutputCol("ner_chunk")

val clinical_assertion = AssertionDLModel.pretrained("assertion_dl_biobert","en", "clinical/models") 
    .setInputCols(Array("sentence", "ner_chunk", "embeddings")) 
    .setOutputCol("assertion")

val pipeline =  new Pipeline().setStages(Array(documentAssembler, 
                                               sentenceDetector, 
                                               tokenizer, 
                                               word_embeddings, 
                                               clinical_ner, 
                                               ner_converter, 
                                               clinical_assertion))

val data = Seq("""Patient has a headache for the last 2 weeks, needs to get a head CT, and appears anxious when she walks fast. No alopecia noted. She denies pain""").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.assert.biobert").predict("""Patient has a headache for the last 2 weeks, needs to get a head CT, and appears anxious when she walks fast. No alopecia noted. She denies pain.""")
```

</div>

## Results

```bash
+----------+---------+-----------+
|chunk     |ner_label|assertion  |
+----------+---------+-----------+
|a headache|PROBLEM  |present    |
|a head CT |TEST     |present    |
|anxious   |PROBLEM  |conditional|
|alopecia  |PROBLEM  |absent     |
|pain      |PROBLEM  |absent     |
+----------+---------+-----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|assertion_dl_biobert|
|Compatibility:|Spark NLP 2.7.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|en|

## Data Source

Trained on i2b2 assertion data.

## Benchmarking

```bash
label                            tp    fp    fn      prec       rec        f1
absent                          769    51    57  0.937805  0.930993  0.934386
present                        2575   161   102  0.941155  0.961898  0.951413
conditional                      20    14    23  0.588235  0.465116  0.519481
associated_with_someone_else     51     9    15  0.85      0.772727  0.809524
hypothetical                    129    13    15  0.908451  0.895833  0.902098
possible                        114    44    80  0.721519  0.587629  0.647727
Macro-average                  3658   292   292  0.824527  0.769033  0.795814
Micro-average                  3658   292   292  0.926076  0.926076  0.926076
```