---
layout: model
title: Detect financial entities
author: John Snow Labs
name: ner_financial_contract
date: 2021-04-01
tags: [ner, clinical, licensed, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.0.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Extract key entities in financial contracts using pretrained NER model.

## Predicted Entities

`ORG`, `PER`, `MISC`, `LOC`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_financial_contract_en_3.0.0_3.0_1617260833629.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_financial_contract_en_3.0.0_3.0_1617260833629.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python

document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("glove_6B_300", "xx")\
    .setInputCols("sentence", "token")\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_financial_contract", "en", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[document_assembler, sentenceDetector, tokenizer, embeddings, clinical_ner, ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

text = """Hans is a professor at the Norwegian University of Copenhagen, and he is a true Copenhagener."""

result = model.transform(spark.createDataFrame([[text]]).toDF("text"))

```
```scala

document_assembler = new DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
    .setInputCols("document")\
    .setOutputCol("sentence")

tokenizer = new Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("glove_6B_300", "xx")\
    .setInputCols(Array("sentence", "token"))\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_financial_contract", "en", "clinical/models")\
    .setInputCols(Array("sentence", "token", "embeddings"))\
    .setOutputCol("ner")

ner_converter = new NerConverter()\
    .setInputCols(Array("sentence", "token", "ner"))\
    .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentenceDetector, tokenizer, embeddings, clinical_ner, ner_converter))

val data = Seq("""Hans is a professor at the Norwegian University of Copenhagen, and he is a true Copenhagener.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.financial_contract").predict("""Put your text here.""")
```

</div>

## Results
```bash
+--------------------+---------+
|chunk               |ner_label|
+--------------------+---------+
|professor           |PER      |
|Norwegian University|PER      |
|Copenhagen          |LOC      |
+--------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_financial_contract|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|