---
layout: model
title: Detect entities in radiology reports (ner_clinical_icdem)
author: John Snow Labs
name: ner_clinical_icdem
date: 2021-04-01
tags: [ner, clinical, licensed, en]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 3.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Extract entities related to pneumonia in radiology reports

## Predicted Entities

`Size`, `Positive1`, `Tests`, `Procedure1`, `Type`, `Extension`, `Bronchial1`, `Vascular1`, `Examined`, `Vascular`, `Laterality1`, `Resuts`, `Grade`, `Examined1`, `Laterality`, `Size1`, `pM`, `pN1`, `Parenchymal`, `Results`, `Resuts1`, `Localization`, `Tests1`, `""`, `OtherMargin`, `DcisMargin`, `Margins`, `Margins1`, `pT1`, `Diagnosis3`, `Localization1`, `Parenchymal1`, `Diagnosis10`, `Size2`, `Nuclear`, `Bronchial`, `Grade1`, `Procedure`, `Focality`, `Nuclear1`, `Localization2`, `pT`, `Results1`, `Positive`, `Type1`, `pN`

{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/demo){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_clinical_icdem_en_3.0.0_3.0_1617260629534.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

...
embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")  .setInputCols(["sentence", "token"])  .setOutputCol("embeddings")
clinical_ner = MedicalNerModel.pretrained("ner_clinical_icdem", "en", "clinical/models")   .setInputCols(["sentence", "token", "embeddings"])   .setOutputCol("ner")
...
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical, clinical_ner, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
results = model.transform(spark.createDataFrame([["EXAMPLE_TEXT"]]).toDF("text"))
```
```scala

...
val embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")
val ner = MedicalNerModel.pretrained("ner_clinical_icdem", "en", "clinical/models")
  .setInputCols(Array("sentence", "token", "embeddings"))
  .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))
val result = pipeline.fit(Seq.empty[String]).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_clinical_icdem|
|Compatibility:|Spark NLP for Healthcare 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, word_vecs]|
|Output Labels:|[ner]|
|Language:|en|