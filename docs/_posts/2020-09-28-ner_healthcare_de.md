---
layout: model
title: Detect symptoms, treatments and other NERs in German
author: John Snow Labs
name: ner_healthcare
date: 2020-09-28
tags: [ner, de, licensed]
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This model can be used to detect symptoms, treatments and other entities in medical text in German language.

## Predicted Entities
DIAGLAB_PROCEDURE, MEDICAL_SPECIFICATION, MEDICAL_DEVICE, MEASUREMENT, BIOLOGICAL_CHEMISTRY, BODY_FLUID, TIME_INFORMATION, LOCAL_SPECIFICATION, BIOLOGICAL_PARAMETER, PROCESS, MEDICATION, DOSING, DEGREE, MEDICAL_CONDITION, PERSON, TISSUE, STATE_OF_HEALTH, BODY_PART, TREATMENT

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_HEALTHCARE_DE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_HEALTHCARE_DE.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_healthcare_de_2.5.5_2.4_1599433028253.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
## How to use
Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}


```python

clinical_ner = NerDLModel.pretrained("ner_healthcare", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate("Das Kleinzellige Bronchialkarzinom (Kleinzelliger Lungenkrebs, SCLC) ist ein hochmalignes bronchogenes Karzinom")

```

</div>

{:.h2_title}
## Results

```bash
+----+-------------------+---------+---------+--------------------------+
|    | chunk             |   begin |   end   | entity                   |
+====+===================+=========+=========+==========================+
|  0 | Kleinzellige      |      4  |    15   | MEDICAL_SPECIFICATION    |
+----+-------------------+---------+---------+--------------------------+
|  1 | Bronchialkarzinom |      17 |   33    | MEDICAL_CONDITION        |
+----+-------------------+---------+---------+--------------------------+
|  2 | Kleinzelliger     |      36 |    48   | MEDICAL_SPECIFICATION    |
+----+-------------------+---------+---------+--------------------------+
|  3 | Lungenkrebs       |      50 |   60    | MEDICAL_CONDITION        |
+----+-------------------+---------+---------+--------------------------+
|  4 | SCLC              |      63 |   66    | MEDICAL_CONDITION        |
+----+-------------------+---------+---------+--------------------------+
|  5 | hochmalignes      |      77 |    88   | MEASUREMENT              |
+----+-------------------+---------+---------+--------------------------+
|  6 | bronchogenes      |      90 |   101   | BODY_PART                |
+----+-------------------+---------+---------+--------------------------+
|  7 | Karzinom          |     103 |   110   | MEDICAL_CONDITION        |
+----+-------------------+---------+---------+--------------------------+
```
{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_healthcare|
|Type:|ner|
|Compatibility:|Spark NLP for Healthcare 2.6.0 +|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|[de]|
|Case sensitive:|false|

