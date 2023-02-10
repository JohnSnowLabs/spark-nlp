---
layout: model
title: Detect Symptoms, Treatments and Other Entities in German
author: John Snow Labs
name: ner_healthcare
date: 2020-09-28
task: Named Entity Recognition
language: de
edition: Healthcare NLP 2.6.0
spark_version: 2.4
tags: [ner, clinical, de, licensed]
supported: true
annotator: MedicalNerModel
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---
 
## Description
This model can be used to detect symptoms, treatments and other entities in medical text in German language.

## Predicted Entities
`DIAGLAB_PROCEDURE`, `MEDICAL_SPECIFICATION`, `MEDICAL_DEVICE`, `MEASUREMENT`, `BIOLOGICAL_CHEMISTRY`, `BODY_FLUID`, `TIME_INFORMATION`, `LOCAL_SPECIFICATION`, `BIOLOGICAL_PARAMETER`, `PROCESS`, `MEDICATION`, `DOSING`, `DEGREE`, `MEDICAL_CONDITION`, `PERSON`, `TISSUE`, `STATE_OF_HEALTH`, `BODY_PART`, `TREATMENT`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_HEALTHCARE_DE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_HEALTHCARE_DE.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_healthcare_de_2.5.5_2.4_1599433028253.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_healthcare_de_2.5.5_2.4_1599433028253.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}
## How to use
Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}


```python
...
word_embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","de","clinical/models")\
   .setInputCols(["document","token"])\
   .setOutputCol("embeddings")
clinical_ner = NerDLModel.pretrained("ner_healthcare", "de", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
...
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, clinical_ner_converter])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
annotations = light_pipeline.fullAnnotate("Das Kleinzellige Bronchialkarzinom (Kleinzelliger Lungenkrebs, SCLC) ist ein hochmalignes bronchogenes Karzinom")

```

```scala
...
val word_embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","de","clinical/models")
   .setInputCols(Array("document","token"))
   .setOutputCol("embeddings")
val ner = NerDLModel.pretrained("ner_healthcare", "de", "clinical/models") 
  .setInputCols("sentence", "token", "embeddings") 
  .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, clinical_ner_converter))
val data = Seq("Das Kleinzellige Bronchialkarzinom (Kleinzelliger Lungenkrebs, SCLC) ist ein hochmalignes bronchogenes Karzinom").toDF("text")
val result = pipeline.fit(data).transform(data)

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
|Compatibility:|Healthcare NLP 2.6.0 +|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|[de]|
|Case sensitive:|false|
| Dependencies:  | w2v_cc_300d                           |

{:.h2_title}
## Data Source
Trained on 2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text with *w2v_cc_300d*.

{:.h2_title}
## Benchmarking
```bash
|    | label               |     tp |    fp |   fn | precision|    recall|       f1 |
|---:|--------------------:|-------:|------:|-----:|---------:|---------:|---------:|
|  0 | BIOLOGICAL_PARAMETER|    103 |    52 |   57 | 0.6645   | 0.6438   |  0.654   |
|  1 | BODY_FLUID          |    166 |    16 |   24 | 0.9121   | 0.8737   | 0.8925   |
|  2 | PERSON              |    475 |    74 |  142 | 0.8652   | 0.7699   | 0.8148   |
|  3 | DOSING              |     38 |    14 |   31 | 0.7308   | 0.5507   | 0.6281   |
|  4 | DIAGLAB_PROCEDURE   |    236 |    58 |   68 | 0.8027   | 0.7763   | 0.7893   |
|  5 | BODY_PART           |    690 |    72 |   79 | 0.9055   | 0.8973   | 0.9014   |
|  6 | MEDICATION          |    391 |   117 |  167 | 0.7697   | 0.7007   | 0.7336   |
|  7 | STATE_OF_HEALTH     |    321 |    41 |   76 | 0.8867   | 0.8086   | 0.8458   |
|  8 | LOCAL_SPECIFICATION |     57 |    19 |   24 |   0.75   | 0.7037   | 0.7261   |
|  9 | MEASUREMENT         |    574 |   260 |  222 | 0.6882   | 0.7211   | 0.7043   |
| 10 | TREATMENT           |    476 |   131 |  135 | 0.7842   | 0.7791   | 0.7816   |
| 11 | MEDICAL_CONDITION   |   1741 |   442 |  271 | 0.7975   | 0.8653   |   0.83   |
| 12 | TIME_INFORMATION    |    651 |   126 |  161 | 0.8378   | 0.8017   | 0.8194   |
| 13 | BIOLOGICAL_CHEMISTRY|    192 |    55 |   60 | 0.7773   | 0.7619   | 0.7695   |

```