---
layout: model
title: Detect Drugs and posology entities including experimental drugs and cycles (ner_posology_experimental)
author: John Snow Labs
name: ner_posology_experimental
date: 2021-09-01
tags: [licensed, clinical, en, ner]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.1.3
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This model detects drugs, experimental drugs, cyclelength, cyclecount, cycledaty, dosage, form, frequency, duration, route, and drug strength in text. It is based on the core `ner_posology` model, supports additional things like drug cycles, and enriched with more data from clinical trials.


## Predicted Entities


`Administration`, `Cyclenumber`, `Strength`, `Cycleday`, `Duration`, `Cyclecount`, `Route`, `Form`, `Frequency`, `Cyclelength`, `Drug`, `Dosage`


{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/demo){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_posology_experimental_en_3.1.3_3.0_1630511369574.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
         
sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_posology_experimental", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter()\
 	.setInputCols(["sentence", "token", "ner"])\
 	.setOutputCol("ner_chunk")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])

model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["Y-90 Humanized Anti-Tac: 10 mCi (if a bone marrow transplant was part of the patient's previous therapy) or 15 mCi of yttrium labeled anti-TAC; followed by calcium trisodium Inj (Ca DTPA)..\n\nCalcium-DTPA: Ca-DTPA will be administered intravenously on Days 1-3 to clear the radioactive agent from the body."]]).toDF("text"))
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
        
val sentence_detector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_posology_experimental", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
 	.setInputCols(Array("sentence", "token", "ner"))
 	.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter))

val data = Seq("""Y-90 Humanized Anti-Tac: 10 mCi (if a bone marrow transplant was part of the patient's previous therapy) or 15 mCi of yttrium labeled anti-TAC; followed by calcium trisodium Inj (Ca DTPA)..\n\nCalcium-DTPA: Ca-DTPA will be administered intravenously on Days 1-3 to clear the radioactive agent from the body.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.posology.experimental").predict("""Y-90 Humanized Anti-Tac: 10 mCi (if a bone marrow transplant was part of the patient's previous therapy) or 15 mCi of yttrium labeled anti-TAC; followed by calcium trisodium Inj (Ca DTPA)..\n\nCalcium-DTPA: Ca-DTPA will be administered intravenously on Days 1-3 to clear the radioactive agent from the body.""")
```

</div>


## Results


```bash
|    | chunk                    |   begin |   end | entity   |
|---:|:-------------------------|--------:|------:|:---------|
|  0 | Anti-Tac                 |      15 |    22 | Drug     |
|  1 | 10 mCi                   |      25 |    30 | Dosage   |
|  2 | 15 mCi                   |     108 |   113 | Dosage   |
|  3 | yttrium labeled anti-TAC |     118 |   141 | Drug     |
|  4 | calcium trisodium Inj    |     156 |   176 | Drug     |
|  5 | Calcium-DTPA             |     191 |   202 | Drug     |
|  6 | Ca-DTPA                  |     205 |   211 | Drug     |
|  7 | intravenously            |     234 |   246 | Route    |
|  8 | Days 1-3                 |     251 |   258 | Cycleday |
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_posology_experimental|
|Compatibility:|Healthcare NLP 3.1.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|


## Data Source


This model is trained on FDA 2018 Medication dataset, enriched with clinical trials data.


## Benchmarking


```bash
label	               tp       fp	   fn	 prec	       rec	       f1
B-Drug               30260    1321  1630  0.95817107  0.9488868    0.95350635
B-Cycleday	         294      1     7     0.99661016  0.9767442    0.9865772
B-Dosage             4019     441   972   0.9011211   0.8052494    0.85049194
I-Strength	         21784    2375  1616  0.9016929   0.9309401    0.9160832
I-Cyclenumber        113      2     1     0.9826087   0.9912280    0.98689955
B-Cyclelength        217      3     0     0.98636365  1.0          0.99313504
B-Administration     97       1     5     0.9897959   0.95098037   0.96999997
I-Cyclecount         174      7     3     0.96132594  0.9830508    0.972067
B-Strength	         18871    1299  1161  0.9355974   0.9420427    0.93880904
B-Frequency	         13064    464   713   0.96570075  0.9482471    0.95689434
B-Cyclenumber        93       2     1     0.97894734  0.9893617    0.9841269
I-Duration	         6116     519   738   0.92177844  0.89232564   0.9068129
B-Cyclecount         120      5     3     0.96        0.9756098    0.9677419
B-Form               10964    912   986   0.92320645  0.9174895    0.9203391
I-Route              275      42    51    0.8675079   0.8435583    0.85536546
I-Cyclelength        261      5     0     0.981203    1.0	       0.9905123
I-Dosage             2385     471   1107  0.835084    0.6829897    0.75141776
I-Cycleday	         548      5     13    0.9909584   0.9768271    0.983842
I-Frequency	         18644    967   1574  0.9506909   0.9221486    0.9362023
I-Administration     303      10    5     0.9680511   0.98376626   0.9758454
I-Form               642      284   553   0.6933045   0.5372385    0.6053748
B-Route              5930     280   692   0.9549114   0.8954998    0.92425185
B-Duration	         2422     261   359   0.9027208   0.87090975   0.88653
I-Drug               11472    1066  1240  0.9149784   0.9024544    0.9086733
Macro-average        149068   10743 13430 0.93426394  0.9111479    0.92256117
Micro-average        149068   10743 13430 0.93277687  0.91735286   0.9250006
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjA5NjIwMzY0MSw0NzE5NDcyNzhdfQ==
-->