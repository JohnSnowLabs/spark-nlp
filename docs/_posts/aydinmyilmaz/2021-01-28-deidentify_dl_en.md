---
layout: model
title: Deidentify PHI (DL)
author: John Snow Labs
name: deidentify_dl
date: 2021-01-28
task: De-identification
language: en
edition: Healthcare NLP 2.7.2
spark_version: 2.4
tags: [en, deidentify, clinical, licensed]
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Deidentify (DL) is a deidentification model. It identifies instances of protected health information in text documents, and it can either obfuscate them (e.g., replacing names with different, fake names) or mask them (e.g., replacing “2020-06-04” with “<DATE>”). This model is useful for maintaining HIPAA compliance when dealing with text documents that contain protected health information.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.1.Pretrained_Clinical_DeIdentificiation.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/deidentify_dl_en_2.7.2_2.4_1611831975581.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/deidentify_dl_en_2.7.2_2.4_1611831975581.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

Use as a downstream task of a pipeline including one of Spark NLP  deidentification ner models (i.e. `ner_deid_large`)

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
deid_ner = NerDLModel.pretrained("ner_deid_large", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

nlp_pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, word_embeddings, deid_ner, ner_converter]) 

result = nlp_pipeline.transform(spark.createDataFrame([['Patient AIQING, 25 month years-old , born in Beijing, was transfered to the The Johns Hopkins Hospital. Phone number: (541) 754-3010. MSW 100009632582 for his colonic polyps. He wants to know the results from them. He is not taking hydrochlorothiazide and is curious about his blood pressure. He said he has cut his alcohol back to 6 pack once a week. He has cut back his cigarettes to one time per week. P:   Follow up with Dr. Hobbs in 3 months. Gilbert P. Perez, M.D.']], ["text"]))

deid= DeIdentificationModel.pretrained("deidentify_dl", "en", "clinical/models") \
      .setInputCols(["sentence", "token", "ner_chunk"]) \
      .setOutputCol("obfuscated") \
      .setMode("obfuscate")

deid_text = deid.transform(result)
```

```scala
...
val deid_ner = NerDLModel.pretrained("ner_deid_large", "en", "clinical/models")
  .setInputCols(Array("sentence", "token", "embeddings"))
  .setOutputCol("ner")

val nlpPipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, word_embeddings, deid_ner, ner_converter))

val result = pipeline.fit(Seq.empty[String]).transform(data)

val results = LightPipeline(model).fullAnnotate("""Patient AIQING, 25 month years-old , born in Beijing, was transfered to the The Johns Hopkins Hospital. Phone number: (541) 754-3010. MSW 100009632582 for his colonic polyps. He wants to know the results from them. He is not taking hydrochlorothiazide and is curious about his blood pressure. He said he has cut his alcohol back to 6 pack once a week. He has cut back his cigarettes to one time per week. P:   Follow up with Dr. Hobbs in 3 months. Gilbert P. Perez, M.D.""")

val deid= DeIdentificationModel.pretrained("deidentify_dl", "en", "clinical/models")
      .setInputCols(Array("sentence", "token", "ner_chunk"))
      .setOutputCol("obfuscated")
      .setMode("obfuscate")

val deid_text = deid.transform(result)
```
</div>

## Results

```bash
|   |                                          sentence |                                      deidentified |
|--:|--------------------------------------------------:|--------------------------------------------------:|
| 0 | Patient AIQING, 25 month years-old , born in B... | Patient CAM, <AGE> month years-old , born in M... |
| 1 |                     Phone number: (541) 754-3010. |                      Phone number: (603)531-7148. |
| 2 |          MSW 100009632582 for his colonic polyps. |                  MSW <ID> for his colonic polyps. |
| 3 |           He wants to know the results from them. |           He wants to know the results from them. |
| 4 | He is not taking hydrochlorothiazide and is cu... | He is not taking hydrochlorothiazide and is cu... |
| 5 | He said he has cut his alcohol back to 6 pack ... | He said he has cut his alcohol back to 6 pack ... |
| 6 | He \nhas cut back his cigarettes to one time p... | He \nhas cut back his cigarettes to one time p... |
| 7 |          P: Follow up with Dr. Hobbs in 3 months. |        P: Follow up with Dr. RODOLPH in 3 months. |
| 8 |                            Gilbert P. Perez, M.D. |                                      Gertie, M.D. |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deidentify_dl|
|Compatibility:|Spark NLP 2.7.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk, token, document]|
|Output Labels:|[dei]|
|Language:|en|
|Dependencies:|embeddings_clinical|

## Data Source

The model was trained based on data from https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/