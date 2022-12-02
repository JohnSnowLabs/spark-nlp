---
layout: model
title: Deidentify PHI (Large)
author: John Snow Labs
name: deidentify_large
language: en
repository: clinical/models
date: 2020-08-04
task: De-identification
edition: Healthcare NLP 2.5.5
spark_version: 2.4
tags: [deidentify, en, clinical, licensed]
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Deidentify (Large) is a deidentification model. It identifies instances of protected health information in text documents, and it can either obfuscate them (e.g., replacing names with different, fake names) or mask them (e.g., replacing "2020-06-04" with "&lt;DATE&gt;"). This model is useful for maintaining HIPAA compliance when dealing with text documents that contain protected health information.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/DEID_PHI_TEXT){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/DEID_PHI_TEXT.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/nerdl_deid_en_1.8.0_2.4_1545462443516.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
nlp_pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter])
result = nlp_pipeline.transform(spark.createDataFrame([['Patient AIQING, 25 month years-old , born in Beijing, was transfered to the The Johns Hopkins Hospital. Phone number: (541) 754-3010. MSW 100009632582 for his colonic polyps. He wants to know the results from them. He is not taking hydrochlorothiazide and is curious about his blood pressure. He said he has cut his alcohol back to 6 pack once a week. He has cut back his cigarettes to one time per week. P:   Follow up with Dr. Hobbs in 3 months. Gilbert P. Perez, M.D.']], ["text"]))

obfuscation = DeIdentificationModel.pretrained("deidentify_large", "en", "clinical/models") \
.setInputCols(["sentence", "token", "ner_chunk"]) \
.setOutputCol("obfuscated") \
.setMode("obfuscate")

deid_text = obfuscation.transform(result)
```

```scala
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))
val data = Seq("Patient AIQING, 25 month years-old , born in Beijing, was transfered to the The Johns Hopkins Hospital. Phone number: (541) 754-3010. MSW 100009632582 for his colonic polyps. He wants to know the results from them. He is not taking hydrochlorothiazide and is curious about his blood pressure. He said he has cut his alcohol back to 6 pack once a week. He has cut back his cigarettes to one time per week. P:   Follow up with Dr. Hobbs in 3 months. Gilbert P. Perez, M.D.").toDF("text")
val result = pipeline.fit(data).transform(data)

val deid = DeIdentificationModel.pretrained("deidentify_large", "en", "clinical/models")
.setInputCols(Array("sentence", "token", "ner_chunk"))
.setOutputCol("obfuscated")
.setMode("obfuscate")

val deid_text = new deid.transform(result)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.de_identify.large").predict("""Patient AIQING, 25 month years-old , born in Beijing, was transfered to the The Johns Hopkins Hospital. Phone number: (541) 754-3010. MSW 100009632582 for his colonic polyps. He wants to know the results from them. He is not taking hydrochlorothiazide and is curious about his blood pressure. He said he has cut his alcohol back to 6 pack once a week. He has cut back his cigarettes to one time per week. P:   Follow up with Dr. Hobbs in 3 months. Gilbert P. Perez, M.D.""")
```

</div>

{:.h2_title}
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
|Model Name:|deidentify_large|
|Type:|deid|
|Compatibility:| Healthcare NLP 2.5.5|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, ner_chunk]|
|Output Labels:|[obfuscated]|
|Language:|en|
|Case sensitive:|false|


{:.h2_title}
## Data Source
The model was trained based on data from [https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/](https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/)