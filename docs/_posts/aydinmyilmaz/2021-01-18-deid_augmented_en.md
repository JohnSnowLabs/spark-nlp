---
layout: model
title: Deidentify Augmented
author: John Snow Labs
name: deid_augmented
date: 2021-01-18
tags: [deidentify, en, clinical, licensed]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Deidentify Augmented is a deidentification model. It identifies instances of protected health information in text documents, and it can either obfuscate them (e.g., replacing names with different, fake names) or mask them (e.g., replacing “2020-06-04” with “<DATE>”). This model is useful for maintaining HIPAA compliance when dealing with text documents that contain protected health information.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentificiation.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/deid_augmented_en_2.7.1_2.4_1611002766883.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
nlp_pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter])

result = nlp_pipeline.transform(spark.createDataFrame(pd.DataFrame({'text': ["""Patient AIQING, 25 month years-old , born in Beijing, was transfered to the The Johns Hopkins Hospital. Phone number: (541) 754-3010. MSW 100009632582 for his colonic polyps. He wants to know the results from them. He is not taking hydrochlorothiazide and is curious about his blood pressure. He said he has cut his alcohol back to 6 pack once a week. He has cut back his cigarettes to one time per week. P:   Follow up with Dr. Hobbs in 3 months. Gilbert P. Perez, M.D."""]})))

obfuscation = DeIdentificationModel.pretrained("deid_augmented, "en", "clinical/models") \
      .setInputCols(["sentence", "token", "ner_chunk"]) \
      .setOutputCol("obfuscated") \
      .setMode("obfuscate")

deid_text = obfuscation.transform(result)
```


</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deid_augmented|
|Type:|ner|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

The model was trained based on data from https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/