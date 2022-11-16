---
layout: model
title: Detect Assertion Status (assertion_dl_scope_L10R10)
author: John Snow Labs
name: assertion_dl_scope_L10R10
date: 2022-03-17
tags: [clinical, licensed, en, assertion]
task: Assertion Status
language: en
edition: Healthcare NLP 3.4.2
spark_version: 3.0
supported: true
annotator: AssertionDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This model considers 10 tokens on the left and 10 tokens on the right side of the clinical entities extracted by NER models and assigns their assertion status based on their context in this scope.


## Predicted Entities


`hypothetical`, `associated_with_someone_else`, `conditional`, `possible`, `absent`, `present`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_dl_scope_L10R10_en_3.4.2_3.0_1647494736416.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document = DocumentAssembler()\
   .setInputCol("text")\
   .setOutputCol("document")
   
sentenceDetector = SentenceDetector()\
  .setInputCols(["document"])\
  .setOutputCol("sentence")
  
token = Tokenizer()\
  .setInputCols(['sentence'])\
  .setOutputCol('token')
  
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")
  
clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
  
ner_converter = NerConverter() \
  .setInputCols(["sentence", "token", "ner"]) \
  .setOutputCol("ner_chunk")
  
clinical_assertion = AssertionDLModel.pretrained("assertion_dl_scope_L10R10", "en", "clinical/models") \
  .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
  .setOutputCol("assertion")
  
nlpPipeline = Pipeline(stages=[document,sentenceDetector, token, word_embeddings,clinical_ner,ner_converter,  clinical_assertion])


text = "Patient with severe fever and sore throat. He shows no stomach pain and he maintained on an epidural and PCA for pain control. He also became short of breath with climbing a flight of stairs. After CT, lung tumor located at the right lower lobe. Father with Alzheimer."


data = spark.createDataFrame([[text]]).toDF("text")
result = nlpPipeline.fit(data).transform(data)






```
```scala
val documentAssembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")
    
val sentenceDetector = new SentenceDetector()
  .setInputCols(Array("document"))
  .setOutputCol("sentence")
  
val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")
  
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")
  
val clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token", "embeddings")) 
  .setOutputCol("ner")
  
val ner_converter = NerConverter()
  .setInputCols(Array("sentence", "token", "ner"))
  .setOutputCol("ner_chunk")
  
val clinical_assertion = AssertionDLModel.pretrained("assertion_dl_scope_L10R10", "en", "clinical/models")
  .setInputCols(Array("sentence", "ner_chunk", "embeddings"))
  .setOutputCol("assertion")
  
val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, clinical_assertion))
val data = Seq("Patient with severe fever and sore throat. He shows no stomach pain and he maintained on an epidural and PCA for pain control. He also became short of breath with climbing a flight of stairs. After CT, lung tumor located at the right lower lobe. Father with Alzheimer.").toDF("text")


val result = pipeline.fit(data).transform(data)
```
</div>


## Results


```bash
+---------------+---------+----------------------------+
|chunk          |entity   |assertion                   |
+---------------+---------+----------------------------+
|severe fever   |PROBLEM  |present                     |
|sore throat    |PROBLEM  |present                     |
|stomach pain   |PROBLEM  |absent                      |
|an epidural    |TREATMENT|present                     |
|PCA            |TREATMENT|present                     |
|pain control   |PROBLEM  |present                     |
|short of breath|PROBLEM  |conditional                 |
|CT             |TEST     |present                     |
|lung tumor     |PROBLEM  |present                     |
|Alzheimer      |PROBLEM  |associated_with_someone_else|
+---------------+---------+----------------------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|assertion_dl_scope_L10R10|
|Compatibility:|Healthcare NLP 3.4.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|en|
|Size:|1.4 MB|


## References


Trained on 2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text with ‘embeddings_clinical’. https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/


## Benchmarking


```bash
label                         tp    fp   fn   prec        rec        f1        
absent                        812   48   71   0.94418603  0.9195923  0.93172693
present                       2463  127  141  0.9509652   0.9458525  0.948402  
conditional                   25    19   28   0.5681818   0.4716981  0.5154639 
associated_with_someone_else  36    7    9    0.8372093   0.8        0.8181818 
hypothetical                  147   31   28   0.8258427   0.84       0.8328612 
possible                      159   87   42   0.64634144  0.7910448  0.71140933
Macro-average	              -     -    -    0.79545444  0.7946979  0.795076  
Micro-average	              -     -    -    0.91946477  0.9194648  0.91946477
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTA5MjcyMTIwMCwtNjY5NTk1MjUwXX0=
-->