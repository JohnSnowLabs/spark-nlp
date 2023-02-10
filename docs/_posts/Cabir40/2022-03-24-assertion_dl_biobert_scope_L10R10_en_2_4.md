---
layout: model
title: Detect Assertion Status (assertion_dl_biobert_scope_L10R10)
author: John Snow Labs
name: assertion_dl_biobert_scope_L10R10
date: 2022-03-24
tags: [licensed, clinical, en, assertion, biobert]
task: Assertion Status
language: en
edition: Healthcare NLP 3.4.2
spark_version: 2.4
supported: true
annotator: AssertionDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This model is trained using `biobert_pubmed_base_cased` BERT token embeddings. It considers 10 tokens on the left and 10 tokens on the right side of the clinical entities extracted by NER models and assigns their assertion status based on their context in this scope.


## Predicted Entities


`present`, `absent`, `possible`, `conditional`, `associated_with_someone_else`, `hypothetical`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_dl_biobert_scope_L10R10_en_3.4.2_2.4_1648148217364.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/assertion_dl_biobert_scope_L10R10_en_3.4.2_2.4_1648148217364.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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


embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")


clinical_ner = MedicalNerModel.pretrained("ner_clinical_biobert", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")


ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")


clinical_assertion = AssertionDLModel.pretrained("assertion_dl_biobert_scope_L10R10","en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")
    
nlpPipeline = Pipeline(stages=[document,
                               sentenceDetector,
                               token, 
                               embeddings, 
                               clinical_ner,
                               ner_converter,  
                               clinical_assertion])


text = "Patient with severe fever and sore throat. He shows no stomach pain and he maintained on an epidural and PCA for pain control. He also became short of breath with climbing a flight of stairs. After CT, lung tumor located at the right lower lobe. Father with Alzheimer."


data = spark.createDataFrame([[text]]).toDF("text")


result = nlpPipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")


val sentenceDetector = SentenceDetectorDLModel.pretrained()
    .setInputCols(Array("document"))
    .setOutputCol("sentence") 


val tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")


val embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")


clinical_ner = MedicalNerModel.pretrained("ner_clinical_biobert", "en", "clinical/models") 
    .setInputCols(Array("sentence", "token", "embeddings")) 
    .setOutputCol("ner")


val ner_converter = new NerConverter()
    .setInputCols(Array("sentence","token","ner"))
    .setOutputCol("ner_chunk")


val clinical_assertion = AssertionDLModel.pretrained("assertion_dl_biobert_scope_L10R10","en", "clinical/models") 
    .setInputCols(Array("sentence", "ner_chunk", "embeddings")) 
    .setOutputCol("assertion")


val pipeline =  new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, embeddings, clinical_ner, ner_converter, clinical_assertion))


val data = Seq("Patient with severe fever and sore throat. He shows no stomach pain and he maintained on an epidural and PCA for pain control. He also became short of breath with climbing a flight of stairs. After CT, lung tumor located at the right lower lobe. Father with Alzheimer.").toDF("text")


val result = pipeline.fit(data).transform(data)
```
</div>


## Results


```bash
+---------------+---------+----------------------------+
|chunk          |ner_label|assertion                   |
+---------------+---------+----------------------------+
|severe fever   |PROBLEM  |present                     |
|sore throat    |PROBLEM  |present                     |
|stomach pain   |PROBLEM  |absent                      |
|an epidural    |TREATMENT|present                     |
|PCA            |TREATMENT|present                     |
|pain control   |TREATMENT|present                     |
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
|Model Name:|assertion_dl_biobert_scope_L10R10|
|Compatibility:|Healthcare NLP 3.4.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|en|
|Size:|3.2 MB|


## References


Trained on 2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text with `biobert_pubmed_base_cased`. https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/


## Benchmarking


```bash
label                         tp   fp   fn   prec       rec        f1       
absent                        839  89   44   0.9040948  0.9501699  0.9265599
present                       2436 127  168  0.9504487  0.9354839  0.9429069
conditional                   29   21   24   0.58       0.5471698  0.5631067
associated_with_someone_else  39   11   6    0.78       0.8666670  0.8210527
hypothetical                  164  44   11   0.7884616  0.9371429  0.8563969
possible                      126  36   75   0.7777778  0.6268657  0.6942149
Macro-average                 3633 328  328  0.7967971  0.8105832  0.8036310
Micro-average                 3633 328  328  0.9171926  0.9171926  0.9171926
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjgwMTU0NTg3LC02MTkwOTA1MDcsLTE4Nj
YzNjg0NTBdfQ==
-->