---
layout: model
title: LeNER-Br Named Entity Recognition for (Brazilian) Portuguese Legal Text (Bert Base Cased)
author: John Snow Labs
name: lener_bert_base
date: 2021-07-28
tags: [pt, ner, lener, licensed, br]
task: Named Entity Recognition
language: pt
edition: Spark NLP for Healthcare 3.1.3
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This model annotates named entities in a (Brazilian) legal text, that can be used to find features such as names of people, organizations, and jurisprudence. The model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together.


This model uses the pre-trained `bert_portuguese_base_cased` embeddings model from `BertEmbeddings` annotator as an input, so be sure to use the same embeddings in the pipeline.


## Predicted Entities


- `ORGANIZACAO` (Organizations)
- `JURISPRUDENCIA` (Jurisprudence)
- `PESSOA` (Person)
- `TEMPO` (Time)
- `LOCAL` (Location)
- `LEGISLACAO` (Laws)
- `O` (Other)


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_LENER/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_LEGAL_PT.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/lener_bert_base_pt_3.1.3_3.0_1627483199555.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

embeddings = BertEmbeddings.pretrained("bert_portuguese_base_cased", "pt")\
    .setInputCols("document", "token") \
    .setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained('lener_bert_base', 'pt', 'clinical/models') \
    .setInputCols(['document', 'token', 'embeddings']) \
    .setOutputCol('ner')

ner_converter = NerConverter() \
    .setInputCols(['document', 'token', 'ner']) \
    .setOutputCol('ner_chunk')

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    embeddings,
    ner_model,
    ner_converter
])

example = spark.createDataFrame(pd.DataFrame({'text': ["""diante do exposto , com fundamento nos artigos 32 , i , e 33 , da lei 8.443/1992 , submetem-se os autos à consideração superior , com posterior encaminhamento ao ministério público junto ao tcu e ao gabinete do relator , propondo : a ) conhecer do recurso e , no mérito , negar-lhe provimento ; b ) comunicar ao recorrente , ao superior tribunal militar e ao tribunal regional federal da 2ª região , a fim de fornecer subsídios para os processos judiciais 2001.34.00.024796-9 e 2003.34.00.044227-3 ; e aos demais interessados a deliberação que vier a ser proferida por esta corte ” ."""]}))

result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = new DocumentAssembler() 
    .setInputCol('text') 
    .setOutputCol('document')

val tokenizer = new Tokenizer() 
    .setInputCols(['document']) 
    .setOutputCol('token')

val embeddings = BertEmbeddings.pretrained("bert_portuguese_base_cased", "pt")
    .setInputCols("document", "token") 
    .setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained('lener_bert_base', 'pt', 'clinical/models')
    .setInputCols(['document', 'token', 'embeddings']) 
    .setOutputCol('ner')

val ner_converter = new NerConverter() 
    .setInputCols(['document', 'token', 'ner']) 
    .setOutputCol('ner_chunk')

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, embeddings, ner_model, ner_converter))

val data = Seq("""diante do exposto , com fundamento nos artigos 32 , i , e 33 , da lei 8.443/1992 , submetem-se os autos à consideração superior , com posterior encaminhamento ao ministério público junto ao tcu e ao gabinete do relator , propondo : a ) conhecer do recurso e , no mérito , negar-lhe provimento ; b ) comunicar ao recorrente , ao superior tribunal militar e ao tribunal regional federal da 2ª região , a fim de fornecer subsídios para os processos judiciais 2001.34.00.024796-9 e 2003.34.00.044227-3 ; e aos demais interessados a deliberação que vier a ser proferida por esta corte""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>


## Results


```bash
+-------------------+----------------+
|              token|             ner|
+-------------------+----------------+
|             diante|               O|
|                 do|               O|
|            exposto|               O|
|                  ,|               O|
|                com|               O|
|         fundamento|               O|
|                nos|               O|
|            artigos|    B-LEGISLACAO|
|                 32|    I-LEGISLACAO|
|                  ,|    I-LEGISLACAO|
|                  i|    I-LEGISLACAO|
|                  ,|    I-LEGISLACAO|
|                  e|    I-LEGISLACAO|
|                 33|    I-LEGISLACAO|
|                  ,|    I-LEGISLACAO|
|                 da|    I-LEGISLACAO|
|                lei|    I-LEGISLACAO|
|         8.443/1992|    I-LEGISLACAO|
|                  ,|               O|
|        submetem-se|               O|
|                 os|               O|
|              autos|               O|
|                  à|               O|
|       consideração|               O|
|           superior|               O|
|                  ,|               O|
|                com|               O|
|          posterior|               O|
|     encaminhamento|               O|
|                 ao|               O|
|         ministério|   B-ORGANIZACAO|
|            público|   I-ORGANIZACAO|
|              junto|               O|
|                 ao|               O|
|                tcu|   B-ORGANIZACAO|
|                  e|               O|
|                 ao|               O|
|           gabinete|               O|
|                 do|               O|
|            relator|               O|
|                  ,|               O|
|           propondo|               O|
|                  :|               O|
|                  a|               O|
|                  )|               O|
|           conhecer|               O|
|                 do|               O|
|            recurso|               O|
|                  e|               O|
|                  ,|               O|
|                 no|               O|
|             mérito|               O|
|                  ,|               O|
|          negar-lhe|               O|
|         provimento|               O|
|                  ;|               O|
|                  b|               O|
|                  )|               O|
|          comunicar|               O|
|                 ao|               O|
|         recorrente|               O|
|                  ,|               O|
|                 ao|               O|
|           superior|   B-ORGANIZACAO|
|           tribunal|   I-ORGANIZACAO|
|            militar|   I-ORGANIZACAO|
|                  e|               O|
|                 ao|               O|
|           tribunal|   B-ORGANIZACAO|
|           regional|   I-ORGANIZACAO|
|            federal|   I-ORGANIZACAO|
|                 da|   I-ORGANIZACAO|
|                 2ª|   I-ORGANIZACAO|
|             região|   I-ORGANIZACAO|
|                  ,|               O|
|                  a|               O|
|                fim|               O|
|                 de|               O|
|           fornecer|               O|
|          subsídios|               O|
|               para|               O|
|                 os|               O|
|          processos|               O|
|          judiciais|               O|
|2001.34.00.024796-9|B-JURISPRUDENCIA|
|                  e|               O|
|2003.34.00.044227-3|B-JURISPRUDENCIA|
|                  ;|               O|
|                  e|               O|
|                aos|               O|
|             demais|               O|
|       interessados|               O|
|                  a|               O|
|        deliberação|               O|
|                que|               O|
|               vier|               O|
|                  a|               O|
|                ser|               O|
|          proferida|               O|
|                por|               O|
|               esta|               O|
|              corte|               O|
|                  ”|               O|
|                  .|               O|
+-------------------+----------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|lener_bert_base|
|Compatibility:|Spark NLP for Healthcare 3.1.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|pt|
|Dependencies:|bert_portuguese_base_cased|


## Data Source


The model was trained on the LeNER-Br data set:


> Pedro H. Luz de Araujo, Teófilo E. de Campos, Renato R. R. de Oliveira, Matheus Stauffer, Samuel Couto and Paulo Bermejo 
> LeNER-Br: a Dataset for Named Entity Recognition in Brazilian Legal Text 
> International Conference on the Computational Processing of Portuguese (PROPOR), September 24-26, Canela, Brazil, 2018.


## Benchmarking


```bash
label             precision  recall  f1-score  support
B-JURISPRUDENCIA  0.84       0.91    0.88      175    
B-LEGISLACAO      0.96       0.96    0.96      347    
B-LOCAL           0.69       0.68    0.68      40     
B-ORGANIZACAO     0.95       0.71    0.81      441    
B-PESSOA          0.91       0.95    0.93      221    
B-TEMPO           0.94       0.86    0.90      176    
I-JURISPRUDENCIA  0.86       0.91    0.89      461    
I-LEGISLACAO      0.98       0.99    0.98      2012   
I-LOCAL           0.54       0.53    0.53      72     
I-ORGANIZACAO     0.94       0.76    0.84      768    
I-PESSOA          0.93       0.98    0.95      461    
I-TEMPO           0.90       0.85    0.88      66     
O                 0.99       1.00    0.99      38419  
accuracy          -          -       0.98      43659  
macro-avg         0.88       0.85    0.86      43659  
weighted-avg      0.98       0.98    0.98      43659  
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjE2ODY0NSw4OTY1MDE1NDFdfQ==
-->