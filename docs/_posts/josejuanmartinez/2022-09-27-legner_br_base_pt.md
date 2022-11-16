---
layout: model
title: Brazilian Portuguese NER for Laws (Base)
author: John Snow Labs
name: legner_br_base
date: 2022-09-27
tags: [pt, licensed]
task: Named Entity Recognition
language: pt
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: LegalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a Deep Learning Portuguese Named Entity Recognition model for the legal domain, trained using Base Bert Embeddings, and is able to predict the following entities:

- ORGANIZACAO (Organizations)
- JURISPRUDENCIA (Jurisprudence)
- PESSOA (Person)
- TEMPO (Time)
- LOCAL (Location)
- LEGISLACAO (Laws)
- O (Other)

You can find different versions of this model in Models Hub:
- With a Deep Learning architecture (non-transformer) and Base Embeddings;
- With a Deep Learning architecture (non-transformer) and Large Embeddings;
- With a Transformers Architecture and Base Embeddings;
- With a Transformers Architecture and Large Embeddings;

## Predicted Entities

`PESSOA`, `ORGANIZACAO`, `LEGISLACAO`, `JURISPRUDENCIA`, `TEMPO`, `LOCAL`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_LEGAL_PT/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_br_base_pt_1.0.0_3.0_1664276774137.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = nlp.Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

embeddings = nlp.BertEmbeddings.pretrained("bert_portuguese_base_cased", "pt")\
    .setInputCols("document", "token") \
    .setOutputCol("embeddings")

ner_model = legal.NerModel.pretrained('legner_br_base', 'pt', 'legal/models') \
    .setInputCols(['document', 'token', 'embeddings']) \
    .setOutputCol('ner')

ner_converter = nlp.NerConverter() \
    .setInputCols(['document', 'token', 'ner']) \
    .setOutputCol('ner_chunk')

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    embeddings,
    ner_model,
    ner_converter
])

example = spark.createDataFrame(pd.DataFrame({'text': ["""Mediante do exposto , com fundamento nos artigos 32 , i , e 33 , da lei 8.443/1992 , submetem-se os autos à consideração superior , com posterior encaminhamento ao ministério público junto ao tcu e ao gabinete do relator , propondo : a ) conhecer do recurso e , no mérito , negar-lhe provimento ; b ) comunicar ao recorrente , ao superior tribunal militar e ao tribunal regional federal da 2ª região , a fim de fornecer subsídios para os processos judiciais 2001.34.00.024796-9 e 2003.34.00.044227-3 ; e aos demais interessados a deliberação que vier a ser proferida por esta corte ” ."""]}))

result = pipeline.fit(example).transform(example)
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
|Model Name:|legner_br_base|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|pt|
|Size:|18.8 MB|

## References

Original texts available in https://paperswithcode.com/sota?task=Token+Classification&dataset=lener_br and in-house data augmentation with weak labelling

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