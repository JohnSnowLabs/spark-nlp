---
layout: model
title: Legal NER (Parties, Dates, Document Type - sm)
author: John Snow Labs
name: legner_contract_doc_parties
date: 2022-08-16
tags: [en, legal, ner, agreements, licensed]
task: Named Entity Recognition
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

IMPORTANT: Don't run this model on the whole legal agreement. Instead:
- Split by paragraphs. You can use [notebook 1](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings_JSL) in Finance or Legal as inspiration;
- Use the `legclf_introduction_clause` Text Classifier to select only these paragraphs; 

This is a Legal NER Model, aimed to process the first page of the agreements when information can be found about:
- Parties of the contract/agreement;
- Aliases of those parties, or how those parties will be called further on in the document;
- Document Type;
- Effective Date of the agreement;

This model can be used all along with its Relation Extraction model to retrieve the relations between these entities, called `legre_contract_doc_parties`

Other models can be found to detect other parts of the document, as Headers/Subheaders, Signers, "Will-do", etc.

## Predicted Entities

`PARTY`, `EFFDATE`, `DOC`, `ALIAS`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/LEGALNER_PARTIES/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_contract_doc_parties_en_1.0.0_3.2_1660647946284.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_contract_doc_parties_en_1.0.0_3.2_1660647946284.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
        
sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base", "en") \
        .setInputCols("sentence", "token") \
        .setOutputCol("embeddings")\

ner_model = legal.NerModel.pretrained('legner_contract_doc_parties', 'en', 'legal/models')\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = nlp.Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

text = ["""
INTELLECTUAL PROPERTY AGREEMENT

This INTELLECTUAL PROPERTY AGREEMENT (this "Agreement"), dated as of December 31, 2018 (the "Effective Date") is entered into by and between Armstrong Flooring, Inc., a Delaware corporation ("Seller") and AFI Licensing LLC, a Delaware limited liability company ("Licensing" and together with Seller, "Arizona") and AHF Holding, Inc. (formerly known as Tarzan HoldCo, Inc.), a Delaware corporation ("Buyer") and Armstrong Hardwood Flooring Company, a Tennessee corporation (the "Company" and together with Buyer the "Buyer Entities") (each of Arizona on the one hand and the Buyer Entities on the other hand, a "Party" and collectively, the "Parties").
"""]

res = model.transform(spark.createDataFrame([text]).toDF("text"))
```

</div>

## Results

```bash
+------------+---------+
|       token|ner_label|
+------------+---------+
|INTELLECTUAL|    B-DOC|
|    PROPERTY|    I-DOC|
|   AGREEMENT|    I-DOC|
|        This|        O|
|INTELLECTUAL|    B-DOC|
|    PROPERTY|    I-DOC|
|   AGREEMENT|    I-DOC|
|           (|        O|
|        this|        O|
|           "|        O|
|   Agreement|        O|
|         "),|        O|
|       dated|        O|
|          as|        O|
|          of|        O|
|    December|B-EFFDATE|
|          31|I-EFFDATE|
|           ,|I-EFFDATE|
|        2018|I-EFFDATE|
|           (|        O|
|         the|        O|
|           "|        O|
|   Effective|        O|
|        Date|        O|
|          ")|        O|
|          is|        O|
|     entered|        O|
|        into|        O|
|          by|        O|
|         and|        O|
|     between|        O|
|   Armstrong|  B-PARTY|
|    Flooring|  I-PARTY|
|           ,|  I-PARTY|
|         Inc|  I-PARTY|
|          .,|        O|
|           a|        O|
|    Delaware|        O|
| corporation|        O|
|          ("|        O|
|      Seller|  B-ALIAS|
|          ")|        O|
|         and|        O|
|         AFI|  B-PARTY|
|   Licensing|  I-PARTY|
|         LLC|  I-PARTY|
|           ,|        O|
|           a|        O|
|    Delaware|        O|
|     limited|        O|
|   liability|        O|
|     company|        O|
|          ("|        O|
|   Licensing|  B-ALIAS|
|           "|        O|
|         and|        O|
|    together|        O|
|        with|        O|
|      Seller|  B-ALIAS|
|           ,|        O|
|           "|        O|
|     Arizona|  B-ALIAS|
|          ")|        O|
|         and|        O|
|         AHF|  B-PARTY|
|     Holding|  I-PARTY|
|           ,|  I-PARTY|
|         Inc|  I-PARTY|
|           .|        O|
|           (|        O|
|    formerly|        O|
|       known|        O|
|          as|        O|
|      Tarzan|        O|
|      HoldCo|        O|
|           ,|        O|
|         Inc|        O|
|         .),|        O|
|           a|        O|
|    Delaware|        O|
| corporation|        O|
|          ("|        O|
|       Buyer|  B-ALIAS|
|          ")|        O|
|         and|        O|
|   Armstrong|  B-PARTY|
|    Hardwood|  I-PARTY|
|    Flooring|  I-PARTY|
|     Company|  I-PARTY|
------------------------
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_contract_doc_parties|
|Type:|legal|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.5 MB|

## References

Manual annotations on CUAD dataset

## Benchmarking

```bash
label       tp     fp    fn    prec          rec           f1
I-PARTY     262    20    61    0.92907804    0.8111455     0.8661157
B-EFFDATE   22     4     9     0.84615386    0.7096774     0.77192986
B-DOC       38     4     12    0.9047619     0.76          0.82608694
I-EFFDATE   95     9     19    0.91346157    0.8333333     0.8715596
I-DOC       93     12    5     0.8857143     0.9489796     0.9162561
B-PARTY     88     10    29    0.8979592     0.75213677    0.81860465
B-ALIAS     64     7     14    0.90140843    0.82051283    0.8590604
```