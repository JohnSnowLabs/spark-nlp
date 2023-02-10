---
layout: model
title: Legal NER (Parties, Dates, Alias, Former names, Document Type - lg)
author: John Snow Labs
name: legner_contract_doc_parties_lg
date: 2023-01-21
tags: [document, contract, agreement, type, parties, aliases, former, names, effective, dates, en, licensed]
task: Named Entity Recognition
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: LegalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

MPORTANT: Don't run this model on the whole legal agreement. Instead:
- Split by paragraphs. You can use [notebook 1](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings_JSL) in Finance or Legal as inspiration;
- Use the `legclf_introduction_clause` Text Classifier to select only these paragraphs; 

This is a Legal NER Model, aimed to process the first page of the agreements when information can be found about:
- Parties of the contract/agreement;
- Their former names;
- Aliases of those parties, or how those parties will be called further on in the document;
- Document Type;
- Effective Date of the agreement;
- Other organizations;

This model can be used all along with its Relation Extraction model to retrieve the relations between these entities, called `legre_contract_doc_parties`

## Predicted Entities

`PARTY`, `EFFDATE`, `DOC`, `ALIAS`, `ORG`, `FORMER_NAME`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/LEGALNER_PARTIES/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_contract_doc_parties_lg_en_1.0.0_3.0_1674321394808.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_contract_doc_parties_lg_en_1.0.0_3.0_1674321394808.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner_model = legal.NerModel.pretrained('legner_contract_doc_parties_lg', 'en', 'legal/models')\
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
|Model Name:|legner_contract_doc_parties_lg|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.3 MB|

## References

Manual annotations on CUAD dataset

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
I-PARTY	 513	 58	 85	 0.8984238	 0.85785955	 0.87767327
B-EFFDATE	 57	 5	 7	 0.91935486	 0.890625	 0.90476185
I-ORG	 208	 32	 49	 0.8666667	 0.8093385	 0.8370222
B-DOC	 80	 9	 21	 0.8988764	 0.7920792	 0.8421053
B-FORMER_NAME	 5	 0	 0	 1.0	 1.0	 1.0
I-EFFDATE	 214	 11	 6	 0.95111114	 0.9727273	 0.96179783
I-FORMER_NAME	 7	 1	 0	 0.875	 1.0	 0.93333334
I-ALIAS	 14	 5	 13	 0.7368421	 0.5185185	 0.6086956
I-DOC	 166	 16	 52	 0.9120879	 0.7614679	 0.83
B-ORG	 131	 28	 42	 0.8238994	 0.75722545	 0.7891567
B-PARTY	 174	 40	 57	 0.8130841	 0.7532467	 0.7820225
B-ALIAS	 170	 17	 13	 0.90909094	 0.92896175	 0.9189189
Macro-average	 1739 222 345 0.8837032 0.8368375 0.859632
Micro-average	 1739 222 345 0.8867925 0.834453 0.859827
```