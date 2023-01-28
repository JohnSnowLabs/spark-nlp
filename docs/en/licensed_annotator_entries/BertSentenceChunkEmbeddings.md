{%- capture title -%}
BertSentenceChunkEmbeddings
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}
This annotator allows aggregating sentence embeddings with ner chunk embeddings to get specific and more accurate resolution codes. It works by averaging sentence and chunk embeddings add contextual information in the embedding value. Input to this annotator is the context (sentence) and ner chunks, while the output is embedding for each chunk that can be fed to the resolver model. 

The `setChunkWeight` parameter can be used to control the influence of surrounding context.

> For more information and examples of `BertSentenceChunkEmbeddings` annotator, you can check the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop), and in special, the notebook [24.1.Improved_Entity_Resolution_with_SentenceChunkEmbeddings.ipynb](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.1.Improved_Entity_Resolution_with_SentenceChunkEmbeddings.ipynb).

{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT, CHUNK
{%- endcapture -%}

{%- capture model_output_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture model_python_medical -%}

# Define the pipeline

document_assembler = nlp.DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

tokenizer = nlp.Tokenizer()\
      .setInputCols(["document"])\
      .setOutputCol("token")

word_embeddings = nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
      .setInputCols(["document", "token"])\
      .setOutputCol("word_embeddings")

clinical_ner = medical.NerModel.pretrained("ner_abbreviation_clinical", "en", "clinical/models") \
      .setInputCols(["document", "token", "word_embeddings"]) \
      .setOutputCol("ner")

ner_converter = medical.NerConverterInternal() \
      .setInputCols(["document", "token", "ner"]) \
      .setOutputCol("ner_chunk")\
      .setWhiteList(['ABBR'])

sentence_chunk_embeddings = medical.BertSentenceChunkEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")\
      .setInputCols(["document", "ner_chunk"])\
      .setOutputCol("sentence_embeddings")\
      .setChunkWeight(0.5)\
      .setCaseSensitive(True)

abbr_resolver = medical.SentenceEntityResolverModel.pretrained("sbiobertresolve_clinical_abbreviation_acronym", "en", "clinical/models") \
      .setInputCols(["ner_chunk", "sentence_embeddings"]) \
      .setOutputCol("abbr_meaning")\
      .setDistanceFunction("EUCLIDEAN")
    

resolver_pipeline = Pipeline(
    stages = [
        document_assembler,
        tokenizer,
        word_embeddings,
        clinical_ner,
        ner_converter,
        sentence_chunk_embeddings,
        abbr_resolver
  ])


# Example results

sample_text = [
"""The patient admitted from the IR for aggressive irrigation of the Miami pouch. DISCHARGE DIAGNOSES: 1. A 58-year-old female with a history of stage 2 squamous cell carcinoma of the cervix status post total pelvic exenteration in 1991.""",
"""Gravid with estimated fetal weight of 6-6/12 pounds. LOWER EXTREMITIES: No edema. LABORATORY DATA: Laboratory tests include a CBC which is normal. 
Blood Type: AB positive. Rubella: Immune. VDRL: Nonreactive. Hepatitis C surface antigen: Negative. HIV: Negative. One-Hour Glucose: 117. Group B strep has not been done as yet."""]

from pyspark.sql.types import StringType, IntegerType

df = spark.createDataFrame(sample_text, StringType()).toDF('text')
df.show(truncate = 100)

+----------------------------------------------------------------------------------------------------+
|                                                                                                text|
+----------------------------------------------------------------------------------------------------+
|The patient admitted from the IR for aggressive irrigation of the Miami pouch. DISCHARGE DIAGNOSE...|
|Gravid with estimated fetal weight of 6-6/12 pounds. LOWER EXTREMITIES: No edema. LABORATORY DATA...|
+----------------------------------------------------------------------------------------------------+

{%- endcapture -%}


{%- capture model_scala_medical -%}

val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

 val sentenceDetector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

 val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("tokens")

 val wordEmbeddings = BertEmbeddings
    .pretrained("biobert_pubmed_base_cased")
    .setInputCols(Array("sentence", "tokens"))
    .setOutputCol("word_embeddings")

 val nerModel = MedicalNerModel
    .pretrained("ner_clinical_biobert", "en", "clinical/models")
    .setInputCols(Array("sentence", "tokens", "word_embeddings"))
    .setOutputCol("ner")

  val nerConverter = new NerConverter()
    .setInputCols("sentence", "tokens", "ner")
    .setOutputCol("ner_chunk")

 val sentenceChunkEmbeddings = BertSentenceChunkEmbeddings
    .pretrained("sbluebert_base_uncased_mli", "en", "clinical/models")
     .setInputCols(Array("sentence", "ner_chunk"))
     .setOutputCol("sentence_chunk_embeddings")

 val pipeline = new Pipeline()
      .setStages(Array(
          documentAssembler,
          sentenceDetector,
          tokenizer,
          wordEmbeddings,
          nerModel,
          nerConverter,
          sentenceChunkEmbeddings))

 val sampleText = "Her Diabetes has become type 2 in the last year with her Diabetes." +
    " He complains of swelling in his right forearm."

 val testDataset = Seq("").toDS.toDF("text")
 val result = pipeline.fit(emptyDataset).transform(testDataset)

 result
    .selectExpr("explode(sentence_chunk_embeddings) AS s")
    .selectExpr("s.result", "slice(s.embeddings, 1, 5) AS averageEmbedding")
    .show(truncate=false)

 +-----------------------------+-----------------------------------------------------------------+
 |                       result|                                                 averageEmbedding|
 +-----------------------------+-----------------------------------------------------------------+
 |Her Diabetes                 |[-0.31995273, -0.04710883, -0.28973156, -0.1294758, 0.12481072]  |
 |type 2                       |[-0.027161136, -0.24613449, -0.0949309, 0.1825444, -0.2252143]   |
 |her Diabetes                 |[-0.31995273, -0.04710883, -0.28973156, -0.1294758, 0.12481072]  |
 |swelling in his right forearm|[-0.45139068, 0.12400375, -0.0075617577, -0.90806055, 0.12871636]|
 +-----------------------------+-----------------------------------------------------------------+

{%- endcapture -%}


{%- capture model_api_link -%}
[BertSentenceChunkEmbeddings](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/embeddings/BertSentenceChunkEmbeddings.html)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[BertSentenceChunkEmbeddings](https://nlp.johnsnowlabs.com/licensed/api/python/reference/autosummary/sparknlp_jsl/annotator/embeddings/bert_sentence_embeddings/index.html#sparknlp_jsl.annotator.embeddings.bert_sentence_embeddings.BertSentenceChunkEmbeddings)
{%- endcapture -%}


{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_scala_medical=model_scala_medical
model_api_link=model_api_link
model_python_api_link=model_python_api_link
%}
