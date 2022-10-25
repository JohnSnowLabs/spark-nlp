{%- capture title -%}
RelationExtractionDL
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}
Extracts and classifies instances of relations between named entities.
In contrast with RelationExtractionModel, RelationExtractionDLModel is based on BERT.
For pretrained models please see the
[Models Hub](https://nlp.johnsnowlabs.com/models?task=Relation+Extraction) for available models.
{%- endcapture -%}

{%- capture model_input_anno -%}
CHUNK, DOCUMENT
{%- endcapture -%}

{%- capture model_output_anno -%}
CATEGORY
{%- endcapture -%}

{%- capture model_python_medical -%}
from johnsnowlabs import * 
# Relation Extraction between body parts
# This is a continuation of the RENerChunksFilter example. See that class on how to extract the relation chunks.
# Define the extraction model
re_ner_chunk_filter = medical.RENerChunksFilter() \
 .setInputCols(["ner_chunks", "dependencies"]) \
 .setOutputCol("re_ner_chunks") \
 .setMaxSyntacticDistance(4) \
 .setRelationPairs(["internal_organ_or_component-direction"])

re_model = medical.RelationExtractionDLModel.pretrained("redl_bodypart_direction_biobert", "en", "clinical/models") \
  .setPredictionThreshold(0.5) \
  .setInputCols(["re_ner_chunks", "sentences"]) \
  .setOutputCol("relations")

trained_pipeline = Pipeline(stages=[
  documenter,
  sentencer,
  tokenizer,
  words_embedder,
  pos_tagger,
  clinical_ner_tagger,
  ner_chunker,
  dependency_parser,
  re_ner_chunk_filter,
  re_model
])

data = spark.createDataFrame([["MRI demonstrated infarction in the upper brain stem , left cerebellum and  right basil ganglia"]]).toDF("text")
result = trained_pipeline.fit(data).transform(data)

# Show results
result.selectExpr("explode(relations) as relations") \
 .select(
   "relations.metadata.chunk1",
   "relations.metadata.entity1",
   "relations.metadata.chunk2",
   "relations.metadata.entity2",
   "relations.result"
 ) \
 .where("result != 0") \
 .show(truncate=False)
+------+---------+-------------+---------------------------+------+
|chunk1|entity1  |chunk2       |entity2                    |result|
+------+---------+-------------+---------------------------+------+
|upper |Direction|brain stem   |Internal_organ_or_component|1     |
|left  |Direction|cerebellum   |Internal_organ_or_component|1     |
|right |Direction|basil ganglia|Internal_organ_or_component|1     |
+------+---------+-------------+---------------------------+------+
{%- endcapture -%}

{%- capture model_scala_medical -%}
from johnsnowlabs import * 
// Relation Extraction between body parts
// This is a continuation of the [[RENerChunksFilter]] example. See that class on how to extract the relation chunks.
// Define the extraction model
val re_ner_chunk_filter = new medical.RENerChunksFilter()
 .setInputCols("ner_chunks", "dependencies")
 .setOutputCol("re_ner_chunks")
 .setMaxSyntacticDistance(4)
 .setRelationPairs(Array("internal_organ_or_component-direction"))

val re_model = medical.RelationExtractionDLModel.pretrained("redl_bodypart_direction_biobert", "en", "clinical/models")
  .setPredictionThreshold(0.5f)
  .setInputCols("re_ner_chunks", "sentences")
  .setOutputCol("relations")

val trained_pipeline = new Pipeline().setStages(Array(
  documenter,
  sentencer,
  tokenizer,
  words_embedder,
  pos_tagger,
  clinical_ner_tagger,
  ner_chunker,
  dependency_parser,
  re_ner_chunk_filter,
  re_model
))

val data = Seq("MRI demonstrated infarction in the upper brain stem , left cerebellum and  right basil ganglia").toDF("text")
val result = trained_pipeline.fit(data).transform(data)

// Show results
//
// result.selectExpr("explode(relations) as relations")
//  .select(
//    "relations.metadata.chunk1",
//    "relations.metadata.entity1",
//    "relations.metadata.chunk2",
//    "relations.metadata.entity2",
//    "relations.result"
//  )
//  .where("result != 0")
//  .show(truncate=false)
// +------+---------+-------------+---------------------------+------+
// |chunk1|entity1  |chunk2       |entity2                    |result|
// +------+---------+-------------+---------------------------+------+
// |upper |Direction|brain stem   |Internal_organ_or_component|1     |
// |left  |Direction|cerebellum   |Internal_organ_or_component|1     |
// |right |Direction|basil ganglia|Internal_organ_or_component|1     |
// +------+---------+-------------+---------------------------+------+
//
{%- endcapture -%}


{%- capture model_python_legal -%}
from johnsnowlabs import * 

document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")
        
tokenizer = nlp.Tokenizer()\
    .setInputCols(["document"])\
    .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base", "en") \
    .setInputCols("sentence", "token") \
    .setOutputCol("embeddings")\
    .setMaxSentenceLength(512)

ner_model = legal.NerModel.pretrained("legner_contract_doc_parties", "en", "legal/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence","token","ner"])\
    .setOutputCol("ner_chunk")
    
re_model = legal.RelationExtractionDLModel.pretrained("legre_contract_doc_parties", "en", "legal/models")\
    .setPredictionThreshold(0.5)\
    .setInputCols(["ner_chunk", "sentence"])\
    .setOutputCol("relations")

pipeline = Pipeline(stages=[
        document_assembler,
        sentence_detector,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter,
        re_model
        ])

result = pipeline.fit(data).transform(data)

{%- endcapture -%}

{%- capture model_scala_legal -%}
from johnsnowlabs import * 
val documentAssembler = new nlp.DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = nlp.SentenceDetectorDLModel
    .pretrained("sentence_detector_dl","xx") 
    .setInputCols("document")
    .setOutputCol("sentence") 


val tokenizer = new nlp.Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

 
val embeddings = nlp.RoBertaEmbeddings
   .pretrained("roberta_embeddings_legal_roberta_base", "en")
   .setInputCols(Array("sentence", "token"))
   .setOutputCol("embeddings")
   .setMaxSentenceLength(512)


val ner_model = legal.NerModel
    .pretrained("legner_contract_doc_parties", "en", "legal/models") 
    .setInputCols(Array("sentence", "token","embeddings")) 
    .setOutputCol("ner")


val ner_converter = new nlp.NerConverter() 
    .setInputCols(Array("sentence", "token", "ner")) 
    .setOutputCol("ner_chunk")
   

val re_model = legal.RelationExtractionDLModel
    .pretrained("legre_contract_doc_parties", "en", "legal/models")
    .setPredictionThreshold(0.5)
    .setInputCols(Array("ner_chunk", "sentence"))
    .setOutputCol("relations")
   
    
val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  ner_model,
  ner_converter,
  re_model
))

val result = pipeline.fit(data).transform(data)

{%- endcapture -%}


{%- capture model_python_finance -%}
from johnsnowlabs import * 

document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")
        
tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

ner_model = finance.NerModel.pretrained("finner_orgs_prods_alias","en","finance/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner_org")

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence","token","ner_org"])\
    .setOutputCol("ner_chunk_org")

token_classifier = nlp.DeBertaForTokenClassification.pretrained("deberta_v3_base_token_classifier_ontonotes", "en")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("ner_date")\
    .setCaseSensitive(True)\
    .setMaxSentenceLength(512) 

ner_converter_date = nlp.NerConverter()\
    .setInputCols(["sentence","token","ner_date"])\
    .setOutputCol("ner_chunk_date")\
    .setWhiteList(["DATE"])

chunk_merger = finance.ChunkMergeApproach()\
    .setInputCols("ner_chunk_org", "ner_chunk_date")\
    .setOutputCol('ner_chunk')

re_model = finance.RelationExtractionDLModel.pretrained("finre_acquisitions_subsidiaries", "en", "finance/models")\
    .setPredictionThreshold(0.3)\
    .setInputCols(["ner_chunk", "document"])\
    .setOutputCol("relations")

pipeline = Pipeline(stages=[
        document_assembler,
        sentence_detector,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter,
        token_classifier,
        ner_converter_date,
        chunk_merger,
        re_model
        ])

result = pipeline.fit(data).transform(data)

{%- endcapture -%}


{%- capture model_scala_finance -%}
from johnsnowlabs import * 
val documentAssembler = new nlp.DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = nlp.SentenceDetectorDLModel
    .pretrained("sentence_detector_dl","xx") 
    .setInputCols("document")
    .setOutputCol("sentence") 


val tokenizer = new nlp.Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

 
val embeddings = nlp.BertEmbeddings
   .pretrained("bert_embeddings_sec_bert_base", "en")
   .setInputCols(Array("sentence", "token"))
   .setOutputCol("embeddings")


val ner_model = finance.NerModel
    .pretrained("finner_orgs_prods_alias", "en", "finance/models") 
    .setInputCols(Array("sentence", "token","embeddings")) 
    .setOutputCol("ner_org")


val ner_converter = new nlp.NerConverter() 
    .setInputCols(Array("sentence", "token", "ner_org")) 
    .setOutputCol("ner_chunk_org")
   

val token_classifier  = nlp.DeBertaForTokenClassification
    .pretrained("deberta_v3_base_token_classifier_ontonotes", "en") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner_date")
    .setCaseSensitive(True)
    .setMaxSentenceLength(512) 

val ner_converter_date = new nlp.NerConverter() 
    .setInputCols(Array("sentence", "token", "ner_date")) 
    .setOutputCol("ner_chunk_date")
    .setWhiteList(Array("DATE"))

val chunk_merger = new finance.ChunkMergeApproach()
    .setInputCols("ner_chunk_org", "ner_chunk_date")
    .setOutputCol('ner_chunk')


val re_model = finance.RelationExtractionDLModel
    .pretrained("finre_acquisitions_subsidiaries", "en", "finance/models")
    .setPredictionThreshold(0.3)
    .setInputCols(Array("ner_chunk", "document"))
    .setOutputCol("relations")
   
    
val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  ner_model,
  ner_converter,
  token_classifier,
  ner_converter_date,
  chunk_merger,
  re_model
))

val result = pipeline.fit(data).transform(data)


{%- endcapture -%}


{%- capture model_api_link -%}
[RelationExtractionDLModel](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/re/RelationExtractionDLModel)
{%- endcapture -%}



{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_scala_medical=model_scala_medical
model_python_legal=model_python_legal
model_scala_legal=model_scala_legal
model_python_finance=model_python_finance
model_scala_finance=model_scala_finance
model_api_link=model_api_link%}