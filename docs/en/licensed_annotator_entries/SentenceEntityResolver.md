{%- capture title -%}
SentenceEntityResolver
{%- endcapture -%}

{%- capture approach -%}
approach
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}
The model transforms a dataset with Input Annotation type SENTENCE_EMBEDDINGS, coming from e.g.
[BertSentenceEmbeddings](/docs/en/transformers#bertsentenceembeddings)
and returns the normalized entity for a particular trained ontology / curated dataset.
(e.g. ICD-10, RxNorm, SNOMED etc.)

To use pretrained models please see the
[Models Hub](https://nlp.johnsnowlabs.com/models?task=Entity+Resolution) for available models.
{%- endcapture -%}

{%- capture model_input_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture model_output_anno -%}
ENTITY
{%- endcapture -%}

{%- capture model_python_medical -%}
from johnsnowlabs import * 
# Resolving CPT
# First define pipeline stages to extract entities
documentAssembler = nlp.DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
sentenceDetector = nlp.SentenceDetectorDLModel.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")
tokenizer = nlp.Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")
word_embeddings = nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")
clinical_ner = medical.NerModel.pretrained("jsl_ner_wip_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")
ner_converter = nlp.NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk") \
    .setWhiteList(["Test","Procedure"])
c2doc = nlp.Chunk2Doc() \
    .setInputCols(["ner_chunk"]) \
    .setOutputCol("ner_chunk_doc")
sbert_embedder = nlp.BertSentenceEmbeddings \
    .pretrained("sbiobert_base_cased_mli","en","clinical/models") \
    .setInputCols(["ner_chunk_doc"]) \
    .setOutputCol("sbert_embeddings")

# Then the resolver is defined on the extracted entities and sentence embeddings
cpt_resolver = medical.SentenceEntityResolverModel.pretrained("sbiobertresolve_cpt_procedures_augmented","en", "clinical/models") \
    .setInputCols(["ner_chunk", "sbert_embeddings"]) \
    .setOutputCol("cpt_code") \
    .setDistanceFunction("EUCLIDEAN")
sbert_pipeline_cpt = Pipeline().setStages([
    documentAssembler,
    sentenceDetector,
    tokenizer,
    word_embeddings,
    clinical_ner,
    ner_converter,
    c2doc,
    sbert_embedder,
    cpt_resolver])

sbert_outputs = sbert_pipeline_cpt.fit(data_ner).transform(data)
# Show results
#
# sbert_outputs
#   .select("explode(arrays_zip(ner_chunk.result ,ner_chunk.metadata, cpt_code.result, cpt_code.metadata, ner_chunk.begin, ner_chunk.end)) as cpt_code")
#   .selectExpr(
#     "cpt_code['0'] as chunk",
#     "cpt_code['1'].entity as entity",
#     "cpt_code['2'] as code",
#     "cpt_code['3'].confidence as confidence",
#     "cpt_code['3'].all_k_resolutions as all_k_resolutions",
#     "cpt_code['3'].all_k_results as all_k_results"
#   ).show(5)
# +--------------------+---------+-----+----------+--------------------+--------------------+
# |               chunk|   entity| code|confidence|   all_k_resolutions|         all_k_codes|
# +--------------------+---------+-----+----------+--------------------+--------------------+
# |          heart cath|Procedure|93566|    0.1180|CCA - Cardiac cat...|93566:::62319:::9...|
# |selective coronar...|     Test|93460|    0.1000|Coronary angiogra...|93460:::93458:::9...|
# |common femoral an...|     Test|35884|    0.1808|Femoral artery by...|35884:::35883:::3...|
# |   StarClose closure|Procedure|33305|    0.1197|Heart closure:::H...|33305:::33300:::3...|
# |         stress test|     Test|93351|    0.2795|Cardiovascular st...|93351:::94621:::9...|
# +--------------------+---------+-----+----------+--------------------+--------------------+
#
{%- endcapture -%}

{%- capture model_scala_medical -%}
from johnsnowlabs import * 
// Resolving CPT
// First define pipeline stages to extract entities
val documentAssembler = new nlp.DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")
val sentenceDetector = nlp.SentenceDetectorDLModel.pretrained()
  .setInputCols("document")
  .setOutputCol("sentence")
val tokenizer = new nlp.Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")
val word_embeddings = nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")
val clinical_ner = medical.NerModel.pretrained("jsl_ner_wip_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token", "embeddings"))
  .setOutputCol("ner")
val ner_converter = new nlp.NerConverter()
  .setInputCols(array("sentence", "token", "ner"))
  .setOutputCol("ner_chunk")
  .setWhiteList("Test","Procedure")
val c2doc = new nlp.Chunk2Doc()
  .setInputCols("ner_chunk")
  .setOutputCol("ner_chunk_doc")
val sbert_embedder = nlp.BertSentenceEmbeddings
  .pretrained("sbiobert_base_cased_mli","en","clinical/models")
  .setInputCols("ner_chunk_doc")
  .setOutputCol("sbert_embeddings")

// Then the resolver is defined on the extracted entities and sentence embeddings
val cpt_resolver = medical.SentenceEntityResolverModel.pretrained("sbiobertresolve_cpt_procedures_augmented","en", "clinical/models")
  .setInputCols(Array("ner_chunk", "sbert_embeddings"))
  .setOutputCol("cpt_code")
  .setDistanceFunction("EUCLIDEAN")
val sbert_pipeline_cpt = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  word_embeddings,
  clinical_ner,
  ner_converter,
  c2doc,
  sbert_embedder,
  cpt_resolver))

// Show results
//
// sbert_outputs
//   .select("explode(arrays_zip(ner_chunk.result ,ner_chunk.metadata, cpt_code.result, cpt_code.metadata, ner_chunk.begin, ner_chunk.end)) as cpt_code")
//   .selectExpr(
//     "cpt_code['0'] as chunk",
//     "cpt_code['1'].entity as entity",
//     "cpt_code['2'] as code",
//     "cpt_code['3'].confidence as confidence",
//     "cpt_code['3'].all_k_resolutions as all_k_resolutions",
//     "cpt_code['3'].all_k_results as all_k_results"
//   ).show(5)
// +--------------------+---------+-----+----------+--------------------+--------------------+
// |               chunk|   entity| code|confidence|   all_k_resolutions|         all_k_codes|
// +--------------------+---------+-----+----------+--------------------+--------------------+
// |          heart cath|Procedure|93566|    0.1180|CCA - Cardiac cat...|93566:::62319:::9...|
// |selective coronar...|     Test|93460|    0.1000|Coronary angiogra...|93460:::93458:::9...|
// |common femoral an...|     Test|35884|    0.1808|Femoral artery by...|35884:::35883:::3...|
// |   StarClose closure|Procedure|33305|    0.1197|Heart closure:::H...|33305:::33300:::3...|
// |         stress test|     Test|93351|    0.2795|Cardiovascular st...|93351:::94621:::9...|
// +--------------------+---------+-----+----------+--------------------+--------------------+
//
{%- endcapture -%}



{%- capture model_python_legal -%}
from johnsnowlabs import * 

documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
        
sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_model = legal.NerModel.pretrained("legner_orgs_prods_alias", "en", "legal/models")\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")
        
ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

chunk2doc = nlp.Chunk2Doc()\
        .setInputCols("ner_chunk")\
        .setOutputCol("ner_chunk_doc")

sentence_embeddings = nlp.UniversalSentenceEncoder.pretrained("tfhub_use", "en") \
      .setInputCols("ner_chunk_doc") \
      .setOutputCol("sentence_embeddings")
    
resolver = legal.SentenceEntityResolverModel.pretrained("legel_edgar_company_name", "en", "legal/models")\
      .setInputCols(["text", "sentence_embeddings"]) \
      .setOutputCol("resolution")\
      .setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter,
        chunk2doc,
        sentence_embeddings,
        resolver
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

 
val embeddings = nlp.BertEmbeddings
   .pretrained("bert_embeddings_sec_bert_base", "en")
   .setInputCols(Array("sentence", "token"))
   .setOutputCol("embeddings")


val ner_model = legal.NerModel
    .pretrained("legner_orgs_prods_alias", "en", "legal/models") 
    .setInputCols(Array("sentence", "token","embeddings")) 
    .setOutputCol("ner")


val ner_converter = new nlp.NerConverter() 
    .setInputCols(Array("sentence", "token", "ner")) 
    .setOutputCol("ner_chunk")

val chunk2doc = new nlp.Chunk2Doc() 
    .setInputCols("ner_chunk") 
    .setOutputCol("ner_chunk_doc")

val sentence_embeddings = nlp.UniversalSentenceEncoder
    .pretrained("tfhub_use", "en") 
    .setInputCols("ner_chunk_doc") 
    .setOutputCol("sentence_embeddings")

val resolver = legal.SentenceEntityResolverModel
    .pretrained("legel_edgar_company_name", "en", "legal/models")
    .setInputCols(Array("text", "sentence_embeddings")) 
    .setOutputCol("resolution")
    .setDistanceFunction("EUCLIDEAN")


val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  ner_model,
  ner_converter,
  chunk2doc,
  sentence_embeddings,
  resolver
))

val result = pipeline.fit(data).transform(data)

{%- endcapture -%}


{%- capture model_python_finance -%}
from johnsnowlabs import * 

documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
        
sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_model = finance.NerModel.pretrained("finner_orgs_prods_alias", "en", "finance/models")\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")
        
ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

chunk2doc = nlp.Chunk2Doc()\
        .setInputCols("ner_chunk")\
        .setOutputCol("ner_chunk_doc")

sentence_embeddings = nlp.UniversalSentenceEncoder.pretrained("tfhub_use", "en") \
      .setInputCols("ner_chunk_doc") \
      .setOutputCol("sentence_embeddings")
    
resolver = finance.SentenceEntityResolverModel.pretrained("finel_edgar_company_name", "en", "finance/models")\
      .setInputCols(["text", "sentence_embeddings"]) \
      .setOutputCol("resolution")\
      .setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter,
        chunk2doc,
        sentence_embeddings,
        resolver
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
    .setOutputCol("ner")


val ner_converter = new nlp.NerConverter() 
    .setInputCols(Array("sentence", "token", "ner")) 
    .setOutputCol("ner_chunk")

val chunk2doc = new nlp.Chunk2Doc() 
    .setInputCols("ner_chunk") 
    .setOutputCol("ner_chunk_doc")

val sentence_embeddings = nlp.UniversalSentenceEncoder
    .pretrained("tfhub_use", "en") 
    .setInputCols("ner_chunk_doc") 
    .setOutputCol("sentence_embeddings")

val resolver = finance.SentenceEntityResolverModel
    .pretrained("finel_edgar_company_name", "en", "finance/models")
    .setInputCols(Array("text", "sentence_embeddings")) 
    .setOutputCol("resolution")
    .setDistanceFunction("EUCLIDEAN")


val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  ner_model,
  ner_converter,
  chunk2doc,
  sentence_embeddings,
  resolver
))

val result = pipeline.fit(data).transform(data)


{%- endcapture -%}


{%- capture model_api_link -%}
[SentenceEntityResolverModel](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/resolution/SentenceEntityResolverModel)
{%- endcapture -%}

{%- capture approach_description -%}
Contains all the parameters and methods to train a SentenceEntityResolverModel.
The model transforms a dataset with Input Annotation type SENTENCE_EMBEDDINGS, coming from e.g.
[BertSentenceEmbeddings](/docs/en/transformers#bertsentenceembeddings)
and returns the normalized entity for a particular trained ontology / curated dataset.
(e.g. ICD-10, RxNorm, SNOMED etc.)

To use pretrained models please use SentenceEntityResolverModel
and see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Entity+Resolution) for available models.
{%- endcapture -%}

{%- capture approach_input_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture approach_output_anno -%}
ENTITY
{%- endcapture -%}

{%- capture approach_python_medical -%}
from johnsnowlabs import * 

# Training a SNOMED resolution model using BERT sentence embeddings
# Define pre-processing pipeline for training data. It needs consists of columns for the normalized training data and their labels.
documentAssembler = nlp.DocumentAssembler() \
  .setInputCol("normalized_text") \
  .setOutputCol("document")

sentenceDetector = nlp.SentenceDetector()\
  .setInputCols(["document"])\
  .setOutputCol("sentence")

bertEmbeddings = nlp.BertSentenceEmbeddings.pretrained("sent_biobert_pubmed_base_cased") \
  .setInputCols(["sentence"]) \
  .setOutputCol("bert_embeddings")

snomedTrainingPipeline = Pipeline(stages=[
  documentAssembler,
  sentenceDetector,
  bertEmbeddings
])
snomedTrainingModel = snomedTrainingPipeline.fit(data)
snomedData = snomedTrainingModel.transform(data).cache()

# Then the Resolver can be trained with
bertExtractor = medical.SentenceEntityResolverApproach() \
  .setNeighbours(25) \
  .setThreshold(1000) \
  .setInputCols(["bert_embeddings"]) \
  .setNormalizedCol("normalized_text") \
  .setLabelCol("label") \
  .setOutputCol("snomed_code") \
  .setDistanceFunction("EUCLIDIAN") \
  .setCaseSensitive(False)

snomedModel = bertExtractor.fit(snomedData)

{%- endcapture -%}

{%- capture approach_python_legal -%}
from johnsnowlabs import * 

# Training a SNOMED resolution model using BERT sentence embeddings
# Define pre-processing pipeline for training data. It needs consists of columns for the normalized training data and their labels.
documentAssembler = nlp.DocumentAssembler() \
  .setInputCol("normalized_text") \
  .setOutputCol("document")

sentenceDetector = nlp.SentenceDetector()\
  .setInputCols(["document"])\
  .setOutputCol("sentence")

bertEmbeddings = nlp.BertSentenceEmbeddings.pretrained("sent_biobert_pubmed_base_cased") \
  .setInputCols(["sentence"]) \
  .setOutputCol("bert_embeddings")

snomedTrainingPipeline = Pipeline(stages=[
  documentAssembler,
  sentenceDetector,
  bertEmbeddings
])
snomedTrainingModel = snomedTrainingPipeline.fit(data)
snomedData = snomedTrainingModel.transform(data).cache()

# Then the Resolver can be trained with
bertExtractor = legal.SentenceEntityResolverApproach() \
  .setNeighbours(25) \
  .setThreshold(1000) \
  .setInputCols(["bert_embeddings"]) \
  .setNormalizedCol("normalized_text") \
  .setLabelCol("label") \
  .setOutputCol("snomed_code") \
  .setDistanceFunction("EUCLIDIAN") \
  .setCaseSensitive(False)

snomedModel = bertExtractor.fit(snomedData)

{%- endcapture -%}

{%- capture approach_python_finance -%}
from johnsnowlabs import * 

# Training a SNOMED resolution model using BERT sentence embeddings
# Define pre-processing pipeline for training data. It needs consists of columns for the normalized training data and their labels.
documentAssembler = nlp.DocumentAssembler() \
  .setInputCol("normalized_text") \
  .setOutputCol("document")

sentenceDetector = nlp.SentenceDetector()\
  .setInputCols(["document"])\
  .setOutputCol("sentence")

bertEmbeddings = nlp.BertSentenceEmbeddings.pretrained("sent_biobert_pubmed_base_cased") \
  .setInputCols(["sentence"]) \
  .setOutputCol("bert_embeddings")

snomedTrainingPipeline = Pipeline(stages=[
  documentAssembler,
  sentenceDetector,
  bertEmbeddings
])
snomedTrainingModel = snomedTrainingPipeline.fit(data)
snomedData = snomedTrainingModel.transform(data).cache()

# Then the Resolver can be trained with
bertExtractor = finance.SentenceEntityResolverApproach() \
  .setNeighbours(25) \
  .setThreshold(1000) \
  .setInputCols(["bert_embeddings"]) \
  .setNormalizedCol("normalized_text") \
  .setLabelCol("label") \
  .setOutputCol("snomed_code") \
  .setDistanceFunction("EUCLIDIAN") \
  .setCaseSensitive(False)

snomedModel = bertExtractor.fit(snomedData)

{%- endcapture -%}

{%- capture approach_scala_medical -%}
from johnsnowlabs import * 
// Training a SNOMED resolution model using BERT sentence embeddings
// Define pre-processing pipeline for training data. It needs consists of columns for the normalized training data and their labels.
val documentAssembler = new nlp.DocumentAssembler()
   .setInputCol("normalized_text")
   .setOutputCol("document")

val sentenceDetector = nlp.SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

 val bertEmbeddings = nlp.BertSentenceEmbeddings.pretrained("sent_biobert_pubmed_base_cased")
   .setInputCols("sentence")
   .setOutputCol("bert_embeddings")
 val snomedTrainingPipeline = new Pipeline().setStages(Array(
   documentAssembler,
   sentenceDetector,
   bertEmbeddings
 ))
 val snomedTrainingModel = snomedTrainingPipeline.fit(data)
 val snomedData = snomedTrainingModel.transform(data).cache()

// Then the Resolver can be trained with
val bertExtractor = new medical.SentenceEntityResolverApproach()
  .setNeighbours(25)
  .setThreshold(1000)
  .setInputCols("bert_embeddings")
  .setNormalizedCol("normalized_text")
  .setLabelCol("label")
  .setOutputCol("snomed_code")
  .setDistanceFunction("EUCLIDIAN")
  .setCaseSensitive(false)

val snomedModel = bertExtractor.fit(snomedData)

{%- endcapture -%}

{%- capture approach_scala_legal -%}
from johnsnowlabs import * 
// Training a SNOMED resolution model using BERT sentence embeddings
// Define pre-processing pipeline for training data. It needs consists of columns for the normalized training data and their labels.
val documentAssembler = new nlp.DocumentAssembler()
   .setInputCol("normalized_text")
   .setOutputCol("document")

val sentenceDetector = nlp.SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

 val bertEmbeddings = nlp.BertSentenceEmbeddings.pretrained("sent_biobert_pubmed_base_cased")
   .setInputCols("sentence")
   .setOutputCol("bert_embeddings")
 val snomedTrainingPipeline = new Pipeline().setStages(Array(
   documentAssembler,
   sentenceDetector,
   bertEmbeddings
 ))
 val snomedTrainingModel = snomedTrainingPipeline.fit(data)
 val snomedData = snomedTrainingModel.transform(data).cache()

// Then the Resolver can be trained with
val bertExtractor = new legal.SentenceEntityResolverApproach()
  .setNeighbours(25)
  .setThreshold(1000)
  .setInputCols("bert_embeddings")
  .setNormalizedCol("normalized_text")
  .setLabelCol("label")
  .setOutputCol("snomed_code")
  .setDistanceFunction("EUCLIDIAN")
  .setCaseSensitive(false)

val snomedModel = bertExtractor.fit(snomedData)

{%- endcapture -%}

{%- capture approach_scala_finance -%}
from johnsnowlabs import * 
// Training a SNOMED resolution model using BERT sentence embeddings
// Define pre-processing pipeline for training data. It needs consists of columns for the normalized training data and their labels.
val documentAssembler = new nlp.DocumentAssembler()
   .setInputCol("normalized_text")
   .setOutputCol("document")

val sentenceDetector = nlp.SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

 val bertEmbeddings = nlp.BertSentenceEmbeddings.pretrained("sent_biobert_pubmed_base_cased")
   .setInputCols("sentence")
   .setOutputCol("bert_embeddings")
 val snomedTrainingPipeline = new Pipeline().setStages(Array(
   documentAssembler,
   sentenceDetector,
   bertEmbeddings
 ))
 val snomedTrainingModel = snomedTrainingPipeline.fit(data)
 val snomedData = snomedTrainingModel.transform(data).cache()

// Then the Resolver can be trained with
val bertExtractor = new finance.SentenceEntityResolverApproach()
  .setNeighbours(25)
  .setThreshold(1000)
  .setInputCols("bert_embeddings")
  .setNormalizedCol("normalized_text")
  .setLabelCol("label")
  .setOutputCol("snomed_code")
  .setDistanceFunction("EUCLIDIAN")
  .setCaseSensitive(false)

val snomedModel = bertExtractor.fit(snomedData)

{%- endcapture -%}

{%- capture approach_api_link -%}
[SentenceEntityResolverApproach](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/resolution/SentenceEntityResolverApproach)
{%- endcapture -%}





{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
approach=approach
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_scala_medical=model_scala_medical
model_python_legal=model_python_legal
model_scala_legal=model_scala_legal
model_python_finance=model_python_finance
model_scala_finance=model_scala_finance
model_api_link=model_api_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_medical=approach_python_medical
approach_python_legal=approach_python_legal
approach_python_finance=approach_python_finance
approach_scala_medical=approach_scala_medical
approach_scala_legal=approach_scala_legal
approach_scala_finance=approach_scala_finance
approach_api_link=approach_api_link
%}
