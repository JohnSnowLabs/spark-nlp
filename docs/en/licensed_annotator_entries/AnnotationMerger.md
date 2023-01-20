{%- capture title -%}
AnnotationMerger
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}
Merge annotations from different pipeline steps that have the same annotation type into a unified annotation. Possible annotations that can be merged include:
- document (e.g., output of `DocumentAssembler` annotator)
- token (e.g., output of `Tokenizer` annotator)
- word_embeddings (e.g., output of `WordEmbeddingsModel` annotator)
- sentence_embeddings (e.g., output of `BertSentenceEmbeddings` annotator)
- category (e.g., output of `RelationExtractionModel` annotator)
- date (e.g., output of `DateMatcher` annotator)
- sentiment (e.g., output of `SentimentDLModel` annotator)
- pos (e.g., output of `PerceptronModel` annotator)
- chunk (e.g., output of `NerConverter` annotator)
- named_entity (e.g., output of `NerDLModel` annotator)
- regex (e.g., output of `RegexTokenizer` annotator)
- dependency (e.g., output of `DependencyParserModel` annotator)
- language (e.g., output of `LanguageDetectorDL` annotator)
- keyword (e.g., output of `YakeModel` annotator)

{%- endcapture -%}

{%- capture model_input_anno -%}
ANY
{%- endcapture -%}

{%- capture model_output_anno -%}
ANY
{%- endcapture -%}

{%- capture model_python_medical -%}
# Create the pipeline with two RE models
documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentencer = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentences")

tokenizer = Tokenizer()\
    .setInputCols(["sentences"])\
    .setOutputCol("tokens")

words_embedder = WordEmbeddingsModel()\
    .pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("embeddings")

pos_tagger = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("pos_tags")

pos_ner_tagger = MedicalNerModel()\
    .pretrained("ner_posology", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_pos")

pos_ner_chunker = NerConverterInternal()\
    .setInputCols(["sentences", "tokens", "ner_pos"])\
    .setOutputCol("pos_ner_chunks")

dependency_parser = DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["sentences", "pos_tags", "tokens"])\
    .setOutputCol("dependencies")

pos_reModel = RelationExtractionModel()\
    .pretrained("posology_re")\
    .setInputCols(["embeddings", "pos_tags", "pos_ner_chunks", "dependencies"])\
    .setOutputCol("pos_relations")\
    .setMaxSyntacticDistance(4)

ade_ner_tagger = MedicalNerModel.pretrained("ner_ade_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ade_ner_tags")  

ade_ner_chunker = NerConverterInternal()\
    .setInputCols(["sentences", "tokens", "ade_ner_tags"])\
    .setOutputCol("ade_ner_chunks")

ade_reModel = RelationExtractionModel()\
    .pretrained("re_ade_clinical", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ade_ner_chunks", "dependencies"])\
    .setOutputCol("ade_relations")\
    .setMaxSyntacticDistance(10)\
    .setRelationPairs(["drug-ade, ade-drug"])

annotation_merger = AnnotationMerger()\
    .setInputCols("ade_relations", "pos_relations")\
    .setInputType("category")\
    .setOutputCol("all_relations")

merger_pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    pos_ner_tagger,
    pos_ner_chunker,
    dependency_parser,
    pos_reModel,
    ade_ner_tagger,
    ade_ner_chunker,
    ade_reModel,
    annotation_merger
])

empty_df= spark.createDataFrame([[""]]).toDF("text")
merger_model= merger_pipeline.fit(empty_df)

# Show example result
text = """
The patient was prescribed 1 unit of naproxen for 5 days after meals for chronic low back pain. The patient was also given 1 unit of oxaprozin daily for rheumatoid arthritis presented with tense bullae and cutaneous fragility on the face and the back of the hands.. 
"""
data = spark.createDataFrame([[text]]).toDF("text")

result = merger_model.transform(data)
result.show()

+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
                text|            document|           sentences|              tokens|          embeddings|            pos_tags|             ner_pos|      pos_ner_chunks|        dependencies|       pos_relations|        ade_ner_tags|      ade_ner_chunks|       ade_relations|       all_relations|
+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+

The patient was ...|[{document, 0, 26...|[{document, 1, 95...|[{token, 1, 3, Th...|[{word_embeddings...|[{pos, 1, 3, DD, ...|[{named_entity, 1...|[{chunk, 28, 33, ...|[{dependency, 1, ...|[{category, 28, 4...|[{named_entity, 1...|[{chunk, 38, 45, ...|[{category, 134, ...|[{category, 134, ...|
+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+


{%- endcapture -%}


{%- capture model_api_link -%}
[AnnotationMerger](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/annotator/AnnotationMerger.html)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[AnnotationMerger](https://nlp.johnsnowlabs.com/licensed/api/python/reference/autosummary/sparknlp_jsl/annotator/annotation_merger/index.html#sparknlp_jsl.annotator.annotation_merger.AnnotationMerger)
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
