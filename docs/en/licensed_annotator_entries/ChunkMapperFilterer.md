{%- capture title -%}
ChunkMapperFilterer
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}

`ChunkMapperFilterer` is an annotator to be used after `ChunkMapper` that allows to filter chunks based on the results of the mapping, whether it was successful or failed.

Example usage and more details can be found on Spark NLP Workshop repository accessible in GitHub, for example the notebook [Healthcare Chunk Mapping](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb).

{%- endcapture -%}

{%- capture model_input_anno -%}
CHUNK, LABEL_DEPENDENCY
{%- endcapture -%}

{%- capture model_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture model_python_medical -%}
document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

sentence_detector = (
    SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
)

tokenizer = Tokenizer().setInputCols("sentence").setOutputCol("token")

word_embeddings = (
    WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(["sentence", "token"])
    .setOutputCol("embeddings")
)

ner_model = (
    MedicalNerModel.pretrained("ner_posology_greedy", "en", "clinical/models")
    .setInputCols(["sentence", "token", "embeddings"])
    .setOutputCol("ner")
)

ner_converter = (
    NerConverter().setInputCols("sentence", "token", "ner").setOutputCol("chunk")
)

chunkerMapper = (
    ChunkMapperModel.pretrained("rxnorm_mapper", "en", "clinical/models")
    .setInputCols(["chunk"])
    .setOutputCol("RxNorm_Mapper")
    .setRel("rxnorm_code")
)

cfModel = (
    ChunkMapperFilterer()
    .setInputCols(["chunk", "RxNorm_Mapper"])
    .setOutputCol("chunks_fail")
    .setReturnCriteria("fail")
)

chunk2doc = Chunk2Doc().setInputCols("chunks_fail").setOutputCol("doc_chunk")

sbert_embedder = (
    BertSentenceEmbeddings.pretrained(
        "sbiobert_base_cased_mli", "en", "clinical/models"
    )
    .setInputCols(["doc_chunk"])
    .setOutputCol("sentence_embeddings")
    .setCaseSensitive(False)
)

resolver = (
    SentenceEntityResolverModel.pretrained(
        "sbiobertresolve_rxnorm_augmented", "en", "clinical/models"
    )
    .setInputCols(["chunks_fail", "sentence_embeddings"])
    .setOutputCol("resolver_code")
    .setDistanceFunction("EUCLIDEAN")
)

resolverMerger = (
    ResolverMerger()
    .setInputCols(["resolver_code", "RxNorm_Mapper"])
    .setOutputCol("RxNorm")
)

mapper_pipeline = Pipeline(
    stages=[
        document_assembler,
        sentence_detector,
        tokenizer,
        word_embeddings,
        ner_model,
        ner_converter,
        chunkerMapper,
        chunkerMapper,
        cfModel,
        chunk2doc,
        sbert_embedder,
        resolver,
        resolverMerger,
    ]
)

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = mapper_pipeline.fit(empty_data)


samples = [
    ["The patient was given Adapin 10 MG, coumadn 5 mg"],
    ["The patient was given Avandia 4 mg, Tegretol, zitiga"],
]

result = model.transform(spark.createDataFrame(samples).toDF("text"))

result.selectExpr(
    "chunk.result as chunk",
    "RxNorm_Mapper.result as RxNorm_Mapper",
    "chunks_fail.result as chunks_fail",
    "resolver_code.result as resolver_code",
    "RxNorm.result as RxNorm",
).show(truncate=False)
+--------------------------------+----------------------+--------------+-------------+------------------------+
chunk                           |RxNorm_Mapper         |chunks_fail   |resolver_code|RxNorm                  |
+--------------------------------+----------------------+--------------+-------------+------------------------+
[Adapin 10 MG, coumadn 5 mg]    |[1000049, NONE]       |[coumadn 5 mg]|[200883]     |[1000049, 200883]       |
[Avandia 4 mg, Tegretol, zitiga]|[261242, 203029, NONE]|[zitiga]      |[220989]     |[261242, 203029, 220989]|
+--------------------------------+----------------------+--------------+-------------+------------------------+

{%- endcapture -%}

{%- capture model_api_link -%}
[ChunkMapperFilterer](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/chunker/ChunkMapperFilterer.html)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[ChunkMapperFilterer](https://nlp.johnsnowlabs.com/licensed/api/python/reference/autosummary/sparknlp_jsl/annotator/chunker/chunkmapper_filterer/index.html#sparknlp_jsl.annotator.chunker.chunkmapper_filterer.ChunkMapperFilterer)
{%- endcapture -%}


{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
approach=approach
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_api_link=model_api_link
model_python_api_link=model_python_api_link
%}
