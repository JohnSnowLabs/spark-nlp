{%- capture title -%}
ChunkMapper
{%- endcapture -%}

{%- capture approach -%}
approach
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}

We can use ChunkMapper to map entities with their associated code/reference based on pre-defined dictionaries. 

This is the AnnotatorModel of the ChunkMapper, which can be used to access pretrained models with the `.pretrained()` or `.load()` methods. To train a new model, check the documentation of the [ChunkMapperApproach](https://nlp.johnsnowlabs.com/docs/en/licensed_annotators#chunkmapperapproach) annotator. 

The annotator also allows using fuzzy matching, which can take into consideration parts of the tokens tha can map even when word order is different, char ngrams that can map even when thre are typos, and using fuzzy distance metric (Jaccard, Levenshtein, etc.).

Example usage and more details can be found on Spark NLP Workshop repository accessible in GitHub, for example the notebook [Healthcare Chunk Mapping](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb).

{%- endcapture -%}

{%- capture model_input_anno -%}
CHUNK
{%- endcapture -%}

{%- capture model_output_anno -%}
LABEL_DEPENDENCY
{%- endcapture -%}

{%- capture model_python_medical -%}

# Use `rxnorm_mapper` pretrained model to map entities with their corresponding RxNorm codes.

document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("ner_chunk")

chunkerMapper = ChunkMapperModel.pretrained("rxnorm_mapper", "en", "clinical/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("rxnorm")\
    .setRels(["rxnorm_code"])

mapper_pipeline = Pipeline().setStages([document_assembler, chunkerMapper])

empty_df = spark.createDataFrame([['']]).toDF('text')
mapper_model = mapper_pipeline.fit(empty_df)

mapper_lp = LightPipeline(mapper_model)
mapper_lp.fullAnnotate("metformin")

[{'ner_chunk': [Annotation(document, 0, 8, metformin, {})],
  'rxnorm': [Annotation(labeled_dependency, 0, 8, 6809, {'entity': 'metformin', 'relation': 'rxnorm_code', 'all_relations': ''})]}]

{%- endcapture -%}


{%- capture model_scala_medical -%}

val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("ner_chunk")

val chunkerMapper = ChunkMapperModel.pretrained("rxnorm_mapper", "en", "clinical/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("rxnorm")\
    .setRels(["rxnorm_code"])

mapper_pipeline = Pipeline().setStages([document_assembler, chunkerMapper])

empty_df = spark.createDataFrame([['']]).toDF('text')
mapper_model = mapper_pipeline.fit(empty_df)

mapper_lp = LightPipeline(mapper_model)
mapper_lp.fullAnnotate("metformin")

[{'ner_chunk': [Annotation(document, 0, 8, metformin, {})],
  'rxnorm': [Annotation(labeled_dependency, 0, 8, 6809, {'entity': 'metformin', 'relation': 'rxnorm_code', 'all_relations': ''})]}]

{%- endcapture -%}

{%- capture model_api_link -%}
[ChunkMapperModel](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/finance/chunk_classification/resolution/ChunkMapperModel.html)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[ChunkMapperModel](https://nlp.johnsnowlabs.com/licensed/api/python/reference/autosummary/sparknlp_jsl/annotator/chunker/chunkmapper/index.html#sparknlp_jsl.annotator.chunker.chunkmapper.ChunkMapperModel)
{%- endcapture -%}


{%- capture approach_description -%}

We can use ChunkMapper to map entities with their associated code/reference based on pre-defined dictionaries. 

This is the AnnotatorApproach of the ChunkMapper, which can be used to train ChunkMapper models by giving a custom mapping dictionary. To use pretriained models, check the documentation of the [ChunkMapperModel](https://nlp.johnsnowlabs.com/docs/en/licensed_annotators#chunkmappermodel) annotator.

The annotator also allows using fuzzy matching, which can take into consideration parts of the tokens tha can map even when word order is different, char ngrams that can map even when thre are typos, and using fuzzy distance metric (Jaccard, Levenshtein, etc.).

Example usage and more details can be found on Spark NLP Workshop repository accessible in GitHub, for example the notebook [Healthcare Chunk Mapping](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb).

{%- endcapture -%}

{%- capture approach_input_anno -%}
CHUNK
{%- endcapture -%}

{%- capture approach_output_anno -%}
LABEL_DEPENDENCY
{%- endcapture -%}

{%- capture approach_python_medical -%}

# First, create a dictionay in JSON format following this schema:
import json

data_set= {
  "mappings": [
    {
      "key": "metformin",
      "relations": [
        {
          "key": "action",
          "values" : ["hypoglycemic", "Drugs Used In Diabetes"]
        },
        {
          "key": "treatment",
          "values" : ["diabetes", "t2dm"]
        }]
    }]
}

with open('sample_drug.json', 'w', encoding='utf-8') as f:
    json.dump(data_set, f, ensure_ascii=False, indent=4)


# Create a pipeline

document_assembler = DocumentAssembler()\
      .setInputCol('text')\
      .setOutputCol('document')

sentence_detector = SentenceDetector()\
      .setInputCols(["document"])\
      .setOutputCol("sentence")

tokenizer = Tokenizer()\
      .setInputCols("sentence")\
      .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")

#NER model to detect drug in the text
clinical_ner = MedicalNerModel.pretrained("ner_posology_small","en","clinical/models")\
	    .setInputCols(["sentence","token","embeddings"])\
	    .setOutputCol("ner")\
      .setLabelCasing("upper")
 
ner_converter = NerConverter()\
      .setInputCols(["sentence", "token", "ner"])\
      .setOutputCol("ner_chunk")\
      .setWhiteList(["DRUG"])

chunkerMapper = ChunkMapperApproach()\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("mappings")\
      .setDictionary("sample_drug.json")\
      .setRels(["action"]) #or treatment

pipeline = Pipeline(
    stages=[
        document_assembler,
        sentence_detector,
        tokenizer,
        word_embeddings,
        clinical_ner,
        ner_converter,
        chunkerMapper,
    ]
)


# Train the model

text = ["The patient was given 1 unit of metformin daily."]
test_data = spark.createDataFrame([text]).toDF("text")

model = pipeline.fit(test_data)

{%- endcapture -%}


{%- capture approach_api_link -%}
[ChunkMapperApproach](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/finance/chunk_classification/resolution/ChunkMapperApproach.html)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[ChunkMapperApproach](https://nlp.johnsnowlabs.com/licensed/api/python/reference/autosummary/sparknlp_jsl/annotator/chunker/chunkmapper/index.html#sparknlp_jsl.annotator.chunker.chunkmapper.ChunkMapperApproach)
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
model_api_link=model_api_link
model_python_api_link=model_python_api_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_medical=approach_python_medical
approach_api_link=approach_api_link
approach_python_api_link=approach_python_api_link
%}
