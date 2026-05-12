/*
 * Copyright 2017-2024 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.ml.ai

import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, IMAGE, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.scalatest.flatspec.AnyFlatSpec

private class TestableVectorDBConnector extends VectorDBConnector {
  def testValidate(schema: StructType): Boolean = validate(schema)
}

class VectorDBConnectorMultimodalTest extends AnyFlatSpec {

  private lazy val spark: SparkSession = SparkSession
    .builder()
    .appName("VectorDBConnectorMultimodalTest")
    .master("local[*]")
    .config("spark.driver.memory", "4G")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.jsl.settings.vectordb.api.key", "")
    .getOrCreate()

  "VectorDBConnector" should "default to text modality mode" taggedAs FastTest in {
    val connector = new VectorDBConnector()
    assert(connector.getModalityMode === "text")
  }

  it should "accept and return image modality mode" taggedAs FastTest in {
    val connector = new VectorDBConnector().setModalityMode("image")
    assert(connector.getModalityMode === "image")
  }

  it should "accept and return text modality mode explicitly" taggedAs FastTest in {
    val connector = new VectorDBConnector().setModalityMode("text")
    assert(connector.getModalityMode === "text")
  }

  it should "reject an unsupported modality mode value" taggedAs FastTest in {
    intercept[IllegalArgumentException] {
      new VectorDBConnector().setModalityMode("audio")
    }
  }

  it should "validate schema correctly for text mode (DOCUMENT + SENTENCE_EMBEDDINGS)" taggedAs FastTest in {
    val connector = new TestableVectorDBConnector()
      .setInputCols("document", "sentence_embeddings")
      .setOutputCol("out")
      .setModalityMode("text")

    val schema = buildSchema("document" -> DOCUMENT, "sentence_embeddings" -> SENTENCE_EMBEDDINGS)

    assert(
      connector.testValidate(schema),
      "text-mode validate should pass with DOCUMENT + SENTENCE_EMBEDDINGS")
  }

  it should "fail schema validation in text mode when first col is IMAGE" taggedAs FastTest in {
    val connector = new TestableVectorDBConnector()
      .setInputCols("image_col", "sentence_embeddings")
      .setOutputCol("out")
      .setModalityMode("text")

    val schema = buildSchema("image_col" -> IMAGE, "sentence_embeddings" -> SENTENCE_EMBEDDINGS)

    assert(
      !connector.testValidate(schema),
      "text-mode validate should fail with IMAGE + SENTENCE_EMBEDDINGS")
  }

  it should "validate schema correctly for image mode (IMAGE + SENTENCE_EMBEDDINGS)" taggedAs FastTest in {
    val connector = new TestableVectorDBConnector()
      .setInputCols("image_assembler", "image_embeddings")
      .setOutputCol("out")
      .setModalityMode("image")

    val schema =
      buildSchema("image_assembler" -> IMAGE, "image_embeddings" -> SENTENCE_EMBEDDINGS)

    assert(
      connector.testValidate(schema),
      "image-mode validate should pass with IMAGE + SENTENCE_EMBEDDINGS")
  }

  it should "fail schema validation in image mode when first col is DOCUMENT" taggedAs FastTest in {
    val connector = new TestableVectorDBConnector()
      .setInputCols("document", "sentence_embeddings")
      .setOutputCol("out")
      .setModalityMode("image")

    val schema = buildSchema("document" -> DOCUMENT, "sentence_embeddings" -> SENTENCE_EMBEDDINGS)

    assert(
      !connector.testValidate(schema),
      "image-mode validate should fail with DOCUMENT + SENTENCE_EMBEDDINGS")
  }

  it should "preserve text-mode backward compatibility (modality key in Pinecone metadata)" taggedAs FastTest in {
    val connector = new VectorDBConnector()
    assert(connector.getProvider === "pinecone")
    assert(connector.getBatchSize === 100)
    assert(connector.getModalityMode === "text")
  }

  it should "allow chaining setModalityMode with other setters" taggedAs FastTest in {
    val connector = new VectorDBConnector()
      .setInputCols("image_assembler", "image_embeddings")
      .setOutputCol("result")
      .setProvider("pinecone")
      .setIndexName("my-multimodal-index")
      .setModalityMode("image")
      .setBatchSize(25)

    assert(connector.getModalityMode === "image")
    assert(connector.getBatchSize === 25)
    assert(connector.getIndexName === "my-multimodal-index")
  }

  "VectorDBConnector in image mode" should "index image embeddings from E5VEmbeddings end-to-end" taggedAs SlowTest in {
    import com.johnsnowlabs.nlp.base.ImageAssembler
    import com.johnsnowlabs.nlp.embeddings.E5VEmbeddings

    val imageFolder = "src/test/resources/image/"

    val imageDF = spark.read
      .format("image")
      .option("dropInvalid", value = true)
      .load(imageFolder)

    val imageAssembler = new ImageAssembler()
      .setInputCol("image")
      .setOutputCol("image_assembler")

    val e5vEmbeddings = E5VEmbeddings
      .pretrained("e5v_int4")
      .setInputCols("image_assembler")
      .setOutputCol("image_embeddings")

    val vectorDB = new VectorDBConnector()
      .setInputCols("image_assembler", "image_embeddings")
      .setOutputCol("vectordb_result")
      .setProvider("pinecone")
      .setIndexName("multimodal-test-index")
      .setNamespace("image-integration-test")
      .setModalityMode("image")
      .setBatchSize(10)

    val pipeline = new Pipeline()
      .setStages(Array(imageAssembler, e5vEmbeddings, vectorDB))

    val result = pipeline.fit(imageDF).transform(imageDF)
    result.select("vectordb_result").show(false)

    val resultCount = result.count()
    assert(resultCount > 0, "Should process at least one image")

    val firstRow = result.select("vectordb_result.metadata").head()
    val metaList = firstRow.getSeq[Map[String, String]](0)
    assert(
      metaList.exists(m => m.getOrElse("vectordb_status", "") == "upserted"),
      "Output annotation should carry vectordb_status=upserted")
  }

  "VectorDBConnector in text mode" should "still upsert text embeddings with modality metadata" taggedAs SlowTest in {
    import spark.implicits._
    import com.johnsnowlabs.nlp.base.DocumentAssembler
    import com.johnsnowlabs.nlp.annotator.Tokenizer
    import com.johnsnowlabs.nlp.embeddings.{AlbertEmbeddings, SentenceEmbeddings}

    val testData = Seq(
      ("mm_text_1", "Multimodal document one", "tech"),
      ("mm_text_2", "Multimodal document two", "ai"))
      .toDF("id", "text", "category")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = AlbertEmbeddings
      .pretrained("albert_embeddings_albert_xlarge_v1")
      .setInputCols("document", "token")
      .setOutputCol("word_embeddings")

    val sentenceEmbeddings = new SentenceEmbeddings()
      .setInputCols("document", "word_embeddings")
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val vectorDB = new VectorDBConnector()
      .setInputCols("document", "sentence_embeddings")
      .setOutputCol("vectordb_result")
      .setProvider("pinecone")
      .setIndexName("multimodal-test-index")
      .setNamespace("text-integration-test")
      .setIdColumn("id")
      .setMetadataColumns(Array("text", "category"))
      .setModalityMode("text") // explicit – same as default
      .setBatchSize(10)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, embeddings, sentenceEmbeddings, vectorDB))

    val result = pipeline.fit(testData).transform(testData)
    result.select("vectordb_result.result").show(false)

    assert(result.count() == 2, "Should process both text documents")
  }

  private def buildSchema(cols: (String, String)*): StructType = {
    val fields = cols.map { case (name, annotatorType) =>
      val metaBuilder = new MetadataBuilder()
      metaBuilder.putString("annotatorType", annotatorType)
      StructField(name, ArrayType(StringType), nullable = false, metaBuilder.build())
    }
    StructType(fields)
  }
}
