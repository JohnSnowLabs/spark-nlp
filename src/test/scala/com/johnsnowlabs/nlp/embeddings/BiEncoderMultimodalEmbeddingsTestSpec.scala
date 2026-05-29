/*
 * Copyright 2017-2026 John Snow Labs
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
package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.ml.ai.{BiEncoderEmbeddingPair, BiEncoderMultimodal}
import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.{Annotation, AnnotationImage, AnnotatorType}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.SparkException
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{MetadataBuilder, StructField, StructType}
import org.scalatest.flatspec.AnyFlatSpec

private object StubBiEncoderMultimodal extends BiEncoderMultimodal {
  override def predict(
      documentAnnotations: Seq[Annotation],
      imageAnnotations: Seq[AnnotationImage]): Seq[BiEncoderEmbeddingPair] = {
    documentAnnotations.zip(imageAnnotations).map { case (document, image) =>
      val sharedSpaceDoc = Array(
        document.result.length.toFloat,
        document.begin.toFloat,
        document.end.toFloat,
        image.width.toFloat)
      val sharedSpaceImage = Array(
        image.width.toFloat,
        image.height.toFloat,
        image.result.length.toFloat,
        document.result.length.toFloat)
      BiEncoderEmbeddingPair(sharedSpaceDoc, sharedSpaceImage)
    }
  }
}

class BiEncoderMultimodalEmbeddingsTestSpec extends AnyFlatSpec with SparkSessionTest {

  private val documentColumnMetadata =
    new MetadataBuilder().putString("annotatorType", AnnotatorType.DOCUMENT).build()
  private val imageColumnMetadata =
    new MetadataBuilder().putString("annotatorType", AnnotatorType.IMAGE).build()

  private val inputSchema = StructType(
    Array(
      StructField(
        "vision_pair_doc",
        Annotation.arrayType,
        nullable = false,
        documentColumnMetadata),
      StructField(
        "vision_pair_image",
        AnnotationImage.arrayType,
        nullable = false,
        imageColumnMetadata)))

  private def annotationRow(annotation: Annotation): Row =
    Row(
      annotation.annotatorType,
      annotation.begin,
      annotation.end,
      annotation.result,
      annotation.metadata,
      annotation.embeddings)

  private def imageAnnotationRow(annotation: AnnotationImage): Row =
    Row(
      annotation.annotatorType,
      annotation.origin,
      annotation.height,
      annotation.width,
      annotation.nChannels,
      annotation.mode,
      annotation.result,
      annotation.metadata,
      annotation.text)

  "BiEncoderMultimodalEmbeddings" should "emit separate text and image embedding columns" taggedAs FastTest in {
    val firstDoc = Annotation(
      AnnotatorType.DOCUMENT,
      begin = 0,
      end = 10,
      result = "Finding one",
      metadata =
        Map("source_file" -> "report_2417.pdf", "page_number" -> "1", "paragraph_index" -> "3"))
    val secondDoc = Annotation(
      AnnotatorType.DOCUMENT,
      begin = 12,
      end = 22,
      result = "Finding two",
      metadata =
        Map("source_file" -> "report_2418.pdf", "page_number" -> "2", "paragraph_index" -> "0"))
    val thirdDoc = Annotation(
      AnnotatorType.DOCUMENT,
      begin = 24,
      end = 33,
      result = "Finding three",
      metadata =
        Map("source_file" -> "report_2418.pdf", "page_number" -> "2", "paragraph_index" -> "1"))

    val firstImage = AnnotationImage(
      annotatorType = AnnotatorType.IMAGE,
      origin = "img_1.png",
      height = 256,
      width = 512,
      nChannels = 3,
      mode = 16,
      result = Array[Byte](1, 2, 3),
      metadata =
        Map("source_file" -> "report_2417.pdf", "page_number" -> "1", "coord" -> "10,20"),
      text = "")
    val secondImage = AnnotationImage(
      annotatorType = AnnotatorType.IMAGE,
      origin = "img_2.png",
      height = 128,
      width = 320,
      nChannels = 3,
      mode = 16,
      result = Array[Byte](4, 5),
      metadata =
        Map("source_file" -> "report_2418.pdf", "page_number" -> "2", "coord" -> "30,40"),
      text = "")
    val thirdImage = AnnotationImage(
      annotatorType = AnnotatorType.IMAGE,
      origin = "img_3.png",
      height = 96,
      width = 192,
      nChannels = 3,
      mode = 16,
      result = Array[Byte](6, 7, 8, 9),
      metadata =
        Map("source_file" -> "report_2418.pdf", "page_number" -> "2", "coord" -> "50,60"),
      text = "")

    val inputRows = Seq(
      Row(Seq(annotationRow(firstDoc)), Seq(imageAnnotationRow(firstImage))),
      Row(
        Seq(annotationRow(secondDoc), annotationRow(thirdDoc)),
        Seq(imageAnnotationRow(secondImage), imageAnnotationRow(thirdImage))))

    val inputDf = spark.createDataFrame(spark.sparkContext.parallelize(inputRows), inputSchema)

    val embeddings = new BiEncoderMultimodalEmbeddings()
      .setInputCols("vision_pair_doc", "vision_pair_image")
      .setOutputCol("mm")
      .setBatchSize(2)
      .setModelIfNotSet(spark, StubBiEncoderMultimodal)

    val resultDf = embeddings.transform(inputDf)
    val collected = resultDf.collect()

    assert(resultDf.columns.contains("mm_doc_embeddings"))
    assert(resultDf.columns.contains("mm_image_embeddings"))
    assert(
      resultDf.schema("mm_doc_embeddings").metadata.getString("annotatorType") ==
        AnnotatorType.SENTENCE_EMBEDDINGS)
    assert(
      resultDf.schema("mm_image_embeddings").metadata.getString("annotatorType") ==
        AnnotatorType.SENTENCE_EMBEDDINGS)

    val firstRowDocEmbeddings =
      collected.head.getAs[Seq[Row]]("mm_doc_embeddings").map(Annotation(_))
    val firstRowImageEmbeddings =
      collected.head.getAs[Seq[Row]]("mm_image_embeddings").map(Annotation(_))
    assert(firstRowDocEmbeddings.length == 1)
    assert(firstRowImageEmbeddings.length == 1)
    assert(firstRowDocEmbeddings.head.result == "Finding one")
    assert(firstRowDocEmbeddings.head.metadata("modality") == "text")
    assert(firstRowDocEmbeddings.head.metadata("embedding_dim") == "4")
    assert(firstRowDocEmbeddings.head.metadata("item_id").contains("report_2417.pdf"))
    assert(firstRowImageEmbeddings.head.result == "img_1.png")
    assert(firstRowImageEmbeddings.head.metadata("modality") == "image")
    assert(firstRowImageEmbeddings.head.metadata("embedding_dim") == "4")
    assert(firstRowImageEmbeddings.head.metadata("item_id").contains("img_1.png"))

    val secondRowDocEmbeddings =
      collected(1).getAs[Seq[Row]]("mm_doc_embeddings").map(Annotation(_))
    val secondRowImageEmbeddings =
      collected(1).getAs[Seq[Row]]("mm_image_embeddings").map(Annotation(_))
    assert(secondRowDocEmbeddings.length == 2)
    assert(secondRowImageEmbeddings.length == 2)
    assert(secondRowDocEmbeddings.map(_.metadata("modality")).forall(_ == "text"))
    assert(secondRowImageEmbeddings.map(_.metadata("modality")).forall(_ == "image"))
    assert(secondRowDocEmbeddings.map(_.embeddings.length).forall(_ == 4))
    assert(secondRowImageEmbeddings.map(_.embeddings.length).forall(_ == 4))
  }

  it should "fail fast when aligned inputs are not paired by row" taggedAs FastTest in {
    val misalignedRow = Row(
      Seq(
        annotationRow(
          Annotation(
            AnnotatorType.DOCUMENT,
            begin = 0,
            end = 4,
            result = "Text",
            metadata = Map("source_file" -> "report.pdf")))),
      Seq(
        imageAnnotationRow(
          AnnotationImage(
            annotatorType = AnnotatorType.IMAGE,
            origin = "img_1.png",
            height = 100,
            width = 100,
            nChannels = 3,
            mode = 16,
            result = Array[Byte](1),
            metadata = Map("source_file" -> "report.pdf"),
            text = "")),
        imageAnnotationRow(
          AnnotationImage(
            annotatorType = AnnotatorType.IMAGE,
            origin = "img_2.png",
            height = 120,
            width = 120,
            nChannels = 3,
            mode = 16,
            result = Array[Byte](2),
            metadata = Map("source_file" -> "report.pdf"),
            text = ""))))

    val inputDf =
      spark.createDataFrame(spark.sparkContext.parallelize(Seq(misalignedRow)), inputSchema)

    val embeddings = new BiEncoderMultimodalEmbeddings()
      .setInputCols("vision_pair_doc", "vision_pair_image")
      .setOutputCol("mm")
      .setModelIfNotSet(spark, StubBiEncoderMultimodal)

    val error = intercept[SparkException] {
      embeddings.transform(inputDf).collect()
    }

    val errorMessages =
      Iterator
        .iterate(Option(error): Option[Throwable])(_.flatMap(t => Option(t.getCause)))
        .takeWhile(_.isDefined)
        .flatten
        .map(_.getMessage)
        .filter(_ != null)
        .mkString("\n")

    assert(errorMessages.contains("must contain the same number of annotations per row"))
  }
}
