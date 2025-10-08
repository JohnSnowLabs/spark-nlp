/*
 * Copyright 2017-2025 John Snow Labs
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
package com.johnsnowlabs.reader

import com.johnsnowlabs.nlp.AnnotatorType.IMAGE
import com.johnsnowlabs.nlp.annotators.cv.util.io.ImageIOUtils
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{AnnotationImage, HasOutputAnnotationCol, HasOutputAnnotatorType}
import com.johnsnowlabs.partition.util.PartitionHelper.{
  datasetWithBinaryFile,
  datasetWithTextFile,
  isStringContent
}
import com.johnsnowlabs.partition.{HasBinaryReaderProperties, Partition}
import com.johnsnowlabs.reader.util.{ImageParser, ImagePromptTemplate}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}

import scala.jdk.CollectionConverters.mapAsJavaMapConverter

/** The Reader2Image annotator allows you to use the reading files with images more smoothly
  * within existing Spark NLP workflows, enabling seamless reuse of your pipelines. Reader2Image
  * can be used for extracting structured image content from various document types using Spark
  * NLP readers. It supports reading from many files types and returns parsed output as a
  * structured Spark DataFrame.
  *
  * Supported formats include HTML and Markdown
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.reader.Reader2Image
  * import com. johnsnowlabs.nlp.base.DocumentAssembler
  * import org.apache.spark.ml.Pipeline
  *
  * val reader2Image = new Reader2Image()
  *   .setContentType("text/html")
  *   .setContentPath("./example-images.html")
  *   .setOutputCol("image")
  *
  * val pipeline = new Pipeline()
  *   .setStages(Array(reader2Image))
  *
  * val pipelineModel = pipeline.fit(emptyDataSet)
  * val resultDf = pipelineModel.transform(emptyDataSet)
  *
  * resultDf.show()
  * +-------------------+--------------------+
  * |           fileName|               image|
  * +-------------------+--------------------+
  * |example-images.html|[{image, example-...|
  * |example-images.html|[{image, example-...|
  * +-------------------+--------------------+
  *
  * resultDf.printSchema()
  *
  * root
  *  |-- fileName: string (nullable = true)
  *  |-- image: array (nullable = false)
  *  |    |-- element: struct (containsNull = true)
  *  |    |    |-- annotatorType: string (nullable = true)
  *  |    |    |-- origin: string (nullable = true)
  *  |    |    |-- height: integer (nullable = false)
  *  |    |    |-- width: integer (nullable = false)
  *  |    |    |-- nChannels: integer (nullable = false)
  *  |    |    |-- mode: integer (nullable = false)
  *  |    |    |-- result: binary (nullable = true)
  *  |    |    |-- metadata: map (nullable = true)
  *  |    |    |    |-- key: string
  *  |    |    |    |-- value: string (valueContainsNull = true)
  *  |    |    |-- text: string (nullable = true)
  *
  * }}}
  */
class Reader2Image(override val uid: String)
    extends Transformer
    with DefaultParamsWritable
    with HasOutputAnnotatorType
    with HasOutputAnnotationCol
    with HasBinaryReaderProperties
    with HasReaderContent {

  def this() = this(Identifiable.randomUID("Reader2Image"))

  val userMessage: Param[String] = new Param[String](this, "userMessage", "custom user message")

  def setUserMessage(value: String): this.type = set(userMessage, value)

  val promptTemplate: Param[String] =
    new Param[String](this, "promptTemplate", "format of the output prompt")

  def setPromptTemplate(value: String): this.type = set(promptTemplate, value)

  val customPromptTemplate: Param[String] =
    new Param[String](this, "customPromptTemplate", "custom prompt template for image models")

  def setCustomPromptTemplate(value: String): this.type = set(promptTemplate, value)

  setDefault(
    contentType -> "",
    outputFormat -> "image",
    explodeDocs -> true,
    userMessage -> "Describe this image",
    promptTemplate -> "qwen2vl-chat",
    readAsImage -> true,
    customPromptTemplate -> "",
    ignoreExceptions -> true)

  override def transform(dataset: Dataset[_]): DataFrame = {
    validateRequiredParameters()
    val partition = partitionBuilder
    val structuredDf = if ($(contentType).trim.isEmpty) {
      val partitionParams =
        Map("outputFormat" -> $(outputFormat), "readAsImage" -> $(readAsImage).toString)
      partitionMixedContent(dataset, $(contentPath), partitionParams)
    } else {
      partitionContent(partition, $(contentPath), isStringContent($(contentType)), dataset)
    }
    if (!structuredDf.isEmpty) {
      val annotatedDf = structuredDf
        .withColumn(
          getOutputCol,
          wrapColumnMetadata(partitionToAnnotation(col(partition.getOutputColumn), col("path"))))

      afterAnnotate(annotatedDf).select("fileName", getOutputCol, "exception")
    } else {
      structuredDf
    }
  }

  override def partitionContent(
      partition: Partition,
      contentPath: String,
      isText: Boolean,
      dataset: Dataset[_]): DataFrame = {

    val ext = contentPath.split("\\.").lastOption.getOrElse("").toLowerCase
    if (! $(ignoreExceptions) && !supportedTypes.contains(ext)) {
      return buildErrorDataFrame(dataset, contentPath, ext)
    }

    if (Seq("png", "jpg", "jpeg", "bmp", "gif").contains(ext)) {
      // Direct image files: bypass Partition, wrap as IMAGE
      val binaryDf = datasetWithBinaryFile(dataset.sparkSession, contentPath)
      val imageUDF = udf((bytes: Array[Byte]) => {
        val metadata = Map("format" -> ext)
        Seq(
          HTMLElement(
            elementType = ElementType.IMAGE,
            content = "",
            metadata = scala.collection.mutable.Map(metadata.toSeq: _*),
            binaryContent = Some(bytes)))
      })
      binaryDf
        .withColumn(partition.getOutputColumn, imageUDF(col("content")))
        .withColumn("fileName", getFileName(col("path")))
        .withColumn("exception", lit(null: String))
        .drop("content")

    } else if (isText) {
      val partitionUDF =
        udf((text: String) => partition.partitionStringContent(text, $(this.headers).asJava))

      datasetWithTextFile(dataset.sparkSession, contentPath)
        .withColumn(partition.getOutputColumn, partitionUDF(col("content")))
        .withColumn("fileName", getFileName(col("path")))
        .withColumn("exception", lit(null: String))
        .drop("content")

    } else {
      val partitionUDF =
        udf((input: Array[Byte]) => partition.partitionBytesContent(input))

      import org.apache.spark.sql.functions._

      val dfWithException = datasetWithBinaryFile(dataset.sparkSession, contentPath)
        .withColumn(partition.getOutputColumn, partitionUDF(col("content")))
        .withColumn("fileName", getFileName(col("path")))
        .withColumn(
          "exception",
          element_at(
            org.apache.spark.sql.functions.transform(
              filter(
                col(partition.getOutputColumn),
                x => x.getField("elementType") === lit("Error")),
              x => x.getField("content")),
            1 // Spark arrays are 1-based
          ))
        .drop("content")

      dfWithException
    }

  }

  override val supportedTypes: Map[String, (String, Boolean)] = Map(
    "html" -> ("text/html", true),
    "htm" -> ("text/html", true),
    "md" -> ("text/markdown", true),
    "eml" -> ("message/rfc822", false),
    "msg" -> ("message/rfc822", false),
    "docx" -> ("application/msword", false),
    "doc" -> ("application/msword", false),
    "ppt" -> ("application/vnd.ms-powerpoint", false),
    "pptx" -> ("application/vnd.ms-powerpoint", false),
    "xlsx" -> ("application/vnd.ms-excel", false),
    "xls" -> ("application/vnd.ms-excel", false),
    "png" -> ("image/raw", false),
    "jpg" -> ("image/raw", false),
    "jpeg" -> ("image/raw", false),
    "bmp" -> ("image/raw", false),
    "gif" -> ("image/raw", false),
    "pdf" -> ("application/pdf", false))

  def partitionToAnnotation: UserDefinedFunction = {
    udf((partitions: Seq[Row], path: String) =>
      elementsAsIndividualAnnotations(partitions, path: String))
  }

  private def elementsAsIndividualAnnotations(
      partitions: Seq[Row],
      path: String): Seq[AnnotationImage] = {
    partitions.flatMap { partition =>
      val elementType = partition.getAs[String]("elementType").toLowerCase

      elementType match {
        case t if t == ElementType.IMAGE.toLowerCase =>
          buildAnnotationImage(partition, path)

        case t if t == ElementType.ERROR.toLowerCase =>
          // Build error annotation from the "content" field
          val errorMessage = partition.getAs[String]("content")
          val origin = retrieveFileName(path)

          Some(
            AnnotationImage(
              annotatorType = IMAGE,
              origin = origin,
              height = 0,
              width = 0,
              nChannels = 0,
              mode = 0,
              result = Array.emptyByteArray,
              metadata = Map(),
              text = errorMessage))

        case _ =>
          None
      }
    }
  }

  private def buildAnnotationImage(partition: Row, path: String): Option[AnnotationImage] = {
    val metadata = partition.getAs[Map[String, String]]("metadata")

    val binaryContentOpt =
      if (partition.schema.fieldNames.contains("binaryContent") && !partition.isNullAt(
          partition.fieldIndex("binaryContent")))
        Some(partition.getAs[Array[Byte]]("binaryContent"))
      else None

    val decodedContent = binaryContentOpt match {
      case Some(bytes) =>
        ImageParser.bytesToBufferedImage(bytes)
      case None =>
        val content = partition.getAs[String]("content")
        val encoding = metadata.getOrElse("encoding", "unknown")
        if (encoding.contains("base64")) {
          ImageParser.decodeBase64(content)
        } else {
          ImageParser.fetchFromUrl(content)
        }
    }

    val origin = retrieveFileName(path)
    val imageFields = ImageIOUtils.bufferedImageToImageFields(decodedContent, origin)

    if (imageFields.isDefined) {
      Some(
        AnnotationImage(
          IMAGE,
          origin,
          imageFields.get.height,
          imageFields.get.width,
          imageFields.get.nChannels,
          imageFields.get.mode,
          imageFields.get.data,
          metadata,
          buildPrompt))
    } else {
      None
    }
  }

  protected def partitionBuilder: Partition = {
    val params = Map(
      "contentType" -> $(contentType),
      "storeContent" -> $(storeContent).toString,
      "titleFontSize" -> $(titleFontSize).toString,
      "inferTableStructure" -> $(inferTableStructure).toString,
      "includePageBreaks" -> $(includePageBreaks).toString,
      "outputFormat" -> $(outputFormat),
      "readAsImage" -> $(readAsImage).toString)
    new Partition(params.asJava)
  }

  private def buildPrompt: String = {
    $(promptTemplate).toLowerCase() match {
      case "qwen2vl-chat" => ImagePromptTemplate.getQwen2VLChatTemplate($(userMessage))
      case "smolvl-chat" => ImagePromptTemplate.getSmolVLMChatTemplate($(userMessage))
      case "internvl-chat" => ImagePromptTemplate.getInternVLChatTemplate($(userMessage))
      case "custom" => ImagePromptTemplate.customTemplate($(customPromptTemplate), $(userMessage))
      case "none" => $(userMessage)
      case _ => $(userMessage)
    }
  }

  def afterAnnotate(dataset: DataFrame): DataFrame = {
    if ($(explodeDocs)) {
      dataset
        .select(dataset.columns.filterNot(_ == getOutputCol).map(col) :+ explode(
          col(getOutputCol)).as("_tmp"): _*)
        .withColumn(
          getOutputCol,
          array(col("_tmp"))
            .as(getOutputCol, dataset.schema.fields.find(_.name == getOutputCol).get.metadata))
        .drop("_tmp")
    } else dataset
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", outputAnnotatorType)
    val outputFields = schema.fields :+
      StructField(
        getOutputCol,
        ArrayType(AnnotationImage.dataType),
        nullable = false,
        metadataBuilder.build)
    StructType(outputFields)
  }

  override val outputAnnotatorType: AnnotatorType = IMAGE

  private lazy val columnMetadata: Metadata = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", outputAnnotatorType)
    metadataBuilder.build
  }

  private def wrapColumnMetadata(col: Column): Column = {
    col.as(getOutputCol, columnMetadata)
  }

  protected def validateRequiredParameters(): Unit = {
    require(
      $(contentPath) != null && $(contentPath).trim.nonEmpty,
      "contentPath must be set and not empty")
    require(
      ResourceHelper.validFile($(contentPath)),
      "contentPath must point to a valid file or directory")
  }

}
