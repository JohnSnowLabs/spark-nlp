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
import com.johnsnowlabs.nlp.{AnnotationImage, HasOutputAnnotationCol, HasOutputAnnotatorType}
import com.johnsnowlabs.partition.util.PartitionHelper.{datasetWithTextFile, isStringContent}
import com.johnsnowlabs.partition.{HasHTMLReaderProperties, HasReaderProperties, Partition}
import com.johnsnowlabs.reader.util.ImageHelper
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{BooleanParam, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{array, col, explode, lit, udf}
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.types.{
  ArrayType,
  Metadata,
  MetadataBuilder,
  StringType,
  StructField,
  StructType
}

import java.io.File
import scala.jdk.CollectionConverters.mapAsJavaMapConverter


/** The Reader2Image annotator allows you to use the reading files with images more smoothly within existing
 * Spark NLP workflows, enabling seamless reuse of your pipelines. Reader2Image can be used for
 * extracting structured image content from various document types using Spark NLP readers. It supports
 * reading from many files types and returns parsed output as a structured Spark DataFrame.
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
    with HasReaderProperties
    with HasHTMLReaderProperties {

  def this() = this(Identifiable.randomUID("Reader2Image"))

  val explodeDocs: BooleanParam =
    new BooleanParam(this, "explodeDocs", "whether to explode the documents into separate rows")

  def setExplodeDocs(value: Boolean): this.type = set(explodeDocs, value)

  setDefault(contentType -> "", outputFormat -> "image", explodeDocs -> true)

  override def transform(dataset: Dataset[_]): DataFrame = {
    validateRequiredParameters()
    val partition = partitionBuilder
    val structuredDf = if ($(contentType).trim.isEmpty) {
      partitionMixedContent(dataset, $(contentPath))
    } else {
      partitionContent(partition, dataset)
    }

    if (!structuredDf.isEmpty) {
      val annotatedDf = structuredDf.withColumn(
        getOutputCol,
        wrapColumnMetadata(partitionAnnotation(col(partition.getOutputColumn), col("path"))))
        .select("fileName", getOutputCol)

      afterAnnotate(annotatedDf)
    } else {
      structuredDf
    }
  }

  private def partitionContent(partition: Partition, dataset: Dataset[_]): DataFrame = {
    if (isStringContent($(contentType))) {
      val partitionUDF =
        udf((text: String) => partition.partitionStringContent(text, $(this.headers).asJava))

      datasetWithTextFile(dataset.sparkSession, $(contentPath))
        .withColumn(partition.getOutputColumn, partitionUDF(col("content")))
        .withColumn("fileName", getFileName(col("path")))
    } else {
      dataset.toDF()
    }
  }

  private def partitionMixedContent(dataset: Dataset[_], dirPath: String): DataFrame = {
    val allFiles = listAllFilesRecursively(new File(dirPath))
    val grouped = allFiles
      .filter(_.isFile)
      .groupBy { file =>
        val ext = file.getName.split("\\.").lastOption.getOrElse("").toLowerCase
        supportedTypes.get(ext).map(_ => ext)
      }
      .collect { case (Some(ext), files) => ext -> files }

    val mixedDfs = grouped.flatMap { case (ext, files) =>
      val (contentType, _) = supportedTypes(ext)
      val partitionParams = Map(
        "contentType" -> contentType,
        "inferTableStructure" -> $(inferTableStructure).toString,
        "outputFormat" -> $(outputFormat))
      val partition = new Partition(partitionParams.asJava)
      val filePaths = files.map(_.getAbsolutePath)

      if (filePaths.nonEmpty) {
        val textDfList = files.map { file =>
          val partitionUDF =
            udf((text: String) => partition.partitionStringContent(text, $(this.headers).asJava))

          val textDf = datasetWithTextFile(dataset.sparkSession, file.getAbsolutePath)
            .withColumn(partition.getOutputColumn, partitionUDF(col("content")))
            .withColumn("fileName", getFileName(col("path")))

          textDf
            .withColumnRenamed(partition.getOutputColumn, "partition")
            .withColumn("fileName", lit(file.getName))
            .drop("content")
        }
        Some(textDfList.reduce(_.unionByName(_)))
      } else None

    }.toSeq

    if (mixedDfs.isEmpty) {
      val schema = StructType(
        Seq(
          StructField("partition", StringType, nullable = true),
          StructField("fileName", StringType, nullable = true)))
      val emptyRDD = dataset.sparkSession.sparkContext.emptyRDD[Row]
      val emptyDF = dataset.sparkSession.createDataFrame(emptyRDD, schema)
      emptyDF
    } else {
      mixedDfs.reduce(_.unionByName(_))
    }
  }

  private def listAllFilesRecursively(dir: File): Seq[File] = {
    val these = Option(dir.listFiles).getOrElse(Array.empty)
    these.filter(_.isFile) ++ these.filter(_.isDirectory).flatMap(listAllFilesRecursively)
  }

  private val supportedTypes: Map[String, (String, Boolean)] = Map(
    "html" -> ("text/html", true),
    "htm" -> ("text/html", true),
    "md" -> ("text/markdown", true))

  def partitionAnnotation: UserDefinedFunction = {
    udf((partitions: Seq[Row], path: String) =>
      elementsAsIndividualAnnotations(partitions, path: String))
  }

  private def elementsAsIndividualAnnotations(
      partitions: Seq[Row],
      path: String): Seq[AnnotationImage] = {
    partitions.flatMap { partition =>
      val elementType = partition.getAs[String]("elementType").toLowerCase
      if (elementType == ElementType.IMAGE.toLowerCase) {
        buildAnnotationImage(partition, path)
      } else {
        None
      }

    }
  }

  private def buildAnnotationImage(partition: Row, path: String): Option[AnnotationImage] = {
    val content = partition.getAs[String]("content")
    val metadata = partition.getAs[Map[String, String]]("metadata")
    val encoding = metadata.getOrElse("encoding", "unknown")
    val decodedContent = if (encoding.contains("base64")) {
      ImageHelper.decodeBase64(content)
    } else {
      ImageHelper.fetchFromUrl(content)
    }
    val imageFields = ImageIOUtils.bufferedImageToImageFields(decodedContent, "test")
    if (imageFields.isDefined) {
      val origin = retrieveFileName(path)
      Some(
        AnnotationImage(
          IMAGE,
          origin,
          imageFields.get.height,
          imageFields.get.width,
          imageFields.get.nChannels,
          imageFields.get.mode,
          imageFields.get.data,
          metadata))
    } else {
      None
    }
  }

  private val getFileName = udf { path: String =>
    retrieveFileName(path)
  }

  private def retrieveFileName(path: String) = if (path != null) path.split("/").last else ""

  protected def partitionBuilder: Partition = {
    val params = Map(
      "contentType" -> $(contentType),
      "storeContent" -> $(storeContent).toString,
      "titleFontSize" -> $(titleFontSize).toString,
      "inferTableStructure" -> $(inferTableStructure).toString,
      "includePageBreaks" -> $(includePageBreaks).toString,
      "outputFormat" -> $(outputFormat))
    new Partition(params.asJava)
  }

  private def afterAnnotate(dataset: DataFrame): DataFrame = {
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
  }
}
