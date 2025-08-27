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
import com.johnsnowlabs.partition.util.PartitionHelper.datasetWithTextFile
import com.johnsnowlabs.partition.{HasHTMLReaderProperties, HasReaderProperties, Partition}
import com.johnsnowlabs.reader.util.ImageHelper
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{BooleanParam, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{array, col, explode, udf}
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}
import org.apache.spark.sql.types.{ArrayType, Metadata, MetadataBuilder, StructField, StructType}

import scala.jdk.CollectionConverters.mapAsJavaMapConverter

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
    val partition = partitionBuilder
    val partitionUDF =
      udf((text: String) => partition.partitionStringContent(text, $(this.headers).asJava))

    val structuredDf = datasetWithTextFile(dataset.sparkSession, $(contentPath))
      .withColumn(partition.getOutputColumn, partitionUDF(col("content")))
      .withColumn("fileName", getFileName(col("path")))

    val annotatedDf = structuredDf.withColumn(
      getOutputCol,
      wrapColumnMetadata(partitionAnnotation(col(partition.getOutputColumn), col("path"))))

    afterAnnotate(annotatedDf)
  }

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
}
