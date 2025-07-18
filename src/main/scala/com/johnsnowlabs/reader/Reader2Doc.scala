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

import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.{Annotation, HasOutputAnnotationCol, HasOutputAnnotatorType}
import com.johnsnowlabs.partition.util.PartitionHelper.{
  datasetWithBinaryFile,
  datasetWithTextFile,
  isStringContent
}
import com.johnsnowlabs.partition.{HasHTMLReaderProperties, HasReaderProperties, Partition}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.{array, col, explode, udf}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}

import scala.jdk.CollectionConverters.mapAsJavaMapConverter

class Reader2Doc(override val uid: String)
    extends Transformer
    with DefaultParamsWritable
    with HasOutputAnnotatorType
    with HasOutputAnnotationCol
    with HasReaderProperties
    with HasHTMLReaderProperties {

  def this() = this(Identifiable.randomUID("Reader2Doc"))

  val explodeDocs: BooleanParam =
    new BooleanParam(this, "explodeDocs", "whether to explode the documents into separate rows")

  def setExplodeDocs(value: Boolean): this.type = set(explodeDocs, value)

  val flattenOutput: BooleanParam =
    new BooleanParam(
      this,
      "flattenOutput",
      "If true, output is flattened to plain text with minimal metadata (sentence key only)")

  def setFlattenOutput(value: Boolean): this.type = set(flattenOutput, value)

  setDefault(this.explodeDocs -> true, contentType -> "", flattenOutput -> false)

  override def transform(dataset: Dataset[_]): DataFrame = {
    validateRequiredParameters()

    val params = Map("contentType" -> $(contentType))
    val partitionInstance = new Partition(params.asJava)
    val partitionDf = partitionContent(partitionInstance, dataset)

    val annotatedDf = partitionDf
      .withColumn(
        getOutputCol,
        wrapColumnMetadata(
          partitionToAnnotation($(flattenOutput))(col("partition"), col("fileName"))))
      .select(getOutputCol)

    afterAnnotate(annotatedDf)
  }

  private def partitionContent(partition: Partition, dataset: Dataset[_]): DataFrame = {

    if (isStringContent($(contentType))) {
      val partitionUDF =
        udf((text: String) => partition.partitionStringContent(text, $(this.headers).asJava))
      val stringContentDF = datasetWithTextFile(dataset.sparkSession, $(contentPath))
      stringContentDF
        .withColumn(partition.getOutputColumn, partitionUDF(col("content")))
        .withColumn("fileName", getFileName(col("path")))
    } else {
      val binaryContentDF = datasetWithBinaryFile(dataset.sparkSession, $(contentPath))
      val partitionUDF =
        udf((input: Array[Byte]) => partition.partitionBytesContent(input))
      binaryContentDF
        .withColumn(partition.getOutputColumn, partitionUDF(col("content")))
        .withColumn("fileName", getFileName(col("path")))
    }
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

  private def validateRequiredParameters(): Unit = {
    require(
      $(contentPath) != null && $(contentPath).trim.nonEmpty,
      "contentPath must be set and not empty")
    require(
      $(contentType) != null && $(contentType).trim.nonEmpty,
      "contentType must be set and not empty")
  }

  private val getFileName = udf { path: String =>
    if (path != null) path.split("/").last else ""
  }

  private def partitionToAnnotation(flatten: Boolean) = udf {
    (partitions: Seq[Row], fileName: String) =>
      if (partitions == null) Nil
      else {
        var currentOffset = 0
        partitions.map { part =>
          val elementType = part.getAs[String]("elementType")
          val content = part.getAs[String]("content")
          val metadata = part.getAs[Map[String, String]]("metadata")
          val begin = currentOffset
          val end = currentOffset + (if (content != null) content.length else 0) - 1
          currentOffset = end + 1

          // Compute new metadata
          val baseMeta = if (metadata != null) metadata else Map.empty[String, String]
          val withExtras = baseMeta +
            ("elementType" -> elementType) +
            ("fileName" -> fileName)
          val finalMeta =
            if (flatten) withExtras.filterKeys(_ == "sentence")
            else withExtras

          Annotation(
            annotatorType = outputAnnotatorType,
            begin = begin,
            end = end,
            result = content,
            metadata = finalMeta,
            embeddings = Array.emptyFloatArray)
        }
      }
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  private lazy val columnMetadata: Metadata = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", outputAnnotatorType)
    metadataBuilder.build
  }

  override def transformSchema(schema: StructType): StructType = {
    val outputFields = schema.fields :+
      StructField(getOutputCol, ArrayType(Annotation.dataType), nullable = false, columnMetadata)
    StructType(outputFields)
  }

  private def wrapColumnMetadata(col: Column): Column = {
    col.as(getOutputCol, columnMetadata)
  }

}

object Reader2Doc extends DefaultParamsReadable[Reader2Doc]
