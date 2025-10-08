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
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, HasOutputAnnotationCol, HasOutputAnnotatorType}
import com.johnsnowlabs.partition._
import com.johnsnowlabs.partition.util.PartitionHelper.isStringContent
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import scala.jdk.CollectionConverters.mapAsJavaMapConverter

/** The Reader2Doc annotator allows you to use the reading files more smoothly within existing
  * Spark NLP workflows, enabling seamless reuse of your pipelines. Reader2Doc can be used for
  * extracting structured content from various document types using Spark NLP readers. It supports
  * reading from many files types and returns parsed output as a structured Spark DataFrame.
  *
  * Supported formats include plain text, HTML, Word (.doc/.docx), Excel (.xls/.xlsx), PowerPoint
  * (.ppt/.pptx), email files (.eml, .msg), and PDFs.
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.reader.Reader2Doc
  * import com. johnsnowlabs.nlp.base.DocumentAssembler
  * import org.apache.spark.ml.Pipeline
  *
  * val reader2Doc = new Reader2Doc()
  *   .setContentType("application/pdf")
  *   .setContentPath(s"$pdfDirectory/")
  *   .setExplodeDocs(true)
  *
  * val pipeline = new Pipeline()
  *   .setStages(Array(reader2Doc))
  *
  * val pipelineModel = pipeline.fit(emptyDataSet)
  * val resultDf = pipelineModel.transform(emptyDataSet)
  *
  * resultDf.show()
  * +------------------------------------------------------------------------------------------------------------------------------------+
  * |document                                                                                                                            |
  * +------------------------------------------------------------------------------------------------------------------------------------+
  * |[{document, 0, 14, This is a Title, {pageNumber -> 1, elementType -> Title, fileName -> pdf-title.pdf}, []}]                        |
  * |[{document, 15, 38, This is a narrative text, {pageNumber -> 1, elementType -> NarrativeText, fileName -> pdf-title.pdf}, []}]      |
  * |[{document, 39, 68, This is another narrative text, {pageNumber -> 1, elementType -> NarrativeText, fileName -> pdf-title.pdf}, []}]|
  * +------------------------------------------------------------------------------------------------------------------------------------+
  * }}}
  */
class Reader2Doc(override val uid: String)
    extends Transformer
    with DefaultParamsWritable
    with HasOutputAnnotatorType
    with HasOutputAnnotationCol
    with HasEmailReaderProperties
    with HasExcelReaderProperties
    with HasHTMLReaderProperties
    with HasPowerPointProperties
    with HasTextReaderProperties
    with HasXmlReaderProperties
    with HasReaderContent {

  def this() = this(Identifiable.randomUID("Reader2Doc"))

  val flattenOutput: BooleanParam =
    new BooleanParam(
      this,
      "flattenOutput",
      "If true, output is flattened to plain text with minimal metadata")

  def setFlattenOutput(value: Boolean): this.type = set(flattenOutput, value)

  val titleThreshold: Param[Float] =
    new Param[Float](
      this,
      "titleThreshold",
      "Minimum font size threshold for title detection in PDF docs")

  def setTitleThreshold(value: Float): this.type = {
    set(titleThreshold, value)
  }

  /** Whether to return all sentences joined into a single document
    *
    * @group param
    */
  val outputAsDocument = new BooleanParam(
    this,
    "outputAsDocument",
    "Whether to return all sentences joined into a single document")

  /** Whether to return all sentences joined into a single document */
  def setOutputAsDocument(value: Boolean): this.type = set(outputAsDocument, value)

  val excludeNonText: BooleanParam =
    new BooleanParam(this, "excludeNonText", "Excludes rows that are not text data. e.g. tables")

  /** Excludes rows that are not text data. e.g. tables */
  def setExcludeNonText(value: Boolean): this.type = set(excludeNonText, value)

  setDefault(
    this.explodeDocs -> false,
    contentType -> "",
    flattenOutput -> false,
    titleThreshold -> 18,
    outputAsDocument -> false,
    outputFormat -> "plain-text",
    excludeNonText -> false)

  override def transform(dataset: Dataset[_]): DataFrame = {
    validateRequiredParameters()
    val structuredDf = if ($(contentType).trim.isEmpty && getInputCol.trim.isEmpty) {
      val partitionParams = Map(
        "inferTableStructure" -> $(inferTableStructure).toString,
        "outputFormat" -> $(outputFormat))
      partitionMixedContent(dataset, $(contentPath), partitionParams)
    } else {
      partitionContent(partitionBuilder, $(contentPath), isStringContent(getContentType), dataset)
    }
    if (!structuredDf.isEmpty) {
      val annotatedDf = structuredDf
        .withColumn(
          getOutputCol,
          wrapColumnMetadata(partitionToAnnotation(col("partition"), col("fileName"))))

      afterAnnotate(annotatedDf).select("fileName", getOutputCol, "exception")
    } else {
      structuredDf
    }
  }

  protected def partitionBuilder: Partition = {
    val params = Map(
      "contentType" -> getContentType,
      "storeContent" -> $(storeContent).toString,
      "titleFontSize" -> $(titleFontSize).toString,
      "inferTableStructure" -> $(inferTableStructure).toString,
      "includePageBreaks" -> $(includePageBreaks).toString,
      "addAttachmentContent" -> $(addAttachmentContent).toString,
      "cellSeparator" -> $(cellSeparator),
      "appendCells" -> $(appendCells).toString,
      "timeout" -> $(timeout).toString,
      "includeSlideNotes" -> $(includeSlideNotes).toString,
      "titleLengthSize" -> $(titleLengthSize).toString,
      "groupBrokenParagraphs" -> $(groupBrokenParagraphs).toString,
      "paragraphSplit" -> $(paragraphSplit),
      "shortLineWordThreshold" -> $(shortLineWordThreshold).toString,
      "maxLineCount" -> $(maxLineCount).toString,
      "threshold" -> $(threshold).toString,
      "xmlKeepTags" -> $(xmlKeepTags).toString,
      "onlyLeafNodes" -> $(onlyLeafNodes).toString,
      "titleThreshold" -> $(titleThreshold).toString,
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

  protected def validateRequiredParameters(): Unit = {
    val hasContentPath = $(contentPath) != null && $(contentPath).trim.nonEmpty
    if (hasContentPath) {
      require(
        ResourceHelper.validFile($(contentPath)),
        "contentPath must point to a valid file or directory")
    }

    require(
      $(outputFormat) == "plain-text",
      "Only 'plain-text' outputFormat is supported for this operation.")
  }

  protected def partitionToAnnotation: UserDefinedFunction = udf {
    (partitions: Seq[Row], fileName: String) =>
      if (partitions == null) Nil
      else if ($(outputAsDocument)) {
        mergeElementsAsDocument(partitions)
      } else {
        elementsAsIndividualAnnotations(partitions)
      }
  }

  private def isTableElement(row: Row): Boolean = {
    val elementType = row.getAs[String]("elementType")
    elementType != null && (elementType.equalsIgnoreCase(ElementType.TABLE) || elementType
      .equalsIgnoreCase(ElementType.IMAGE))
  }

  private def mergeElementsAsDocument(partitions: Seq[Row]): Seq[Annotation] = {
    val partitionsToKept =
      if ($(excludeNonText)) partitions.filterNot(isTableElement) else partitions

    val allText =
      partitionsToKept.flatMap(part => Option(part.getAs[String]("content"))).mkString(" ")

    val begin = 0
    val end = if (allText.isEmpty) 0 else allText.length - 1
    val meta = Map("sentence" -> "0")
    Seq(
      Annotation(
        annotatorType = outputAnnotatorType,
        begin = begin,
        end = end,
        result = allText,
        metadata = meta,
        embeddings = Array.emptyFloatArray))
  }

  private def elementsAsIndividualAnnotations(partitions: Seq[Row]): Seq[Annotation] = {
    var currentOffset = 0
    partitions.flatMap { part =>
      val elementType = part.getAs[String]("elementType")
      if ($(excludeNonText) && isTableElement(part)) {
        None
      } else {
        val content = part.getAs[String]("content")
        val metadata = part.getAs[Map[String, String]]("metadata")
        val begin = currentOffset
        val end = currentOffset + (if (content != null) content.length else 0) - 1
        currentOffset = end + 1
        val baseMeta = if (metadata != null) metadata else Map.empty[String, String]
        val withExtras = baseMeta + ("elementType" -> elementType)
        val finalMeta =
          if ($(flattenOutput)) withExtras.filterKeys(_ == "sentence") else withExtras
        Some(
          Annotation(
            annotatorType = outputAnnotatorType,
            begin = begin,
            end = end,
            result = content,
            metadata = finalMeta,
            embeddings = Array.emptyFloatArray))
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
