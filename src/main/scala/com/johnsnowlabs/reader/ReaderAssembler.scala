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

import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, IMAGE}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{
  Annotation,
  AnnotationImage,
  HasOutputAnnotationCol,
  HasOutputAnnotatorType
}
import com.johnsnowlabs.partition.util.PartitionHelper.{
  datasetWithBinaryFile,
  datasetWithTextFile,
  isStringContent
}
import com.johnsnowlabs.partition.{HasBinaryReaderProperties, HasTextReaderProperties, Partition}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Column, DataFrame, Dataset}

import java.io.File
import scala.collection.mutable
import scala.jdk.CollectionConverters.mapAsJavaMapConverter

/** The ReaderAssembler annotator provides a unified interface for combining multiple Spark NLP
 * readers (such as Reader2Doc, Reader2Table, and Reader2Image) into a single, configurable
 * component. It automatically orchestrates the execution of different readers based on input type,
 * configured priorities, and fallback strategies allowing you to handle diverse content formats
 * without manually chaining multiple readers in your pipeline.
 *
 * ReaderAssembler simplifies the process of building flexible pipelines capable of ingesting and
 * processing documents, tables, and images in a consistent way. It handles reader selection,
 * ordering, and fault-tolerance internally, ensuring that pipelines remain concise, robust, and
 * easy to maintain.
 *
 * ==Example==
 * {{{
 * import com.johnsnowlabs.reader.ReaderAssembler
 * import com.johnsnowlabs.nlp.base.DocumentAssembler
 * import org.apache.spark.ml.Pipeline
 *
 * val readerAssembler = new ReaderAssembler()
 *   .setContentType("text/html")
 *   .setContentPath(s"$htmlFilesDirectory/table-image.html")
 *   .setOutputCol("document")
 *
 * val pipeline = new Pipeline()
 *   .setStages(Array(readerAssembler))
 *
 * val pipelineModel = pipeline.fit(emptyDataSet)
 * val resultDf = pipelineModel.transform(emptyDataSet)
 *
 * resultDf.show()
 * +--------+--------------------+--------------------+--------------------+---------+
 * |fileName|       document_text|      document_table|      document_image|exception|
 * +--------+--------------------+--------------------+--------------------+---------+
 * |    null|[{document, 0, 26...|[{document, 0, 50...|[{image, , 5, 5, ...|     null|
 * +--------+--------------------+--------------------+--------------------+---------+
 * }}}
 *
 * This annotator is especially useful when working with heterogeneous input data — for example,
 * when a dataset includes PDFs, spreadsheets, and images — allowing Spark NLP to automatically
 * invoke the appropriate reader for each file type while preserving a unified schema in the output.
 */


class ReaderAssembler(override val uid: String)
    extends Transformer
    with DefaultParamsWritable
    with HasOutputAnnotatorType
    with HasOutputAnnotationCol
    with HasBinaryReaderProperties
    with HasTextReaderProperties
    with HasReaderContent {

  def this() = this(Identifiable.randomUID("ReaderAssembler"))

  val excludeNonText: BooleanParam =
    new BooleanParam(this, "excludeNonText", "Excludes rows that are not text data. e.g. tables")

  /** Excludes rows that are not text data. e.g. tables */
  def setExcludeNonText(value: Boolean): this.type = set(excludeNonText, value)

  val userMessage: Param[String] = new Param[String](this, "userMessage", "custom user message")

  def setUserMessage(value: String): this.type = set(userMessage, value)

  val promptTemplate: Param[String] =
    new Param[String](this, "promptTemplate", "format of the output prompt")

  def setPromptTemplate(value: String): this.type = set(promptTemplate, value)

  val customPromptTemplate: Param[String] =
    new Param[String](this, "customPromptTemplate", "custom prompt template for image models")

  def setCustomPromptTemplate(value: String): this.type = set(promptTemplate, value)

  setDefault(
    this.explodeDocs -> false,
    contentType -> "",
    outputFormat -> "json-table",
    inferTableStructure -> true,
    flattenOutput -> false,
    excludeNonText -> false)

  private lazy val reader2DocOutputCol: String = s"${getOutputCol}_text"
  private lazy val reader2TableOutputCol: String = s"${getOutputCol}_table"
  private lazy val reader2ImageOutputCol: String = s"${getOutputCol}_image"

  override val supportedTypes: Map[String, (String, Boolean)] = Map(
    "txt" -> ("text/plain", true),
    "html" -> ("text/html", true),
    "htm" -> ("text/html", true),
    "md" -> ("text/markdown", true),
    "xml" -> ("application/xml", true),
    "csv" -> ("text/csv", true),
    "pdf" -> ("application/pdf", false),
    "doc" -> ("application/msword", false),
    "docx" -> ("application/msword", false),
    "xls" -> ("application/vnd.ms-excel", false),
    "xlsx" -> ("application/vnd.ms-excel", false),
    "ppt" -> ("application/vnd.ms-powerpoint", false),
    "pptx" -> ("application/vnd.ms-powerpoint", false),
    "eml" -> ("message/rfc822", false),
    "msg" -> ("message/rfc822", false),
    "png" -> ("image/raw", false),
    "jpg" -> ("image/raw", false),
    "jpeg" -> ("image/raw", false),
    "bmp" -> ("image/raw", false),
    "gif" -> ("image/raw", false),
    "pdf" -> ("application/pdf", false))

  override def transform(dataset: Dataset[_]): DataFrame = {

    val structureDf =
      if (getInputCol != null && getInputCol.nonEmpty && isStringContent($(contentType))) {
        processStringInputFromDataset(dataset)
      } else {
        $(contentType) match {
          // Plain-text-like formats (txt, html, csv, etc.)
          case ct if isStringContent(ct) =>
            partitionStringContent(dataset)

          // Known binary formats (pdf, doc, docx, pptx, etc.)
          case ct if supportedTypes.exists { case (_, (mime, isText)) =>
                mime == ct && !isText
              } =>
            partitionBinaryContent(dataset)

          // Default fallback: mixed directory (or unknown but supported)
          case _ =>
            partitionMixedContentDir(dataset)
        }

      }

    if (structureDf.isEmpty) {
      structureDf
    } else {

      val annotatedDf = structureDf
        .withColumn(
          reader2DocOutputCol,
          wrapDocColumn(
            reader2Doc.partitionToAnnotation(col("partition_text"), col("fileName")),
            reader2DocOutputCol))
        .withColumn(
          reader2TableOutputCol,
          wrapTableColumn(
            reader2Table.partitionToAnnotation(col("partition_table"), col("fileName")),
            reader2TableOutputCol))
        .withColumn(
          reader2ImageOutputCol,
          wrapImageColumn(
            reader2Image.partitionToAnnotation(col("partition_image"), col("fileName")),
            reader2ImageOutputCol))

      afterAnnotate(annotatedDf)
        .select(
          "fileName",
          reader2DocOutputCol,
          reader2TableOutputCol,
          reader2ImageOutputCol,
          "exception")
    }

  }

  private def partitionStringContent(dataset: Dataset[_]): DataFrame = {
    if ($(contentType) == "text/csv") {
      val imageCol = typedLit(Seq())
      return partitionContentFromPath(
        partitionTableBuilder,
        $(contentPath),
        isText = true,
        dataset)
        .withColumnRenamed("partition", "partition_text")
        .withColumn("partition_table", col("partition_text"))
        .withColumn("partition_image", imageCol)
    }

    val contentDf = datasetWithTextFile(dataset.sparkSession, $(contentPath))

    val partitionTextUDF =
      udf((text: String) =>
        partitionTextBuilder.partitionStringContent(text, $(this.headers).asJava))
    val partitionTableUDF =
      udf((text: String) =>
        partitionTableBuilder.partitionStringContent(text, $(this.headers).asJava))
    val partitionImageUDF =
      udf((text: String) =>
        partitionImageBuilder.partitionStringContent(text, $(this.headers).asJava))

    contentDf
      .withColumn("partition_text", partitionTextUDF(col("content")))
      .withColumn("partition_table", partitionTableUDF(col("content")))
      .withColumn("partition_image", partitionImageUDF(col("content")))
      .withColumn("fileName", getFileName(col("path")))
      .withColumn("exception", lit(null: String))
      .drop("content")
  }

  private def partitionBinaryContent(dataset: Dataset[_]): DataFrame = {
    val contentDf = datasetWithBinaryFile(dataset.sparkSession, $(contentPath))

    val partitionTextUDF =
      udf((input: Array[Byte]) => partitionTextBuilder.partitionBytesContent(input))
    val partitionTableUDF =
      udf((input: Array[Byte]) => partitionTableBuilder.partitionBytesContent(input))
    val partitionImageUDF =
      udf((input: Array[Byte]) => partitionImageBuilder.partitionBytesContent(input))

    contentDf
      .withColumn("partition_text", partitionTextUDF(col("content")))
      .withColumn("partition_table", partitionTableUDF(col("content")))
      .withColumn("partition_image", partitionImageUDF(col("content")))
      .withColumn("fileName", getFileName(col("path")))
      .withColumn("exception", lit(null: String))
      .drop("content")
  }

  def afterAnnotate(dataset: DataFrame): DataFrame = {
    val reader2DocOutputCol = s"${getOutputCol}_text"
    val reader2TableOutputCol = s"${getOutputCol}_table"
    val reader2ImageOutputCol = s"${getOutputCol}_image"

    if ($(explodeDocs)) {
      // helper function to explode and safely re-wrap one column
      def explodeColumn(df: DataFrame, colName: String): DataFrame = {
        if (df.columns.contains(colName)) {
          val fieldMeta = df.schema.fields
            .find(_.name == colName)
            .map(_.metadata)
            .getOrElse(new MetadataBuilder().build())
          df
            .select(
              df.columns.filterNot(_ == colName).map(col) :+ explode_outer(col(colName)).as(
                "_tmp"): _*)
            .withColumn(colName, array(col("_tmp")).as(colName, fieldMeta))
            .drop("_tmp")
        } else {
          df
        }
      }

      // Apply explode logic to all three columns if present
      val explodedDf = Seq(reader2DocOutputCol, reader2TableOutputCol, reader2ImageOutputCol)
        .foldLeft(dataset)((df, colName) => explodeColumn(df, colName))

      explodedDf
    } else {
      dataset
    }
  }

  private def partitionTextBuilder: Partition =
    buildPartition(Map("inferTableStructure" -> "false"), "partition")

  private def partitionTableBuilder: Partition =
    buildPartition(Map("inferTableStructure" -> "true"), "partition_table")

  private def partitionImageBuilder: Partition =
    buildPartition(
      Map("inferTableStructure" -> "false", "readAsImage" -> "true"),
      "partition_image")

  private def buildPartition(
      overrides: Map[String, String],
      outputCol: String,
      contentType: Option[String] = None): Partition = {
    val baseParams = Map(
      "contentType" -> (if (contentType.isDefined) contentType.get else getContentType),
      "storeContent" -> $(storeContent).toString,
      "titleFontSize" -> $(titleFontSize).toString,
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

    val finalParams = baseParams ++ overrides
    new Partition(finalParams.asJava).setOutputColumn(outputCol)
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override val outputAnnotatorType: AnnotatorType = DOCUMENT
  private val outputImageAnnotatorType: AnnotatorType = IMAGE

  private lazy val docColumnMetadata: Metadata = {
    new MetadataBuilder().putString("annotatorType", outputAnnotatorType).build()
  }

  private lazy val tableColumnMetadata: Metadata = {
    new MetadataBuilder().putString("annotatorType", outputAnnotatorType).build()
  }

  private lazy val imageColumnMetadata: Metadata = {
    new MetadataBuilder().putString("annotatorType", outputImageAnnotatorType).build()
  }

  private def wrapDocColumn(col: Column, outputCol: String): Column = {
    col.as(outputCol, docColumnMetadata)
  }

  private def wrapTableColumn(col: Column, outputCol: String): Column = {
    col.as(outputCol, tableColumnMetadata)
  }

  private def wrapImageColumn(col: Column, outputCol: String): Column = {
    col.as(outputCol, imageColumnMetadata)
  }

  protected def validateRequiredParameters(): Unit = {
    val hasContentPath = $(contentPath) != null && $(contentPath).trim.nonEmpty
    if (hasContentPath) {
      require(
        ResourceHelper.validFile($(contentPath)),
        "contentPath must point to a valid file or directory")
    }
  }

  override def transformSchema(schema: StructType): StructType = {
    val reader2DocOutputCol = s"${getOutputCol}_text"
    val reader2TableOutputCol = s"${getOutputCol}_table"
    val reader2ImageOutputCol = s"${getOutputCol}_image"

    val outputFields = Seq(
      StructField(
        reader2DocOutputCol,
        ArrayType(Annotation.dataType),
        nullable = false,
        docColumnMetadata),
      StructField(
        reader2TableOutputCol,
        ArrayType(Annotation.dataType),
        nullable = false,
        tableColumnMetadata),
      StructField(
        reader2ImageOutputCol,
        ArrayType(AnnotationImage.dataType),
        nullable = false,
        imageColumnMetadata))

    StructType(schema.fields ++ outputFields)
  }

  private def partitionMixedContentDir(dataset: Dataset[_]): DataFrame = {

    val allFiles = listAllFilesRecursively(new File($(contentPath))).map(_.toString)
    if (allFiles.isEmpty) {
      return buildEmptyDataFrame(dataset)
    }

    val grouped = allFiles
      .groupBy { path =>
        val ext = path.substring(path.lastIndexOf('.') + 1).toLowerCase
        if (supportedTypes.contains(ext)) Some(ext)
        else if (! $(ignoreExceptions)) Some(s"__unsupported__$ext")
        else None
      }
      .collect { case (Some(ext), files) => ext -> files }

    if (grouped.isEmpty) {
      return buildEmptyDataFrame(dataset)
    }

    val dfs = grouped.flatMap {
      // Unsupported types → build error DataFrames
      case (ext, files) if ext.startsWith("__unsupported__") =>
        val badExt = ext.stripPrefix("__unsupported__")
        val dfs = files.map(path => buildErrorDataFrame(dataset, path, badExt))
        Some(dfs.reduce(_.unionByName(_, allowMissingColumns = true)))

      case (ext, files) if supportedTypes(ext)._1 == "text/csv" =>
        Some(processCsvFiles(dataset, files, ext))

      case (ext, files) if Seq("png", "jpg", "jpeg", "bmp", "gif").contains(ext) =>
        Some(processImageFiles(dataset, files, ext))

      // Default case → text or binary partitioning
      case (ext, files) =>
        val (mimeType, isText) = supportedTypes(ext)
        Some(processGenericFiles(dataset, files, mimeType, isText))
    }.toSeq

    if (dfs.isEmpty) buildEmptyDataFrame(dataset)
    else dfs.reduce(_.unionByName(_, allowMissingColumns = true))
  }

  private def processGenericFiles(
      dataset: Dataset[_],
      files: Seq[String],
      mimeType: String,
      isText: Boolean): DataFrame = {

    val spark = dataset.sparkSession
    val pathsStr = files.mkString(",")

    val partitionTextBuilder =
      buildPartition(
        Map("inferTableStructure" -> "false", "contentType" -> mimeType),
        "partition_text")
    val partitionTableBuilder =
      buildPartition(
        Map("inferTableStructure" -> "true", "contentType" -> mimeType),
        "partition_table")
    val partitionImageBuilder =
      buildPartition(
        Map("inferTableStructure" -> "false", "readAsImage" -> "true", "contentType" -> mimeType),
        "partition_image")

    val baseDf =
      if (isText) datasetWithTextFile(spark, pathsStr)
      else datasetWithBinaryFile(spark, pathsStr)

    val textUdf =
      if (isText)
        udf((content: String) =>
          partitionTextBuilder.partitionStringContent(content, $(this.headers).asJava))
      else udf((bytes: Array[Byte]) => partitionTextBuilder.partitionBytesContent(bytes))

    val tableUdf =
      if (isText)
        udf((content: String) =>
          partitionTableBuilder.partitionStringContent(content, $(this.headers).asJava))
      else udf((bytes: Array[Byte]) => partitionTableBuilder.partitionBytesContent(bytes))

    val imageUdf =
      if (isText)
        udf((content: String) =>
          partitionImageBuilder.partitionStringContent(content, $(this.headers).asJava))
      else udf((bytes: Array[Byte]) => partitionImageBuilder.partitionBytesContent(bytes))

    val df = baseDf
      .withColumn("partition_text", textUdf(col("content")))
      .withColumn("partition_table", tableUdf(col("content")))
      .withColumn("partition_image", imageUdf(col("content")))
      .withColumn("fileName", getFileName(col("path")))
      .withColumn("exception", lit(null: String))
      .drop("content")

    if ($(ignoreExceptions)) df.filter(col("exception").isNull) else df
  }

  private def processCsvFiles(dataset: Dataset[_], files: Seq[String], ext: String): DataFrame = {
    val pathsStr = files.mkString(",")
    val mimeType = supportedTypes(ext)._1
    val partitionTableBuilder =
      buildPartition(
        Map("inferTableStructure" -> "true", "contentType" -> mimeType),
        "partition_table")

    val imageElement = Seq(
      HTMLElement(ElementType.ERROR, s"Could not parse image", mutable.Map()))
    val imageCol = typedLit(imageElement)

    val csvDf = partitionContentFromPath(partitionTableBuilder, pathsStr, isText = true, dataset)
      .withColumnRenamed("partition", "partition_text")
      .withColumn("partition_table", col("partition_text"))
      .withColumn("partition_image", imageCol)
      .withColumn("fileName", getFileName(col("path")))
      .withColumn("exception", lit(null: String))

    if ($(ignoreExceptions)) csvDf.filter(col("exception").isNull) else csvDf
  }

  private def processImageFiles(
      dataset: Dataset[_],
      files: Seq[String],
      ext: String): DataFrame = {
    val spark = dataset.sparkSession
    val pathsStr = files.mkString(",")
    val binaryDf = datasetWithBinaryFile(spark, pathsStr)
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
      .withColumn("partition_image", imageUDF(col("content")))
      .withColumn("partition_text", typedLit(Seq.empty[HTMLElement]))
      .withColumn("partition_table", typedLit(Seq.empty[HTMLElement]))
      .withColumn("fileName", getFileName(col("path")))
      .withColumn("exception", lit(null: String))
      .drop("content")
  }

  /** Process in-memory string inputs from a dataset column instead of file paths. This is similar
    * to partitionContentFromDataFrame but handles text/table/image partitions.
    */
  private def processStringInputFromDatasetV2(dataset: Dataset[_]): DataFrame = {
    val mimeType = getContentType
    val partitionTextBuilder =
      buildPartition(
        Map("inferTableStructure" -> "false", "contentType" -> mimeType),
        "partition_text")
    val partitionTableBuilder =
      buildPartition(
        Map("inferTableStructure" -> "true", "contentType" -> mimeType),
        "partition_table")

    val textUdf =
      udf((text: String) =>
        partitionTextBuilder.partitionStringContent(text, $(this.headers).asJava))
    val tableUdf =
      udf((text: String) =>
        partitionTableBuilder.partitionStringContent(text, $(this.headers).asJava))

    val emptyImageArray = typedLit(Seq.empty[HTMLElement])

    dataset
      .withColumn("partition_text", textUdf(col(getInputCol)))
      .withColumn("partition_table", tableUdf(col(getInputCol)))
      .withColumn("partition_image", emptyImageArray)
      .withColumn("fileName", lit(null: String))
      .withColumn("exception", lit(null: String))
  }

  /** Handles cases where input is already in a column (string-based only). Supports text, tables,
    * and embedded base64 images.
    */
  private def processStringInputFromDataset(dataset: Dataset[_]): DataFrame = {

    val mimeType = getContentType

    val partitionTextBuilder =
      buildPartition(
        Map("inferTableStructure" -> "false", "contentType" -> mimeType),
        "partition_text")
    val partitionTableBuilder =
      buildPartition(
        Map("inferTableStructure" -> "true", "contentType" -> mimeType),
        "partition_table")
    val partitionImageBuilder =
      buildPartition(
        Map("inferTableStructure" -> "false", "readAsImage" -> "true", "contentType" -> mimeType),
        "partition_image")

    val textUdf =
      udf((text: String) =>
        partitionTextBuilder.partitionStringContent(text, $(this.headers).asJava))

    val tableUdf =
      udf((text: String) =>
        partitionTableBuilder.partitionStringContent(text, $(this.headers).asJava))

    val imageUdf =
      udf((text: String) =>
        partitionImageBuilder.partitionStringContent(text, $(this.headers).asJava))

    dataset
      .withColumn("partition_text", textUdf(col(getInputCol)))
      .withColumn("partition_table", tableUdf(col(getInputCol)))
      .withColumn("partition_image", imageUdf(col(getInputCol)))
      .withColumn("fileName", lit(null: String))
      .withColumn("exception", lit(null: String))
  }

  override def buildErrorDataFrame(
      dataset: Dataset[_],
      filePath: String,
      badExt: String): DataFrame = {
    val spark = dataset.sparkSession
    import spark.implicits._

    val errorElement = Seq(
      HTMLElement(ElementType.ERROR, s"Unsupported file type: .$badExt", mutable.Map()))
    val errorCol = typedLit(errorElement)

    Seq(filePath)
      .toDF("path")
      .withColumn("fileName", getFileName(col("path")))
      .withColumn("partition_text", errorCol)
      .withColumn("partition_table", errorCol)
      .withColumn("partition_image", errorCol)
      .withColumn("exception", lit(s"Unsupported file type: .$badExt"))
      .select(
        "path",
        "partition_text",
        "partition_table",
        "partition_image",
        "fileName",
        "exception")
  }

  private lazy val reader2Doc: Reader2Doc = new Reader2Doc()
    .setContentType($(contentType))
    .setContentPath($(contentPath))
    .setExplodeDocs($(explodeDocs))
    .setExcludeNonText(true)
    .setAddAttachmentContent($(addAttachmentContent))
    .setCellSeparator($(cellSeparator))
    .setAppendCells($(appendCells))
    .setTitleThreshold($(titleThreshold))
    .setIncludeSlideNotes($(includeSlideNotes))
    .setTitleLengthSize($(titleLengthSize))
    .setGroupBrokenParagraphs($(groupBrokenParagraphs))
    .setParagraphSplit($(paragraphSplit))
    .setShortLineWordThreshold($(shortLineWordThreshold))
    .setMaxLineCount($(maxLineCount))
    .setThreshold($(threshold))
    .setOutputCol(reader2DocOutputCol)

  private lazy val reader2Table: Reader2Table = new Reader2Table()
    .setContentType($(contentType))
    .setContentPath($(contentPath))
    .setExplodeDocs($(explodeDocs))
    .setAddAttachmentContent($(addAttachmentContent))
    .setCellSeparator($(cellSeparator))
    .setAppendCells($(appendCells))
    .setTitleThreshold($(titleThreshold))
    .setIncludeSlideNotes($(includeSlideNotes))
    .setTitleLengthSize($(titleLengthSize))
    .setGroupBrokenParagraphs($(groupBrokenParagraphs))
    .setParagraphSplit($(paragraphSplit))
    .setShortLineWordThreshold($(shortLineWordThreshold))
    .setMaxLineCount($(maxLineCount))
    .setThreshold($(threshold))
    .setOutputCol(reader2TableOutputCol)

  private lazy val reader2Image: Reader2Image = new Reader2Image()
    .setContentType($(contentType))
    .setContentPath($(contentPath))
    .setExplodeDocs($(explodeDocs))
    .setAddAttachmentContent($(addAttachmentContent))
    .setCellSeparator($(cellSeparator))
    .setAppendCells($(appendCells))
    .setTitleThreshold($(titleThreshold))
    .setOutputCol(reader2ImageOutputCol)

}
