package com.johnsnowlabs.reader

import com.johnsnowlabs.partition.util.PartitionHelper.{
  datasetWithBinaryFile,
  datasetWithTextFile
}
import com.johnsnowlabs.partition.{HasReaderProperties, Partition}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import java.io.File
import scala.jdk.CollectionConverters.mapAsJavaMapConverter

trait HasReaderContent extends HasReaderProperties {

  val supportedTypes: Map[String, (String, Boolean)] = Map(
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
    "msg" -> ("message/rfc822", false))

  def buildErrorDataFrame(dataset: Dataset[_], contentPath: String, ext: String): DataFrame = {
    val fileName = retrieveFileName(contentPath)
    val errorMessage = s"File type .$ext not supported"

    val errorPartition = HTMLElement(
      elementType = ElementType.ERROR,
      content = errorMessage,
      metadata = scala.collection.mutable.Map[String, String](),
      binaryContent = None)

    val spark = dataset.sparkSession
    import spark.implicits._

    val errorArray = Seq((contentPath, Seq(errorPartition), fileName, errorMessage))
    errorArray
      .toDF("path", "partition", "fileName", "exception")
  }

  def retrieveFileName(path: String): String = if (path != null) path.split("/").last else ""

  def partitionMixedContent(
      dataset: Dataset[_],
      dirPath: String,
      partitionParams: Map[String, String]): DataFrame = {

    val allFiles = listAllFilesRecursively(new File(dirPath))

    val grouped = allFiles
      .filter(_.isFile)
      .groupBy { file =>
        val ext = file.getName.split("\\.").lastOption.getOrElse("").toLowerCase
        if (supportedTypes.contains(ext)) {
          Some(ext)
        } else if (! $(ignoreExceptions)) {
          Some(s"__unsupported__$ext")
        } else {
          None
        }
      }
      .collect { case (Some(ext), files) => ext -> files }

    if (grouped.isEmpty) {
      return buildEmptyDataFrame(dataset)
    }

    val mixedDfs = grouped.flatMap { case (ext, files) =>
      if (ext.startsWith("__unsupported__")) {
        val badExt = ext.stripPrefix("__unsupported__")
        val dfs = files.map(file => buildErrorDataFrame(dataset, file.getAbsolutePath, badExt))
        Some(dfs.reduce(_.unionByName(_, allowMissingColumns = true)))
      } else {
        val (contentType, isText) = supportedTypes(ext)
        val filePartitionParam = Map("contentType" -> contentType) ++ partitionParams
        val partition = new Partition(filePartitionParam.asJava)

        val filePathsStr = files.map(_.getAbsolutePath).mkString(",")
        if (filePathsStr.nonEmpty) {
          val partitionDf = partitionContent(partition, filePathsStr, isText, dataset)
          Some(
            if ($(ignoreExceptions)) partitionDf.filter(col("exception").isNull)
            else partitionDf)
        } else None
      }
    }.toSeq

    if (mixedDfs.isEmpty) {
      buildEmptyDataFrame(dataset)
    } else {
      mixedDfs.reduce(_.unionByName(_, allowMissingColumns = true))
    }
  }

  def partitionContent(
      partition: Partition,
      contentPath: String,
      isText: Boolean,
      dataset: Dataset[_]): DataFrame = {

    val ext = contentPath.split("\\.").lastOption.getOrElse("").toLowerCase
    if (! $(ignoreExceptions) && !supportedTypes.contains(ext)) {
      return buildErrorDataFrame(dataset, contentPath, ext)
    }

    val partitionDf = if (isText) {
      val stringContentDF = if ($(contentType) == "text/csv" || ext == "csv") {
        partition.setOutputColumn("csv")
        partition
          .partition(contentPath)
          .withColumnRenamed(partition.getOutputColumn, "partition")
      } else {
        val partitionUDF =
          udf((text: String) => partition.partitionStringContent(text, $(this.headers).asJava))
        datasetWithTextFile(dataset.sparkSession, contentPath)
          .withColumn(partition.getOutputColumn, partitionUDF(col("content")))
      }
      stringContentDF
        .withColumn("fileName", getFileName(col("path")))
        .withColumn("exception", lit(null: String))
        .drop("content")
    } else {
      val binaryContentDF = datasetWithBinaryFile(dataset.sparkSession, contentPath)
      val partitionUDF =
        udf((input: Array[Byte]) => partition.partitionBytesContent(input))

      binaryContentDF
        .withColumn(partition.getOutputColumn, partitionUDF(col("content")))
        .withColumn("fileName", getFileName(col("path")))
        .withColumn("exception", lit(null: String))
        .drop("content")
    }

    if ($(ignoreExceptions)) {
      partitionDf.filter(col("exception").isNull)
    } else partitionDf
  }

  val getFileName: UserDefinedFunction = udf { path: String =>
    if (path != null) path.split("/").last else ""
  }

  private def listAllFilesRecursively(dir: File): Seq[File] = {
    val these = Option(dir.listFiles).getOrElse(Array.empty)
    these.filter(_.isFile) ++ these.filter(_.isDirectory).flatMap(listAllFilesRecursively)
  }

  private def buildEmptyDataFrame(dataset: Dataset[_]): DataFrame = {
    val schema = StructType(
      Seq(
        StructField("partition", StringType, nullable = true),
        StructField("fileName", StringType, nullable = true)))
    val emptyRDD = dataset.sparkSession.sparkContext.emptyRDD[Row]
    dataset.sparkSession.createDataFrame(emptyRDD, schema)
  }

}
