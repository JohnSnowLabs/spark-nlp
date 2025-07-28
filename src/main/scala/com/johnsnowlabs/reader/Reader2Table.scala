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

import com.johnsnowlabs.nlp.Annotation
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/** The Reader2Table annotator allows you to use the reading files more smoothly within existing
  * Spark NLP workflows, enabling seamless reuse of your pipelines. Reader2Doc can be used for
  * extracting structured content from various document types using Spark NLP readers. It supports
  * reading from many files types and returns parsed output as a structured Spark DataFrame.
  *
  * Supported formats include plain text, HTML, Word (.doc/.docx), Excel (.xls/.xlsx), PowerPoint
  * (.ppt/.pptx)
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.reader.Reader2Table
  * import com. johnsnowlabs.nlp.base.DocumentAssembler
  * import org.apache.spark.ml.Pipeline
  *
  * val reader2Table = new Reader2Table()
  *   .setContentType("application/csv")
  *   .setContentPath(s"$pdfDirectory/")
  *
  * val pipeline = new Pipeline()
  *   .setStages(Array(reader2Table))
  *
  * val pipelineModel = pipeline.fit(emptyDataSet)
  * val resultDf = pipelineModel.transform(emptyDataSet)
  *
  * resultDf.show()
  * +----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |fileName        |document                                                                                                                                                                                    |
  * +----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |stanley-cups.csv|[{document, 0, 137, {"caption":"","header":[],"rows":[["Team","Location","Stanley Cups"],["Blues","STL","1"],["Flyers","PHI","2"],["Maple Leafs","TOR","13"]]}, {elementType -> Table}, []}]|
  * +----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * }}}
  */
class Reader2Table(override val uid: String) extends Reader2Doc {

  def this() = this(Identifiable.randomUID("Reader2Table"))

  setDefault(outputFormat -> "json-table", inferTableStructure -> true)

  override def transform(dataset: Dataset[_]): DataFrame = {
    super.transform(dataset)
  }

  override def partitionToAnnotation(flatten: Boolean): UserDefinedFunction = udf {
    (partitions: Seq[Row]) =>
      if (partitions == null) Nil
      else {
        val outputFormatValue = $(outputFormat)
        val asDocument = $(outputAsDocument)
        val acceptedTypes = getAcceptedTypes
        val elements = partitions.flatMap { part =>
          val elementType = part.getAs[String]("elementType")
          if (acceptedTypes.contains(elementType))
            Some((elementType, part.getAs[String]("content")))
          else None
        }
        if (asDocument)
          mergeElementsAsDocument(elements, outputFormatValue)
        else
          elementsAsIndividualAnnotations(partitions, flatten, acceptedTypes)
      }
  }

  private def getAcceptedTypes: Set[String] = {
    val officeDocTypes = Set(
      "application/msword",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "application/vnd.ms-excel",
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      "application/vnd.ms-powerpoint",
      "application/vnd.openxmlformats-officedocument.presentationml.presentation")

    if (officeDocTypes.contains($(contentType))) {
      Set(ElementType.HTML, ElementType.JSON)
    } else {
      Set(ElementType.TABLE)
    }
  }

  private def mergeElementsAsDocument(
      elements: Seq[(String, String)],
      outputFormatValue: String): Seq[Annotation] = {
    val mergedContent = outputFormatValue match {
      case "json-table" =>
        val jsons = elements.collect {
          case (element, content)
              if element == ElementType.JSON || element == ElementType.TABLE =>
            content
        }
        s"[${jsons.mkString(",")}]"
      case "html-table" =>
        val htmls = elements.collect {
          case (element, content)
              if element == ElementType.HTML || element == ElementType.TABLE =>
            content
        }
        s"""<div class="tables-group">${htmls.mkString(" ")}</div>"""
    }
    val meta = Map("elementType" -> "TableGroup")
    Seq(
      Annotation(
        annotatorType = outputAnnotatorType,
        begin = 0,
        end = if (mergedContent.isEmpty) 0 else mergedContent.length - 1,
        result = mergedContent,
        metadata = meta,
        embeddings = Array.emptyFloatArray))
  }

  private def elementsAsIndividualAnnotations(
      partitions: Seq[Row],
      flatten: Boolean,
      acceptedTypes: Set[String]): Seq[Annotation] = {
    var currentOffset = 0
    partitions.flatMap { part =>
      val elementType = part.getAs[String]("elementType")
      if (acceptedTypes.contains(elementType)) {
        val content = part.getAs[String]("content")
        val metadata = part.getAs[Map[String, String]]("metadata")
        val begin = currentOffset
        val end = currentOffset + (if (content != null) content.length else 0) - 1
        currentOffset = end + 1

        val baseMeta = if (metadata != null) metadata else Map.empty[String, String]
        val withExtras = baseMeta + ("elementType" -> elementType)
        val finalMeta = if (flatten) withExtras.filterKeys(_ == "sentence") else withExtras

        Some(
          Annotation(
            annotatorType = outputAnnotatorType,
            begin = begin,
            end = end,
            result = content,
            metadata = finalMeta,
            embeddings = Array.emptyFloatArray))
      } else None
    }
  }

  override def validateRequiredParameters(): Unit = {
    require(
      $(contentPath) != null && $(contentPath).trim.nonEmpty,
      "contentPath must be set and not empty")
    require(
      $(contentType) != null && $(contentType).trim.nonEmpty,
      "contentType must be set and not empty")
    require(
      Set("html-table", "json-table").contains($(outputFormat)),
      "outputFormat must be either 'html-table' or 'json-table'.")
    require($(inferTableStructure), "inferTableStructure must be set to true for Reader2Table.")
  }

}

object Reader2Table extends DefaultParamsReadable[Reader2Table]
