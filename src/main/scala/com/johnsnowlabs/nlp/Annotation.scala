/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.nlp

import org.apache.spark.sql.api.java.UDF1
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Dataset, Row}

import scala.collection.{Map, mutable}

/** represents annotator's output parts and their details
  *
  * @param annotatorType
  *   the type of annotation
  * @param begin
  *   the index of the first character under this annotation
  * @param end
  *   the index after the last character under this annotation
  * @param metadata
  *   associated metadata for this annotation
  */
case class Annotation(
    annotatorType: String,
    begin: Int,
    end: Int,
    result: String,
    metadata: Map[String, String],
    embeddings: Array[Float] = Array.emptyFloatArray)
    extends IAnnotation {

  override def equals(obj: Any): Boolean = {
    obj match {
      case annotation: Annotation =>
        this.annotatorType == annotation.annotatorType &&
        this.begin == annotation.begin &&
        this.end == annotation.end &&
        this.result == annotation.result &&
        this.metadata == annotation.metadata &&
        this.embeddings.sameElements(annotation.embeddings)
      case _ => false
    }
  }

  override def toString: String = {
    s"Annotation(type: $annotatorType, begin: $begin, end: $end, result: $result)"
  }

  def getAnnotatorType: String = {
    annotatorType
  }

  def getBegin: Int = {
    begin
  }

  def getEnd: Int = {
    end
  }

  def getResult: String = {
    result
  }

  def getMetadata: Map[String, String] = {
    metadata
  }

}

case class JavaAnnotation(
    annotatorType: String,
    begin: Int,
    end: Int,
    result: String,
    metadata: java.util.Map[String, String],
    embeddings: Array[Float] = Array.emptyFloatArray)
    extends IAnnotation {

  def getAnnotatorType: String = {
    annotatorType
  }

  def getBegin: Int = {
    begin
  }

  def getEnd: Int = {
    end
  }

  def getResult: String = {
    result
  }

  def getMetadata: java.util.Map[String, String] = {
    metadata
  }

}

object Annotation {

  case class AnnotationContainer(__annotation: Array[Annotation])

  object extractors {

    /** annotation container ready for extraction */
    protected class AnnotationData(dataset: Dataset[Row]) {
      def collect(column: String): Array[Array[Annotation]] = {
        Annotation.collect(dataset, column)
      }

      def take(column: String, howMany: Int): Array[Array[Annotation]] = {
        Annotation.take(dataset, column, howMany)
      }
    }

    implicit def data2andata(dataset: Dataset[Row]): AnnotationData = new AnnotationData(dataset)
  }

  private val ANNOTATION_NAME = "__annotation"
  private val RESULT = "result"
  private val EMBEDDINGS = "embeddings"

  /** This is spark type of an annotation representing its metadata shape */
  val dataType = new StructType(
    Array(
      StructField("annotatorType", StringType, nullable = true),
      StructField("begin", IntegerType, nullable = false),
      StructField("end", IntegerType, nullable = false),
      StructField("result", StringType, nullable = true),
      StructField("metadata", MapType(StringType, StringType), nullable = true),
      StructField(EMBEDDINGS, ArrayType(FloatType, false), true)))

  val arrayType = new ArrayType(dataType, true)

  /** This method converts a [[org.apache.spark.sql.Row]] into an [[Annotation]]
    *
    * @param row
    *   spark row to be converted
    * @return
    *   annotation
    */
  def apply(row: Row): Annotation = {
    Annotation(
      row.getString(0),
      row.getInt(1),
      row.getInt(2),
      row.getString(3),
      row.getMap[String, String](4),
      row.getSeq[Float](5).toArray)
  }

  def apply(rawText: String): Annotation =
    Annotation(
      AnnotatorType.DOCUMENT,
      0,
      rawText.length - 1,
      rawText,
      Map.empty[String, String],
      Array.emptyFloatArray)

  /** dataframe collect of a specific annotation column */
  def collect(dataset: Dataset[Row], column: String): Array[Array[Annotation]] = {
    require(dataset.columns.contains(column), s"column $column not present in data")
    import dataset.sparkSession.implicits._
    dataset
      .withColumnRenamed(column, ANNOTATION_NAME)
      .select(ANNOTATION_NAME)
      .as[AnnotationContainer]
      .map(_.__annotation)
      .collect()
  }

  def collect(
      dataset: Dataset[Row],
      column: String,
      columns: String*): Array[Array[Annotation]] = {

    dataset
      .select(column, columns: _*)
      .collect()
      .map { row =>
        (0 to columns.length)
          .flatMap(idx => getAnnotations(row, idx))
          .toArray
      }
  }

  def getAnnotations(row: Row, colNum: Int): Seq[Annotation] = {
    row.getAs[mutable.Seq[Row]](colNum).map(obj => Annotation(obj)).toSeq
  }

  def getAnnotations(row: Row, colName: String): Seq[Annotation] = {
    row.getAs[mutable.Seq[Row]](colName).map(obj => Annotation(obj)).toSeq
  }

  /** dataframe take of a specific annotation column */
  def take(dataset: Dataset[Row], column: String, howMany: Int): Array[Array[Annotation]] = {
    require(dataset.columns.contains(column), s"column $column not present in data")
    import dataset.sparkSession.implicits._
    dataset
      .withColumnRenamed(column, ANNOTATION_NAME)
      .select(ANNOTATION_NAME)
      .as[AnnotationContainer]
      .map(_.__annotation)
      .take(howMany)
  }

  private def annotationResult(
      row: Row,
      valueSeparator: String,
      parseEmbeddings: Boolean): String = {
    row.getString(0) match {
      case (AnnotatorType.WORD_EMBEDDINGS | AnnotatorType.SENTENCE_EMBEDDINGS)
          if parseEmbeddings =>
        row.getSeq[Float](5).mkString(valueSeparator)
      case _ => row.getString(3)
    }
  }

  private def flattenAnnotationRows(
      annotations: scala.collection.Seq[Row],
      valueSeparator: String,
      annotationSeparator: String,
      parseEmbeddings: Boolean): String = {
    Option(annotations)
      .map(
        _.map(row => annotationResult(row, valueSeparator, parseEmbeddings)).mkString(
          annotationSeparator))
      .getOrElse("")
  }

  private def flattenAnnotationRowsDetail(
      annotations: scala.collection.Seq[Row],
      valueSeparator: String,
      annotationSeparator: String,
      parseEmbeddings: Boolean): String = {
    Option(annotations)
      .map(_.map { row =>
        val metadataWithResult = row.getMap[String, String](4) ++ Map(RESULT -> row.getString(3))
        val metadata = row.getString(0) match {
          case (AnnotatorType.WORD_EMBEDDINGS | AnnotatorType.SENTENCE_EMBEDDINGS)
              if parseEmbeddings =>
            metadataWithResult ++ Map(EMBEDDINGS -> row.getSeq[Float](5).mkString(valueSeparator))
          case _ => metadataWithResult
        }
        metadata.mkString(valueSeparator).replace(" -> ", "->")
      }.mkString(annotationSeparator))
      .getOrElse("")
  }

  private def flattenAnnotationRowsAsArray(
      annotations: scala.collection.Seq[Row],
      parseEmbeddings: Boolean): scala.collection.Seq[String] = {
    Option(annotations)
      .map(_.map(row => annotationResult(row, " ", parseEmbeddings)))
      .getOrElse(Seq.empty[String])
  }

  private def flattenAnnotationRowsMetadata(
      annotations: scala.collection.Seq[Row]): Map[String, String] = {
    Option(annotations).map(_.flatMap(_.getMap[String, String](4)).toMap).getOrElse(Map.empty)
  }

  /** dataframe annotation flatmap of results into strings */
  def flatten(vSep: String, aSep: String, parseEmbeddings: Boolean): UserDefinedFunction = {
    val func = new UDF1[scala.collection.Seq[Row], String] {
      override def call(annotations: scala.collection.Seq[Row]): String =
        flattenAnnotationRows(annotations, vSep, aSep, parseEmbeddings)
    }
    udf(func, StringType)
  }

  /** dataframe annotation flatmap of results and metadata key values into strings */
  def flattenDetail(vSep: String, aSep: String, parseEmbeddings: Boolean): UserDefinedFunction = {
    val func = new UDF1[scala.collection.Seq[Row], String] {
      override def call(annotations: scala.collection.Seq[Row]): String =
        flattenAnnotationRowsDetail(annotations, vSep, aSep, parseEmbeddings)
    }
    udf(func, StringType)
  }

  /** dataframe annotation flatmap of result values as ArrayType */
  def flattenArray(parseEmbeddings: Boolean): UserDefinedFunction = {
    val func = new UDF1[scala.collection.Seq[Row], scala.collection.Seq[String]] {
      override def call(annotations: scala.collection.Seq[Row]): scala.collection.Seq[String] =
        flattenAnnotationRowsAsArray(annotations, parseEmbeddings)
    }
    udf(func, ArrayType(StringType))
  }

  /** dataframe annotation flatmap of metadata values as ArrayType */
  def flattenArrayMetadata: UserDefinedFunction = {
    val func = new UDF1[scala.collection.Seq[Row], Map[String, String]] {
      override def call(annotations: scala.collection.Seq[Row]): Map[String, String] =
        flattenAnnotationRowsMetadata(annotations)
    }
    udf(func, MapType(StringType, StringType))
  }

  private def isInside(a: Annotation, begin: Int, end: Int): Boolean = {
    a.begin >= begin && a.end <= end
  }

  private def searchLabel(
      annotations: Array[Annotation],
      l: Int,
      r: Int,
      begin: Int,
      end: Int): Seq[Annotation] = {

    def getAnswers(ind: Int) = {
      val suitable =
        if (isInside(annotations(ind), begin, end))
          annotations.toList.drop(ind)
        else
          annotations.toList.drop(ind + 1)

      suitable.takeWhile(a => isInside(a, begin, end))
    }

    val k = (l + r) / 2

    if (l >= r)
      getAnswers(l)
    else if (begin < annotations(k).begin)
      searchLabel(annotations, l, k - 1, begin, end)
    else if (begin > annotations(k).begin)
      searchLabel(annotations, k + 1, r, begin, end)
    else
      getAnswers(k)
  }

  /*
    Returns Annotations that coverages text segment from begin till end (inclusive)
   */
  def searchCoverage(annotations: Array[Annotation], begin: Int, end: Int): Seq[Annotation] = {
    searchLabel(annotations, 0, annotations.length - 1, begin, end)
  }

  def getColumnByType(
      dataset: Dataset[_],
      inputCols: Array[String],
      annotatorType: String): StructField = {
    dataset.schema.fields
      .find(field =>
        inputCols.contains(field.name) &&
          field.metadata.contains("annotatorType") &&
          field.metadata.getString("annotatorType") == annotatorType)
      .getOrElse(throw new IllegalArgumentException(
        s"Could not find a column of type $annotatorType in inputCols"))
  }

}
