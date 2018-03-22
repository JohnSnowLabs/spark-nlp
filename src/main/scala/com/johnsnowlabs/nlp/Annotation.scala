package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotators.common.TokenizedWithSentence
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.udf

import scala.collection.Map

/**
  * represents annotator's output parts and their details
  * @param annotatorType the type of annotation
  * @param begin the index of the first character under this annotation
  * @param end the index after the last character under this annotation
  * @param metadata associated metadata for this annotation
  */
case class Annotation(annotatorType: String, begin: Int, end: Int, result: String, metadata: Map[String, String])

object Annotation {

  case class AnnotationContainer(__annotation: Array[Annotation])

  object extractors {
    /** annotation container ready for extraction */
    protected class AnnotationData(dataset: Dataset[Row]){
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
  val RESULT = "result"

  /** This is spark type of an annotation representing its metadata shape */
  val dataType = new StructType(Array(
    StructField("annotatorType", StringType, nullable = true),
    StructField("begin", IntegerType, nullable = false),
    StructField("end", IntegerType, nullable = false),
    StructField("result", StringType, nullable = true),
    StructField("metadata", MapType(StringType, StringType), nullable = true)
  ))


  /**
    * This method converts a [[org.apache.spark.sql.Row]] into an [[Annotation]]
    * @param row spark row to be converted
    * @return annotation
    */
  def apply(row: Row): Annotation = {
    Annotation(
      row.getString(0),
      row.getInt(1),
      row.getInt(2),
      row.getString(3),
      row.getMap[String, String](4)
    )
  }
  def apply(rawText: String): Annotation = Annotation(
    AnnotatorType.DOCUMENT,
    0,
    rawText.length,
    rawText,
    Map.empty[String, String]
  )

  /** dataframe collect of a specific annotation column*/
  def collect(dataset: Dataset[Row], column: String): Array[Array[Annotation]] = {
    require(dataset.columns.contains(column), s"column $column not present in data")
    import dataset.sparkSession.implicits._
    dataset
      .withColumnRenamed(column, ANNOTATION_NAME)
      .select(ANNOTATION_NAME)
      .as[AnnotationContainer]
      .map(_.__annotation)
      .collect
  }

  def collect(dataset: Dataset[Row], column: String, columns: String*): Array[Array[Annotation]] = {

    dataset
      .select(column, columns :_*)
      .collect()
      .map { row =>
        (0 to columns.length)
          .flatMap(idx => getAnnotations(row, idx))
          .toArray
      }
  }

  protected def getAnnotations(row: Row, colNum: Int): Seq[Annotation] = {
    row.getAs[Seq[Row]](colNum).map(obj => Annotation(obj))
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

  /** dataframe annotation flatmap of metadata values */
  def flatten(vSep: String, aSep: String): UserDefinedFunction = {
    udf {
      (annotations: Seq[Row]) => annotations.map(r =>
        r.getString(3)
      ).mkString(aSep)
    }
  }

  /** dataframe annotation flatmap of metadata key values */
  def flattenKV(vSep: String, aSep: String): UserDefinedFunction = {
    udf {
      (annotations: Seq[Row]) => annotations.map(r =>
        (r.getMap[String, String](4) ++ Map(RESULT -> r.getString(3))).mkString(vSep).replace(" -> ", "->")
      ).mkString(aSep)
    }
  }

  /** dataframe annotation flatmap of metadata values as ArrayType */
  def flattenArray: UserDefinedFunction = {
    udf {
      (annotations: Seq[Row]) => annotations.map(r =>
        r.getString(3)
      )
    }
  }

  private def isInside(a: Annotation, begin: Int, end: Int): Boolean = {
    a.begin >= begin && a.end <= end
  }

  private def searchLabel(annotations: Array[Annotation], l: Int, r: Int, begin: Int, end: Int): Seq[Annotation] = {

    def getAnswers(ind: Int) = {
      val suitable = if (isInside(annotations(ind), begin, end))
        annotations.toList.drop(ind)
      else
        annotations.toList.drop(ind + 1)

      suitable.takeWhile(a => isInside(a, begin, end))
    }

    val k = (l + r) / 2

    if (l  >= r)
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

}