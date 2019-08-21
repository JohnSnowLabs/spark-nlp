package com.johnsnowlabs.nlp.annotators.ocr

import com.johnsnowlabs.nlp.{Annotation, ParamsAndFeaturesReadable, RawAnnotator}
import com.johnsnowlabs.nlp.AnnotatorType.CHUNK
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.ArrayType
import com.johnsnowlabs.nlp.annotators.ocr.schema._

import scala.collection.mutable.ArrayBuffer

class PositionFinder(override val uid: String) extends RawAnnotator[PositionFinder] {

  override val outputAnnotatorType: AnnotatorType = CHUNK

  /** A Tokenizer could require only for now a SentenceDetector annotator */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array[AnnotatorType](CHUNK)

  def this() = this(Identifiable.randomUID("POSITION_FINDER"))

  val pageMatrixCol: Param[String] = new Param(this, "pageMatrixCol", "Column name for Page Matrix schema")
  val matchingWindow: IntParam = new IntParam(this, "matchingWindow", "Textual range to match in context, applies in both direction")
  val windowPageTolerance: BooleanParam = new BooleanParam(this, "windowPageTolerance", "whether or not to increase tolerance as page number grows")

  def setPageMatrixCol(value: String): this.type = set(pageMatrixCol, value)
  def getPageMatrixCol: String = $(pageMatrixCol)

  def setMatchingWindow(value: Int): this.type = {
    if (value < 0)
      throw new IllegalArgumentException("Matching window must be non-negative")
    set(matchingWindow, value)
  }
  def getMatchingWindow: Int = $(matchingWindow)

  def setWindowPageTolerance(value: Boolean): this.type = set(windowPageTolerance, value)
  def getWindowPageTolerance: Boolean = $(windowPageTolerance)

  setDefault(
    matchingWindow -> 10,
    windowPageTolerance -> true
  )

  private val parseCoordinates = udf {
    (chunkRaw: Seq[Row], pageRaw: Seq[Row]) => {
      val chunkAnnotations = chunkRaw.map(Annotation(_))

      val bounds = Array.ofDim[Int](pageRaw.length)
      var last = 0

      /** useful for identifying which page entities belong to */
      val matrix = pageRaw.zipWithIndex.flatMap{case (p, i) =>
        val pm = PageMatrix.fromRow(p)
        last += pm.mapping.length
        bounds(i) = last
        pm.mapping
      }

      val coordinates = ArrayBuffer.empty[Coordinate]

      def closeRectangle(minX: Float, maxX: Float, lastPosition: Mapping, targetBegin: Int, chunkIndex: Int): Unit = {
        val x = minX
        val y = lastPosition.y

        val w = (maxX - minX) + lastPosition.width
        val h = lastPosition.height

        coordinates += Coordinate(chunkIndex, bounds.count(targetBegin > _) + lastPosition.p, x, y, w, h)
      }

      chunkAnnotations.zipWithIndex.flatMap{case (target, chunkIndex) =>
        val tolerance = {
          val tol = if ($(windowPageTolerance)) bounds.count(target.begin > _) + 1 else 1
          tol * $(matchingWindow)
        }
        val line = matrix.slice(target.begin - tolerance, target.end + 1 + tolerance)
        val textLine = line.map(_.toString).mkString
        val textLineLength = textLine.length
        var c = 0
        while (c < textLineLength && !textLine.drop(c).startsWith(target.result)) {
          c += 1
        }
        if (c < textLineLength) {

          val matchedLine = line.slice(c, c + target.result.length)

          var minX = -1.0f
          var maxX = -1.0f

          var lastPos = matchedLine.head

          for (pos <- matchedLine) {
            if (pos != null) {
              /** check if we are one line below previous, close rectangle if so */
              if (pos.y < lastPos.y) {
                closeRectangle(minX, maxX, lastPos, target.begin, chunkIndex)
                minX = -1.0f
                maxX = -1.0f
              }
              lastPos = pos
              if (minX == -1 || pos.x < minX) minX = pos.x
              if (maxX == -1 || pos.x > maxX) maxX = pos.x
            }
          }

          /** close lingering rectangle */
          closeRectangle(minX, maxX, lastPos, target.begin, chunkIndex)

          coordinates

        } else {
          Seq.empty[Coordinate]
        }
      }
    }
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    require(
      dataset.schema.fields.exists(field => field.name == $(pageMatrixCol) && field.dataType == PageMatrix.dataType),
      s"because there is no column with name ${$(pageMatrixCol)} and an appropriate PageMatrix schema"
    )
    require(
      getInputCols.length == 1,
      "because inputCols is not 1"
    )
    require(
      dataset.schema.fields.exists(field => field.name == getInputCols.head && field.dataType == ArrayType(Annotation.dataType)),
      s"because there is no column with name ${getInputCols.head} and an appropriate Annotation schema"
    )

    dataset.withColumn(getOutputCol, parseCoordinates(col(getInputCols.head), col(getPageMatrixCol)))

  }

}

object PositionFinder extends ParamsAndFeaturesReadable[PositionFinder]