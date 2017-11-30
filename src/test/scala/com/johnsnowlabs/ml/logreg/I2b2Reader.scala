package com.johnsnowlabs.ml.logreg

import java.io.File
import org.apache.spark.sql.{DataFrame, SparkSession}
import scala.io.Source


class I2b2DatasetReader(datasetPath: String) {

  def readDataset(implicit session: SparkSession): DataFrame = {
    import session.implicits._
    // read list of ast files, without extension
    val astFileNames = {
      val ast = new File(s"$datasetPath/ast/")
      if (ast.exists && ast.isDirectory)
        ast.listFiles.filter(_.isFile).toList.map(_.getName.dropRight(4))
      else
        List[String]()
    }

    // extract datapoints from each file
    val dataset =
      for {name <- astFileNames
           annotation <- Source.fromFile(s"$datasetPath/ast/$name.ast").getLines()
           sourceTxt = Source.fromFile(s"$datasetPath/txt/$name.txt").getLines().toList
    } yield {
      val record = I2b2Annotation(annotation)
      val text = sourceTxt(record.sourceLine - 1)
      I2b2AnnotationAndText(text, record.target, record.label, record.start, record.end)
    }
    dataset.toDF()
  }
}
case class I2b2Annotation(target: String, label: String, start:Int, end:Int, sourceLine:Int)
case class I2b2AnnotationAndText(text: String, target: String, label: String, start:Int, end:Int)

object I2b2Annotation {

  private def extractTarget(text:String): String = {
    val pattern = "c=\"(.*)\"".r
    pattern.findFirstMatchIn(text).map(_.group(1)).
      getOrElse(throw new RuntimeException("Broken dataset - bad target"))
  }

  private def extractSourceLine(text: String): Int = {
    val pattern = "(\\d+):\\d+".r
    pattern.findFirstMatchIn(text).map(_.group(1)).
      getOrElse(throw new RuntimeException("Broken dataset - bad source line")).toInt
  }

  def extractLimits(text: String): (Int, Int) = {
    val pattern = "\\d+:(\\d+)".r
    pattern.findAllMatchIn(text).map(_.group(1)).toList match {
      case start::end::Nil => (start.toInt, end.toInt)
      case _ => throw new RuntimeException("Broken dataset - bad start and end")
    }
  }

  def extractLabel(text: String) = {
    val pattern = "a=\"(.*)\"".r
    pattern.findFirstMatchIn(text).map(_.group(1)).
      getOrElse(throw new RuntimeException("Broken dataset - bad source line"))
  }

  def apply(annotation: String): I2b2Annotation = {
    val chunks = annotation.split("\\|\\|")
    val target = extractTarget(chunks(0))
    val sourceLine = extractSourceLine(chunks(0))
    val (start, end) = extractLimits(chunks(0))
    val label = extractLabel(chunks(2))
    I2b2Annotation(target, label, start, end, sourceLine)
  }
}
