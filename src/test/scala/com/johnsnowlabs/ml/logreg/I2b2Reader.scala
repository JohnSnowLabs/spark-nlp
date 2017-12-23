package com.johnsnowlabs.ml.logreg

import com.johnsnowlabs.nlp.embeddings.{WordEmbeddings, WordEmbeddingsIndexer}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, SparkSession}
import java.io.File
import scala.io.Source

/**
  * Reader for the i2b2 dataset
  *
*/

class I2b2DatasetReader(wordEmbeddingsFile: String, targetLengthLimit:Int) extends Serializable  {

  var fileDb = wordEmbeddingsFile + ".db"

  /* receives the location of a single dataset (e.g. 'beth'),
   * and returns a sequence of datapoins I2b2AnnotationAndText
   * */
  private def read(path: String): Seq[I2b2AnnotationAndText] = {

    // read list of ast files, without extension
    val astFileNames = {
      val ast = new File(s"$path/ast/")
      if (ast.exists && ast.isDirectory)
        ast.listFiles.filter(_.isFile).toList.map(_.getName.dropRight(4))
      else
        List[String]()
    }

    var tooLong = 0

    // extract datapoints from each file
    val datapoints = for {name <- astFileNames
           annotation <- Source.fromFile(s"$path/ast/$name.ast").getLines()
           sourceTxt = Source.fromFile(s"$path/txt/$name.txt").getLines().toList
      } yield {
        val record = I2b2Annotation(annotation)
        val text = sourceTxt(record.sourceLine - 1)
        if(record.target.split(" ").length > targetLengthLimit){
          tooLong += 1
          null
        }
        else
          I2b2AnnotationAndText(text, record.target, record.label, record.start, record.end)
      }
    println("number of targets too long: " + tooLong)
    datapoints
  }

  /* reads all the locations for all datasets (e.g. ['beth', 'partners']),
   * and returns a Spark DataFrame
   * */
  def readDataFrame(datasetPaths: Seq[String]) (implicit session: SparkSession): DataFrame= {
    import session.implicits._
    datasetPaths.flatMap(read).filter(_!=null).toDF //.withColumn("label", labelToNumber($"label"))
  }


  lazy val wordVectors: Option[WordEmbeddings] = Option(wordEmbeddingsFile).map {
    wordEmbeddingsFile =>
      require(new File(wordEmbeddingsFile).exists())
      val fileDb = wordEmbeddingsFile + ".db"
      if (!new File(fileDb).exists())
        WordEmbeddingsIndexer.indexBinary(wordEmbeddingsFile, fileDb)
  }.filter(_ => new File(fileDb).exists())
    .map(_ => WordEmbeddings(fileDb, 200))


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
    val pattern = "c=\".*\" (\\d+):\\d+".r
    pattern.findFirstMatchIn(text).map(_.group(1)).
      getOrElse(throw new RuntimeException("Broken dataset - bad source line")).toInt
  }

  def extractLimits(text: String): (Int, Int) = {
    val startPattern = "\\d+:(\\d+)\\s\\d+:\\d+".r
    val endPattern = "\\d+:\\d+\\s\\d+:(\\d+)".r

    val start = startPattern.findAllMatchIn(text).map(_.group(1)).toList match {
      case s::Nil => s.toInt
      case _ => throw new RuntimeException("Broken dataset - bad start")
    }

    val end = endPattern.findAllMatchIn(text).map(_.group(1)).toList match {
      case e::Nil => e.toInt
      case _ => throw new RuntimeException("Broken dataset - bad end")
    }
    (start, end)
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
