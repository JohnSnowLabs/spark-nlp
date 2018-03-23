package com.johnsnowlabs.ml.logreg


import com.johnsnowlabs.nlp.annotators.assertion.logreg.{SimpleTokenizer, Tokenizer, Windowing}
import com.johnsnowlabs.nlp.embeddings.{WordEmbeddings, WordEmbeddingsIndexer}
import org.apache.spark.sql._
import scala.io.Source

/**
  * Reader for this dataset,
  * https://github.com/mongoose54/negex/blob/master/genConText/rsAnnotations-1-120-random.txt
  */

class NegexDatasetReader(targetLengthLimit: Int = 10) extends Serializable {
  /* returns token numbers for the target within the tokenized sentence */
  private def getTargetIndices(sentence: String, target: String) = {

    val targetTokens = target.split(" ").map(_.trim.toUpperCase).filter(_!="")
    val firstTargetIdx = sentence.split(" ").map(_.trim).
      indexOfSlice(targetTokens)
    val lastTargetIdx = firstTargetIdx + targetTokens.size - 1

    if( lastTargetIdx < 0 || firstTargetIdx < 0)
      print(sentence)
    (firstTargetIdx, lastTargetIdx)
  }

  val specialChars = Seq(',', '.', ';', '.', ':', '/', '"')

  // these lines are ill formed
  val blackList = Seq("2149", "1826", "987", "1321")

  def readDataframe(datasetPath: String)(implicit session:SparkSession):DataFrame = {
      import session.implicits._
      readDataset(datasetPath).toDF
  }

  def readDataset(datasetPath: String) : Seq[Datapoint] = {

    Source.fromFile(datasetPath).getLines
      .map{ line =>
        line.flatMap{ // separate special chars
          case c if specialChars.contains(c)=> s" $c "
          case c => Seq(c)
        }
      }
      .filter{line =>
        // target must be smaller than right context
        line.split("\t")(2).split(" ").filter(_!="").length < targetLengthLimit &&
          // line must contain the target
          line.split("\t")(3).contains(line.split("\t")(2).toUpperCase) &&
          // skip broken lines
          !blackList.exists(line.split("\t")(0).contains)
      }
      .map{ line =>
        val chunks = line.split("\t")
        // keep single spaces only
        val doc = chunks(3).split(" ").map(_.trim).filter(_!="").mkString(" ")
        val (s, e) = getTargetIndices(doc, chunks(2))
        Datapoint(doc.map(_.toLower),
          chunks(2).toLowerCase.trim,
          chunks(4).split(" ")(0).trim, // take Affirmed or Negated
          s, e)
      }.toSeq
  }
}

case class Datapoint(sentence: String, target: String, label: String, start:Int, end:Int)
