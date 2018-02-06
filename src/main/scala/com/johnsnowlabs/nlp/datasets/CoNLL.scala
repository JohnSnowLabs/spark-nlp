package com.johnsnowlabs.nlp.datasets

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import com.johnsnowlabs.nlp.annotators.common.{IndexedTaggedWord, NerTagged, PosTagged, TaggedSentence}
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ResourceHelper}
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.collection.mutable.ArrayBuffer

case class CoNLL(targetColumn: Int = 3, annotatorType: String) {
  require(annotatorType == AnnotatorType.NAMED_ENTITY || annotatorType == AnnotatorType.POS)

  /*
    Reads Dataset in CoNLL format and pack it into docs
   */
  def readDocs(er: ExternalResource): Seq[(String, Seq[TaggedSentence])] = {
    val lines = ResourceHelper.parseLines(er)

    readLines(lines)
  }

  def readLines(lines: Array[String]): Seq[(String, Seq[TaggedSentence])] = {
    val doc = new StringBuilder()
    val tokens = new ArrayBuffer[IndexedTaggedWord]()
    val labels = new ArrayBuffer[TaggedSentence]()

    def addSentence(): Unit = {
      if (tokens.nonEmpty) {
        labels.append(TaggedSentence(tokens.toArray))
        tokens.clear()
      }
    }

    val docs = lines
      .flatMap{line =>
        val items = line.split(" ")
        if (items.nonEmpty && items(0) == "-DOCSTART-") {
          addSentence()

          val result = (doc.toString, labels.toList)
          doc.clear()
          labels.clear()

          if (result._1.nonEmpty)
            Some(result._1, result._2)
          else
            None
        } else if (items.length <= 1) {
          if (doc.nonEmpty && doc.last != '\n') {
            doc.append("\n\n")
            addSentence()
          }
          None
        } else
        {
          if (doc.nonEmpty)
            doc.append(" ")

          val begin = doc.length
          doc.append(items(0))
          val end = doc.length - 1
          val tag = items(targetColumn)
          tokens.append(IndexedTaggedWord(items(0), tag, begin, end))
          None
        }
      }

    addSentence()

    val last = if (doc.nonEmpty) Seq((doc.toString, labels.toList)) else Seq.empty

    docs ++ last
  }

  def pack(sentences: Seq[TaggedSentence]): Seq[Annotation] = {
    if (annotatorType == AnnotatorType.NAMED_ENTITY)
      NerTagged.pack(sentences)
    else
      PosTagged.pack(sentences)
  }

  def readDataset(er: ExternalResource,
                  spark: SparkSession,
                  textColumn: String = "text",
                  labelColumn: String = "label"): Dataset[_] = {

    import spark.implicits._

    readDocs(er).map(p => (p._1, pack(p._2))).toDF(textColumn, labelColumn)
  }

  def readDatasetFromLines(lines: Array[String],
                           spark: SparkSession,
                           textColumn: String = "text",
                           labelColumn: String = "label"): Dataset[_] = {

    import spark.implicits._

    val seq = readLines(lines).map(p => (p._1, pack(p._2)))
    seq.toDF(textColumn, labelColumn)
  }
}
