package com.johnsnowlabs.nlp.annotators.sentence_detector_dl

import org.apache.spark.ml.util.Identifiable

import scala.collection.mutable.{ArrayBuffer, Map}
import scala.util.Random
import java.io.PrintWriter

import org.apache.spark.ml.param.Param

import scala.io.Source
import org.json4s.jackson.Serialization

class SentenceDetectorDLEncoder extends Serializable {

  protected val windowSize = 5
  protected val minFreq = 100

  protected val skipChars = Array(' ', '\n')
  protected val eosChars = Array('.', ':', '?', '!', ';')
  protected val eosCandidatesPattern = "[\\.:?!;\\n]".r

  private var freqMap =  Map[Char, Int]()

  def getSkipChars = this.skipChars

  def getVocabularyJSON(): String = {
    implicit val formats = org.json4s.DefaultFormats

    Serialization.write(freqMap.keys.map(ch => (ch.toString, freqMap(ch))).toList)
  }

  def setVocabulary(vocabularyJSON: String) = {
    implicit val formats = org.json4s.DefaultFormats

    freqMap.clear()
    Serialization.read[List[(String, Int)]](vocabularyJSON).foreach(v => freqMap(v._1(0)) = v._2)
  }

  def saveVocabulary(filename: String) = {
    new PrintWriter(filename) {
        write(getVocabularyJSON)
        close()
      }
  }

  def loadVocabulary(filename: String) = {
    val json = Source.fromFile(filename).getLines().mkString("\n")
    setVocabulary(json)
  }

  def loadMultiLangVocabulary(filenames: Array[String]): Unit = {
    freqMap.clear()
    var i = 0
    filenames.foreach(filename => {
      implicit val formats = org.json4s.DefaultFormats

      val json = Source.fromFile(filename).getLines().mkString("\n")

      Serialization.read[List[(String, Int)]](json).foreach(
        v => {
          if (!freqMap.contains(v._1(0))){
            freqMap(v._1(0)) = i
            i += 1
          }
        }
      )
    })
  }

  def buildVocabulary(trainingTextData: String): Unit = {

    trainingTextData.foreach(
      ch => {
        if (!freqMap.contains(ch)) {freqMap(ch) = 0}
        freqMap(ch) += 1
      }
    )

    val boundedMinFreq = scala.math.max(freqMap.values.toArray.sorted.reverse.take(300).min, minFreq)
    freqMap = freqMap.retain((ch, n) => (n > boundedMinFreq))

    val charIds = freqMap.keysIterator.toList

    freqMap.transform((ch, n) => charIds.indexOf(ch))
  }

  protected def encodeChar(char: Char): Float = {
    if (freqMap.contains(char)) freqMap(char) else 0.0f
  }

  def getRightContext(text: String, pos: Int): String = {
    var rightContext = ""

    var j = pos + 1

    while (j < text.length && rightContext.length < windowSize * 3) {

      if (!((rightContext.length == 0) && skipChars.contains(text(j)))){
        rightContext = rightContext + text(j)
      }

      j += 1
    }

    rightContext.replaceAll("\n", " ").replaceAll(" +", " ").slice(0, windowSize)
  }

  def getLeftContext(text: String, pos: Int): String = {
    var leftContext = ""

    var j = pos - 1

    while (j > 0 && leftContext.length < windowSize * 3) {

      if (!((leftContext.length == 0) && skipChars.contains(text(j)))){
        leftContext = leftContext + text(j)
      }

      j -= 1
    }

    leftContext.replaceAll("\n", " ").replaceAll(" +", " ").slice(0, windowSize).reverse
  }

  def getTrainingData(text: String): (Array[Float], Array[Array[Float]]) = {

    var probNL = text.split("\n").length.toFloat / text.length.toFloat;

    var eosChar: Option[Char] = None

    var leftContext = ""
    var rightContext = ""

    var i = -1

    var examples = ArrayBuffer[(Float, String)]()

    text.foreach(ch => {
      i += 1

      if (eosChars.contains(ch)) {
        eosChar = Some(ch)
        leftContext = getLeftContext(text, i)
        rightContext = getRightContext(text, i)
      } else {

        if (ch == '\n') {
          //positive example

          if (eosChar.isDefined) {
            examples.append((1.0f, leftContext + eosChar.get + rightContext))
            examples.append((1.0f, leftContext + '\n' + rightContext))
          } else {
            examples.append((1.0f, getLeftContext(text, i) + '\n' + getRightContext(text, i)))
          }
          eosChar = None

        } else if (!skipChars.contains(ch)) {
          if (eosChar.isDefined) {
            //negative example
            examples.append((0.0f, leftContext + eosChar.get + rightContext))
          } else if (scala.math.random < probNL) {
            examples.append((0.0f, getLeftContext(text, i) + '\n' + getRightContext(text, i - 1)))
          }

          eosChar = None
        }

      }
    })

    examples
      .map(ex => (ex._1, ex._2.map(ch => encodeChar(ch)).toArray))
      .filter(ex => (ex._2.length == (2 * windowSize + 1)))
      .toArray
      .unzip
  }

  def getEOSPositions(text: String, impossiblePenultimates: Array[String] = Array()): Iterator[(Int, Array[Float])] = {


    eosCandidatesPattern.findAllMatchIn(text)
      .map(m => m.start)
      .map(
        pos => {
          //get context and get rid of multiple spaces
          val leftC = getLeftContext(text, pos)
          val rightC = getRightContext(text, pos)

          if (impossiblePenultimates.find(penultimate => leftC.endsWith(penultimate)).isDefined){
            (-1, "")
          } else {
            (pos, leftC.slice(leftC.length - windowSize, leftC.length) + text(pos) + rightC.slice(0, windowSize + 1))
          }
        }
      )
      .filter(p => p._1 >= 0)
      .map(ex => (ex._1, ex._2.map(ch => encodeChar(ch)).toArray))
  }

}

object SentenceDetectorDLEncoder extends SentenceDetectorDLEncoder

class SentenceDetectorDLEncoderParam(parent: Identifiable, name: String, doc: String)
  extends Param[SentenceDetectorDLEncoder](parent, name, doc) {

  override def jsonEncode(encoder: SentenceDetectorDLEncoder): String = {
    implicit val formats = org.json4s.DefaultFormats

    encoder.getVocabularyJSON()
  }

  override def jsonDecode(json: String): SentenceDetectorDLEncoder = {

    implicit val formats = org.json4s.DefaultFormats

    val encoder = new SentenceDetectorDLEncoder()
    encoder.setVocabulary(json)

    encoder
  }
}