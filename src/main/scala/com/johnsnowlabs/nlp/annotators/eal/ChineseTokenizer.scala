package com.johnsnowlabs.nlp.annotators.eal

import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach}
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{DoubleParam, IntParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

import scala.collection.mutable

class ChineseTokenizer(override val uid: String) extends AnnotatorApproach[ChineseTokenizerModel]{

  def this() = this(Identifiable.randomUID("CHINESE_TOKENIZER"))

  override val description: String = "Chinese word segmentation without corpus"

  val maxWordLength = new IntParam(this, "maxWordLength", "Maximum word length")

  val minFrequency = new DoubleParam(this, "minFrequency", "Minimum frequency")

  val minEntropy = new DoubleParam(this, "minEntropy", "Minimum entropy")

  val minAggregation = new DoubleParam(this, "minAggregation", "Minimum aggregation")

  val wordSegmentMethod = new Param[String](this, "wordSegmentMethod", "How to treat a combination of shorter words: LONG, SHORT, ALL")

  val knowledgeBase = new ExternalResourceParam(this, "knowledgeBase", "Text fragment that will be used as knowledge base to segment a sentence with the words generated from it")

  def setWordSegmentMethod(method: String): this.type = {
    method.toUpperCase() match {
      case "LONG" => set(wordSegmentMethod, "LONG")
      case "SHORT" => set(wordSegmentMethod, "SHORT")
      case "ALL" => set(wordSegmentMethod, "ALL")
      case _ => throw new MatchError(s"Invalid WordSegmentMethod parameter. Must be either ${method.mkString("|")}")
    }
  }

  def setKnowledgeBase(path: String, readAs: ReadAs.Format = ReadAs.TEXT,
                       options: Map[String, String] = Map("format" -> "text")): this.type =
    set(knowledgeBase, ExternalResource(path, readAs, options))

  def setMaxWordLength(value: Int): this.type = set(maxWordLength, value)

  def setMinFrequency(value: Double): this.type = set(minFrequency, value)

  def setMinEntropy(value: Double): this.type = set(minEntropy, value)

  def setMinAggregation(value: Double): this.type = set(minAggregation, value)

  setDefault(maxWordLength -> 5, minFrequency -> 0.00005, minEntropy -> 2.0, minAggregation -> 50,
             wordSegmentMethod -> "ALL")

  case class WordInfo(text: String, var aggregation: Double = 0, frequencyEntropy: Double = 0.0,
                      leftEntropy: Double = 0.0, rightEntropy: Double = 0.0)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): ChineseTokenizerModel = {
    var words: Array[String] = Array()
    if (get(knowledgeBase).isDefined) {
      val externalKnowledgeBase = ResourceHelper.parseLines($(knowledgeBase)).mkString(",")
      words = getWords(externalKnowledgeBase)
    } else {
      val knowledgeBase = dataset.select("document.result").rdd.flatMap( row =>
        row.get(0).asInstanceOf[mutable.WrappedArray[String]]
      ).collect().mkString(" ")
      words = getWords(knowledgeBase)
   }

    new ChineseTokenizerModel()
      .setWords(words)
      .setWordSegmentMethod($(wordSegmentMethod))
      .setMaxWordLength($(maxWordLength))
  }

  private def getWords(knowledgeBase: String): Array[String] = {
    val wordCandidates = generateCandidateWords(knowledgeBase)
    val words = wordCandidates.filter(wordCandidate => filterFunction(wordCandidate._2)).keys.toArray
    words
  }

  def generateCandidateWords(knowledgeBase: String): Map[String, WordInfo] = {
    val pattern = "[\\s\\d,.<>/?:;\'\"[\\\\]{}()\\|~!@#$%^&*\\-_=+a-zA-Z，。《》、？：；“”‘’｛｝【】（）…￥！—┄－]+"
    val cleanKnowledgeBase = knowledgeBase.replaceAll(pattern, " ")
    val suffixWithIndexes = indexOfSortedSuffix(cleanKnowledgeBase)
    val wordsInfo = suffixWithIndexes
      .groupBy(_._1).mapValues(_.map(_._2))
      .flatMap{ suffixInfo =>
        val wordInfo = getWordInfo(suffixInfo, cleanKnowledgeBase)
        Map(wordInfo.text -> wordInfo)
      }
    computeAggregation(wordsInfo)
    wordsInfo
  }

  private def indexOfSortedSuffix(cleanKnowledgeBase: String): IndexedSeq[(String, (Int, Int))] = {
    val indexes = 0 until cleanKnowledgeBase.length

    val suffixInfo = indexes.flatMap{index =>
      val suffixIndexes = getSuffixIndexes(index, cleanKnowledgeBase)
      val characters = suffixIndexes.map(suffixIndex => cleanKnowledgeBase.slice(suffixIndex._1, suffixIndex._2))
      characters zip suffixIndexes
    }
    suffixInfo.sortBy(_._1)
  }

  private def getSuffixIndexes(index: Int, sentence: String): List[(Int, Int)] = {
    val begin = index + 1
    val end = (begin + $(maxWordLength)).min(sentence.length + 1)
    val secondElement = (begin until end).toList
    val firstElement = List.fill(secondElement.length)(index)
    firstElement zip secondElement
  }

  private def getWordInfo(suffixInfo: (String, IndexedSeq[(Int, Int)]), knowledgeBase: String): WordInfo = {
    val neighbors = suffixInfo._2.map { indexRange =>
      val left = knowledgeBase.slice(indexRange._1 - 1, indexRange._1)
      val right = knowledgeBase.slice(indexRange._2, indexRange._2 + 1)
      (left, right)
    }
    val leftNeighbors = neighbors.map(_._1).toList.filter(neighbor => neighbor != "")
    val rightNeighbors = neighbors.map(_._2).toList.filter(neighbor => neighbor != "")
    val leftEntropy = computeEntropy(leftNeighbors)
    val rightEntropy = computeEntropy(rightNeighbors)
    val frequencyEntropy = suffixInfo._2.size.toDouble/ knowledgeBase.length.toDouble
    WordInfo(suffixInfo._1, 0, frequencyEntropy, leftEntropy, rightEntropy)
  }

  private def computeEntropy(elements: List[String]): Double = {

    val occurrencesElements = elements.groupBy(identity).mapValues(_.size).map(_._2.toDouble)
    //The entropy is the sum of -p[i]*log(p[i]) for every unique element i in the list, and p[i] is its frequency
    val entropyElements = occurrencesElements.map(element => (-element/elements.size) * math.log(element/elements.size))
    entropyElements.sum
  }

  private def computeAggregation(wordsInfoMap: Map[String, WordInfo]): Unit = {
    wordsInfoMap.foreach{ wordInfo =>
      val text = wordInfo._2.text
      if (text.length > 1) {
        val subParts = text.zipWithIndex.map{case (_, index) => (text.slice(0, index), text.slice(index, text.length))
        }.filter(subpart => subpart._1 != "" && subpart._2 != "")
        val aggregation = subParts.map{ subPart =>
          val frequency1 = wordsInfoMap(subPart._1).frequencyEntropy
          val frequency2 = wordsInfoMap(subPart._2).frequencyEntropy
          wordInfo._2.frequencyEntropy / frequency1 / frequency2
        }.min
        wordInfo._2.aggregation = aggregation
      }
    }
  }

  private def filterFunction(wordInfo: WordInfo): Boolean = {
    wordInfo.text.length > 1 &&
      wordInfo.aggregation > $(minAggregation) &&
      wordInfo.frequencyEntropy > $(minFrequency) &&
      wordInfo.leftEntropy > $(minEntropy) && wordInfo.rightEntropy > $(minEntropy)
  }

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)
  override val outputAnnotatorType: AnnotatorType = TOKEN
}

object ChineseTokenizer extends DefaultParamsReadable[ChineseTokenizer]