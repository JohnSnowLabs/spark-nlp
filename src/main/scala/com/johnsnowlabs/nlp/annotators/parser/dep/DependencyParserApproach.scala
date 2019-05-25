package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition._
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.IntParam
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset
import org.slf4j.LoggerFactory

class DependencyParserApproach(override val uid: String) extends AnnotatorApproach[DependencyParserModel] {

  override val description: String =
    "Dependency Parser is an unlabeled parser that finds a grammatical relation between two words in a sentence"

  private val logger = LoggerFactory.getLogger("DependencyParserApproach")

  def this() = this(Identifiable.randomUID(DEPENDENCY))

  val dependencyTreeBank = new ExternalResourceParam(this, "dependencyTreeBank", "Dependency treebank source files")
  val conllU = new ExternalResourceParam(this, "conllU", "Universal Dependencies source files")
  val numberOfIterations = new IntParam(this, "numberOfIterations", "Number of iterations in training, converges to better accuracy")

  def setDependencyTreeBank(path: String, readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                            options: Map[String, String] = Map.empty[String, String]): this.type =
    set(dependencyTreeBank, ExternalResource(path, readAs, options))

  def setConllU(path: String, readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                options: Map[String, String] = Map.empty[String, String]): this.type =
    set(conllU, ExternalResource(path, readAs, options))

  def setNumberOfIterations(value: Int): this.type = set(numberOfIterations, value)

  setDefault(dependencyTreeBank, ExternalResource("", ReadAs.LINE_BY_LINE,  Map.empty[String, String]))
  setDefault(conllU, ExternalResource("", ReadAs.LINE_BY_LINE,  Map.empty[String, String]))
  setDefault(numberOfIterations, 10)

  def getNumberOfIterations: Int = $(numberOfIterations)

  override val outputAnnotatorType:String = DEPENDENCY

  override val inputAnnotatorTypes = Array(DOCUMENT, POS, TOKEN)

  private lazy val conllUAsArray = ResourceHelper.parseLines($(conllU))

  def readCONLL(filesContent: Seq[Iterator[String]]): List[Sentence] = {

    val buffer = StringBuilder.newBuilder

    filesContent.foreach{fileContent =>
      fileContent.foreach(line => buffer.append(line+System.lineSeparator()))
      buffer.append(System.lineSeparator())
    }

    val wholeText = buffer.toString()
    val sections = wholeText.split(s"${System.lineSeparator()}${System.lineSeparator()}").toList

    val sentences = sections.map(
      s => {
        val lines = s.split(s"${System.lineSeparator()}").toList
        val body  = lines.map( l => {
          val arr = l.split("\\s+")
          val (raw, pos, dep) = (arr(0), arr(1), arr(2).toInt)
          // CONLL dependency layout assumes [root, word1, word2, ..., wordn]  (where n == lines.length)
          // our   dependency layout assumes [word0, word1, ..., word(n-1)] { root }
          val dep_ex = if(dep==0) lines.length+1-1 else dep-1
          WordData(raw, pos, dep_ex)
        })
        body  // Don't pretty up the sentence itself
      }
    )
    sentences
  }

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): DependencyParserModel = {

    validateTrainingFiles()
    val trainingSentences = getTrainingSentences
    val (classes, tagDictionary) = TagDictionary.classesAndTagDictionary(trainingSentences)
    val tagger = new Tagger(classes, tagDictionary)
    val taggerNumberOfIterations = getNumberOfIterations

    val dependencyMaker = new DependencyMaker(tagger)

    val dependencyMakerPerformanceProgress = (0 until taggerNumberOfIterations).map{ seed =>
      dependencyMaker.train(trainingSentences, seed)
    }
    logger.info(s"Dependency Maker Performance = $dependencyMakerPerformanceProgress")

    new DependencyParserModel()
      .setPerceptron(dependencyMaker)
  }

  def validateTrainingFiles(): Unit = {
    if ($(dependencyTreeBank).path != "" && $(conllU).path != "") {
      throw new IllegalArgumentException("Use either TreeBank or CoNLL-U format file both are not allowed.")
    }
    if ($(dependencyTreeBank).path == "" && $(conllU).path == "") {
      throw new IllegalArgumentException("Either TreeBank or CoNLL-U format file is required.")
    }
  }

  def getTrainingSentences: List[Sentence] = {
    if ($(dependencyTreeBank).path != ""){
      val filesContentTreeBank = getFilesContentTreeBank
      readCONLL(filesContentTreeBank)
    } else {
      getTrainingSentencesFromConllU(conllUAsArray)
    }
  }

  def  getFilesContentTreeBank: Seq[Iterator[String]] = ResourceHelper.getFilesContentBuffer($(dependencyTreeBank))

  def getTrainingSentencesFromConllU(conllUAsArray: Array[String]): List[Sentence] = {

    val conllUSentences = conllUAsArray.filterNot(line => lineIsComment(line))
    val indexSentenceBoundaries = conllUSentences.zipWithIndex.filter(_._1 == "").map(_._2)
    val cleanConllUSentences = indexSentenceBoundaries.zipWithIndex.map{case (indexSentenceBoundary, index) =>
      if (index == 0){
        conllUSentences.slice(index, indexSentenceBoundary)
      } else {
        conllUSentences.slice(indexSentenceBoundaries(index-1)+1, indexSentenceBoundary)
      }
    }
    val sentences = cleanConllUSentences.map{cleanConllUSentence =>
      transformToSentences(cleanConllUSentence)
    }
    sentences.toList
  }

  def lineIsComment(line: String): Boolean = {
    if (line.nonEmpty){
      line(0) == '#'
    } else {
      false
    }
  }

  def transformToSentences(cleanConllUSentence: Array[String]): Sentence = {
    val ID_INDEX = 0
    val WORD_INDEX = 1
    val POS_INDEX = 4
    val HEAD_INDEX = 6
    val SEPARATOR = "\\t"

    val sentences = cleanConllUSentence.map{conllUWord =>
      val wordArray = conllUWord.split(SEPARATOR)
      if (!wordArray(ID_INDEX).contains(".")){
        var head = wordArray(HEAD_INDEX).toInt
        if (head == 0){
          head = cleanConllUSentence.length
        } else {
          head = head-1
        }
        WordData(wordArray(WORD_INDEX), wordArray(POS_INDEX), head)
      } else {
        WordData("", "", -1)
      }
    }

    sentences.filter(word => word.dep != -1).toList
  }

}

object DependencyParserApproach extends DefaultParamsReadable[DependencyParserApproach]