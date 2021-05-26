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

/** Unlabeled parser that finds a grammatical relation between two words in a sentence. Its input is a directory with dependency treebank files.
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/parser/dep]] for further reference on how to use this API.
  *
  * @groupname anno Annotator types
  * @groupdesc anno Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  **/
class DependencyParserApproach(override val uid: String) extends AnnotatorApproach[DependencyParserModel] {

  override val description: String = "Dependency Parser is an unlabeled parser that finds a grammatical relation between two words in a sentence"

  private val logger = LoggerFactory.getLogger("DependencyParserApproach")

  def this() = this(Identifiable.randomUID(DEPENDENCY))

  /** Dependency treebank source files
    *
    * @group param
    **/
  val dependencyTreeBank = new ExternalResourceParam(this, "dependencyTreeBank", "Dependency treebank source files")
  /** Universal Dependencies source files
    *
    * @group param
    **/
  val conllU = new ExternalResourceParam(this, "conllU", "Universal Dependencies source files")
  /** Number of iterations in training, converges to better accuracy
    *
    * @group param
    **/
  val numberOfIterations = new IntParam(this, "numberOfIterations", "Number of iterations in training, converges to better accuracy")


  /** Dependency treebank folder with files in [[ http://www.nltk.org/nltk_data/ Penn Treebank format]]
    *
    * @group setParam
    **/
  def setDependencyTreeBank(path: String, readAs: ReadAs.Format = ReadAs.TEXT,
                            options: Map[String, String] = Map.empty[String, String]): this.type =
    set(dependencyTreeBank, ExternalResource(path, readAs, options))

  /** Path to a file in [[https://universaldependencies.org/format.html CoNLL-U format]]
    *
    * @group setParam
    **/
  def setConllU(path: String, readAs: ReadAs.Format = ReadAs.TEXT,
                options: Map[String, String] = Map.empty[String, String]): this.type =
    set(conllU, ExternalResource(path, readAs, options))

  /** Number of iterations in training, converges to better accuracy
    *
    * @group setParam
    **/
  def setNumberOfIterations(value: Int): this.type = set(numberOfIterations, value)

  setDefault(dependencyTreeBank, ExternalResource("", ReadAs.TEXT, Map.empty[String, String]))
  setDefault(conllU, ExternalResource("", ReadAs.TEXT, Map.empty[String, String]))
  setDefault(numberOfIterations, 10)

  /** Number of iterations in training, converges to better accuracy
    *
    * @group getParam
    **/
  def getNumberOfIterations: Int = $(numberOfIterations)

  /** Output annotation type : DEPENDENCY
    *
    * @group anno
    **/
  override val outputAnnotatorType: String = DEPENDENCY
  /** Input annotation type : DOCUMENT, POS, TOKEN
    *
    * @group anno
    **/
  override val inputAnnotatorTypes = Array(DOCUMENT, POS, TOKEN)

  private lazy val conllUAsArray = ResourceHelper.parseLines($(conllU))

  def readCONLL(filesContent: Seq[Iterator[String]]): List[Sentence] = {

    val buffer = StringBuilder.newBuilder

    filesContent.foreach { fileContent =>
      fileContent.foreach(line => buffer.append(line + System.lineSeparator()))
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

  /** Gets a list of ConnlU training sentences */
  def getTrainingSentences: List[Sentence] = {
    if ($(dependencyTreeBank).path != ""){
      val filesContentTreeBank = getFilesContentTreeBank
      readCONLL(filesContentTreeBank)
    } else {
      getTrainingSentencesFromConllU(conllUAsArray)
    }
  }

  /** Gets a iterable TreeBank */
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