package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.annotators.parser.TagDictionary
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

  private val logger = LoggerFactory.getLogger("NerCrfApproach")

  def this() = this(Identifiable.randomUID(DEPENDENCY))

  val dependencyTreeBank = new ExternalResourceParam(this, "dependencyTreeBank", "Dependency treebank source files")

  val numberOfIterations = new IntParam(this, "numberOfIterations", "Number of iterations in training, converges to better accuracy")

  def setDependencyTreeBank(path: String, readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                            options: Map[String, String] = Map.empty[String, String]): this.type =
    set(dependencyTreeBank, ExternalResource(path, readAs, options))

  def setNumberOfIterations(value: Int): this.type = set(numberOfIterations, value)

  setDefault(numberOfIterations, 10)

  def getNumberOfIterations: Int = $(numberOfIterations)

  override val outputAnnotatorType:String = DEPENDENCY

  override val inputAnnotatorTypes = Array(DOCUMENT, POS, TOKEN)

  private lazy val filesContent = ResourceHelper.getFilesContentAsArray($(dependencyTreeBank))

  private lazy val trainingSentences = filesContent.flatMap(fileContent => readCONLL(fileContent)).toList

  def readCONLL(filesContent: String): List[Sentence] = {

    val sections = filesContent.split("\\n\\n").toList

    val sentences = sections.map(
      s => {
        val lines = s.split("\\n").toList
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

    val (classes, tagDictionary) = TagDictionary.classesAndTagDictionary(trainingSentences)

    val tagger = new Tagger(classes, tagDictionary)
    val taggerNumberOfIterations = getNumberOfIterations
    val dependencyMakerNumberOfIterations = getNumberOfIterations + 5

    val taggerPerformanceProgress = (0 until taggerNumberOfIterations).map { seed =>
        tagger.train(trainingSentences, seed) //Iterates to increase accuracy
    }
    logger.info(s"Tagger Performance = $taggerPerformanceProgress")

    var perceptronAsArray = tagger.getPerceptronAsArray

    val greedyTransition = new GreedyTransitionApproach()
    val dependencyMaker = greedyTransition.loadPerceptronInTraining(perceptronAsArray)

    val dependencyMakerPerformanceProgress = (0 until dependencyMakerNumberOfIterations).map{ seed =>
      dependencyMaker.train(trainingSentences, seed, tagger)
    }
    logger.info(s"Dependency Maker Performance = $dependencyMakerPerformanceProgress")

    perceptronAsArray = dependencyMaker.getPerceptronAsArray

    new DependencyParserModel()
      .setPerceptronAsArray(perceptronAsArray)
  }

}

object DependencyParserApproach extends DefaultParamsReadable[DependencyParserApproach]