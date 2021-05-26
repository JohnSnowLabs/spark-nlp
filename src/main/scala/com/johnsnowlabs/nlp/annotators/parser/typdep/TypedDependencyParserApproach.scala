package com.johnsnowlabs.nlp.annotators.parser.typdep

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DEPENDENCY, LABELED_DEPENDENCY, POS, TOKEN}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.IntParam
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset


/** Labeled parser that finds a grammatical relation between two words in a sentence. Its input is a CoNLL2009 or ConllU dataset.
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/parser/typdep ]] for further reference on this API.
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
class TypedDependencyParserApproach(override val uid: String) extends AnnotatorApproach[TypedDependencyParserModel] {

  /** Typed Dependency Parser is a labeled parser that finds a grammatical relation between two words in a sentence */
  override val description: String = "Typed Dependency Parser is a labeled parser that finds a grammatical relation between two words in a sentence"
  /** Input annotation type : LABELED_DEPENDENCY
    *
    * @group anno
    **/
  override val outputAnnotatorType: String = LABELED_DEPENDENCY
  /** Input annotation type : TOKEN, POS, DEPENDENCY
    *
    * @group anno
    **/
  override val inputAnnotatorTypes = Array(TOKEN, POS, DEPENDENCY)

  def this() = this(Identifiable.randomUID("TYPED_DEPENDENCY"))

  /** Number of iterations in training, converges to better accuracy
    *
    * @group param
    **/
  val numberOfIterations = new IntParam(this, "numberOfIterations", "Number of iterations in training, converges to better accuracy")
  /** Path to file with CoNLL 2009 format
    *
    * @group param
    **/
  val conll2009 = new ExternalResourceParam(this, "conll2009", "Path to file with CoNLL 2009 format")
  /** Universal Dependencies source files
    *
    * @group param
    **/
  val conllU = new ExternalResourceParam(this, "conllU", "Universal Dependencies source files")

  //TODO: Enable more training parameters from Options

  /** Path to a file in [[https://ufal.mff.cuni.cz/conll2009-st/trial-data.html CoNLL 2009 format]]
    *
    * @group setParam
    **/
  def setConll2009(path: String, readAs: ReadAs.Format = ReadAs.TEXT,
                   options: Map[String, String] = Map.empty[String, String]): this.type = {
    set(conll2009, ExternalResource(path, readAs, options))
  }

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

  setDefault(conll2009, ExternalResource("", ReadAs.TEXT, Map.empty[String, String]))
  setDefault(conllU, ExternalResource("", ReadAs.TEXT, Map.empty[String, String]))
  setDefault(numberOfIterations, 10)


  private lazy val trainFile = {
    getTrainingFile
  }

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): TypedDependencyParserModel = {

    validateTrainingFiles()
    val options = getOptionsInstance
    options.setNumberOfTrainingIterations($(numberOfIterations))
    val typedDependencyParser = getTypedDependencyParserInstance
    typedDependencyParser.setOptions(options)

    val dependencyPipe = getDependencyPipeInstance(options)
    typedDependencyParser.setDependencyPipe(dependencyPipe)
    dependencyPipe.createAlphabets(trainFile.path, trainFile.conllFormat)

    val trainDependencies = getTrainDependenciesInstance(trainFile, dependencyPipe, typedDependencyParser, options)
    trainDependencies.startTraining()

    val dictionaries = trainDependencies.getDependencyPipe.getDictionariesSet.getDictionaries

    dictionaries.foreach(dictionary => dictionary.setMapAsString(dictionary.getMap.toString))

    typedDependencyParser.getDependencyPipe.closeAlphabets()

    new TypedDependencyParserModel()
      .setOptions(trainDependencies.getOptions)
      .setDependencyPipe(trainDependencies.getDependencyPipe)
      .setConllFormat(getTrainingFile.conllFormat)
  }

  def validateTrainingFiles(): Unit = {
    if ($(conll2009).path != "" && $(conllU).path != "") {
      throw new IllegalArgumentException("Use either CoNLL-2009 or CoNLL-U format file both are not allowed.")
    }
    if ($(conll2009).path == "" && $(conllU).path == "") {
      throw new IllegalArgumentException("Either CoNLL-2009 or CoNLL-U format file is required.")
    }
  }

  def getTrainingFile: TrainFile = {
    if ($(conll2009).path != ""){
      TrainFile($(conll2009).path, "2009")

    } else {
      TrainFile($(conllU).path, "U")
    }
  }

  private def getOptionsInstance: Options = {
    new Options()
  }

  private def getTypedDependencyParserInstance: TypedDependencyParser = {
    new TypedDependencyParser()
  }

  private def getDependencyPipeInstance(options: Options): DependencyPipe = {
    new DependencyPipe(options)
  }

  private def getTrainDependenciesInstance(trainFile: TrainFile, dependencyPipe: DependencyPipe,
                                           typedDependencyParser: TypedDependencyParser,
                                           options: Options): TrainDependencies = {
    new TrainDependencies(trainFile, dependencyPipe, typedDependencyParser, options)
  }

}

object TypedDependencyParserApproach extends DefaultParamsReadable[TypedDependencyParserApproach]
