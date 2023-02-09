/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.spell.context

import com.github.liblevenshtein.transducer.factory.TransducerBuilder
import com.github.liblevenshtein.transducer.{Algorithm, Candidate}
import com.johnsnowlabs.ml.tensorflow.{TensorflowSpell, TensorflowWrapper, Variables}
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.annotators.spell.context.parser._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach, AnnotatorType, HasFeatures}
import org.apache.commons.io.IOUtils
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, SparkSession}
import org.slf4j.LoggerFactory
import org.tensorflow.Graph
import org.tensorflow.proto.framework.GraphDef

import java.io.{BufferedWriter, File, FileWriter}
import java.util
import scala.collection.mutable
import scala.language.existentials

case class LangModelSentence(ids: Array[Int], cids: Array[Int], cwids: Array[Int], len: Int)

object CandidateStrategy {
  val ALL_UPPER_CASE = 0
  val FIRST_LETTER_CAPITALIZED = 1
  val ALL = 2
}

/** Trains a deep-learning based Noisy Channel Model Spell Algorithm. Correction candidates are
  * extracted combining context information and word information.
  *
  * For instantiated/pretrained models, see [[ContextSpellCheckerModel]].
  *
  * Spell Checking is a sequence to sequence mapping problem. Given an input sequence, potentially
  * containing a certain number of errors, `ContextSpellChecker` will rank correction sequences
  * according to three things:
  *   1. Different correction candidates for each word — '''word level'''.
  *   1. The surrounding text of each word, i.e. it’s context — '''sentence level'''.
  *   1. The relative cost of different correction candidates according to the edit operations at
  *      the character level it requires — '''subword level'''.
  *
  * For an in-depth explanation of the module see the article
  * [[https://medium.com/spark-nlp/applying-context-aware-spell-checking-in-spark-nlp-3c29c46963bc Applying Context Aware Spell Checking in Spark NLP]].
  *
  * For extended examples of usage, see the article
  * [[https://towardsdatascience.com/training-a-contextual-spell-checker-for-italian-language-66dda528e4bf Training a Contextual Spell Checker for Italian Language]],
  * the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/italian/Training_Context_Spell_Checker_Italian.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/spell/context/ContextSpellCheckerTestSpec.scala ContextSpellCheckerTestSpec]].
  *
  * ==Example==
  * For this example, we use the first Sherlock Holmes book as the training dataset.
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellCheckerApproach
  *
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols("document")
  *   .setOutputCol("token")
  *
  * val spellChecker = new ContextSpellCheckerApproach()
  *   .setInputCols("token")
  *   .setOutputCol("corrected")
  *   .setWordMaxDistance(3)
  *   .setBatchSize(24)
  *   .setEpochs(8)
  *   .setLanguageModelClasses(1650)  // dependant on vocabulary size
  *   // .addVocabClass("_NAME_", names) // Extra classes for correction could be added like this
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   tokenizer,
  *   spellChecker
  * ))
  *
  * val path = "src/test/resources/spell/sherlockholmes.txt"
  * val dataset = spark.sparkContext.textFile(path)
  *   .toDF("text")
  * val pipelineModel = pipeline.fit(dataset)
  * }}}
  *
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach NorvigSweetingApproach]]
  *   and
  *   [[com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteApproach SymmetricDeleteApproach]]
  *   for alternative approaches to spell checking
  * @param uid
  *   required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
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
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class ContextSpellCheckerApproach(override val uid: String)
    extends AnnotatorApproach[ContextSpellCheckerModel]
    with HasFeatures
    with WeightedLevenshtein {

  override val description: String = "Context Spell Checker"

  private val logger = LoggerFactory.getLogger("ContextSpellCheckerApproach")

  /** List of parsers for special classes (Default: `List(new DateToken, new NumberToken)`).
    *
    * @group param
    */
  val specialClasses = new Param[List[SpecialClassParser]](
    this,
    "specialClasses",
    "List of parsers for special classes.")

  /** @group setParam */
  def setSpecialClasses(parsers: List[SpecialClassParser]): this.type =
    set(specialClasses, parsers)

  /** Number of classes to use during factorization of the softmax output in the LM (Default:
    * `2000`).
    *
    * @group param
    */
  val languageModelClasses = new Param[Int](
    this,
    "languageModelClasses",
    "Number of classes to use during factorization of the softmax output in the LM.")

  /** @group setParam */
  def setLanguageModelClasses(k: Int): this.type = set(languageModelClasses, k)

  /** Maximum distance for the generated candidates for every word (Default: `3`).
    *
    * @group param
    */
  val wordMaxDistance = new IntParam(
    this,
    "wordMaxDistance",
    "Maximum distance for the generated candidates for every word.")

  /** @group setParam */
  def setWordMaxDistance(k: Int): this.type = {
    require(k >= 1, "Please provided a minumum candidate distance of at least 1.")
    set(wordMaxDistance, k)
  }

  /** Maximum number of candidates for every word (Default: `6`).
    *
    * @group param
    */
  val maxCandidates =
    new IntParam(this, "maxCandidates", "Maximum number of candidates for every word.")

  /** @group setParam */
  def setMaxCandidates(k: Int): this.type = set(maxCandidates, k)

  /** What case combinations to try when generating candidates (Default: `CandidateStrategy.ALL`).
    *
    * @group param
    */
  val caseStrategy = new IntParam(
    this,
    "caseStrategy",
    "What case combinations to try when generating candidates.")

  /** @group setParam */
  def setCaseStrategy(k: Int): this.type = set(caseStrategy, k)

  /** Threshold perplexity for a word to be considered as an error (Default: `10f`).
    *
    * @group param
    */
  val errorThreshold = new FloatParam(
    this,
    "errorThreshold",
    "Threshold perplexity for a word to be considered as an error.")

  /** @group setParam */
  def setErrorThreshold(t: Float): this.type = set(errorThreshold, t)

  /** Number of epochs to train the language model (Default: `2`).
    *
    * @group param
    */
  val epochs = new IntParam(this, "epochs", "Number of epochs to train the language model.")

  /** @group setParam */
  def setEpochs(k: Int): this.type = set(epochs, k)

  /** Batch size for the training in NLM (Default: `24`).
    *
    * @group param
    */
  val batchSize = new IntParam(this, "batchSize", "Batch size for the training in NLM.")

  /** @group setParam */
  def setBatchSize(k: Int): this.type = set(batchSize, k)

  /** Initial learning rate for the LM (Default: `.7f`).
    *
    * @group param
    */
  val initialRate = new FloatParam(this, "initialRate", "Initial learning rate for the LM.")

  /** @group setParam */
  def setInitialRate(r: Float): this.type = set(initialRate, r)

  /** Final learning rate for the LM (Default: `0.0005f`).
    *
    * @group param
    */
  val finalRate = new FloatParam(this, "finalRate", "Final learning rate for the LM.")

  /** @group setParam */
  def setFinalRate(r: Float): this.type = set(finalRate, r)

  /** Percentage of datapoints to use for validation (Default: `.1f`).
    *
    * @group param
    */
  val validationFraction =
    new FloatParam(this, "validationFraction", "percentage of datapoints to use for validation.")

  /** @group setParam */
  def setValidationFraction(r: Float): this.type = set(validationFraction, r)

  /** Min number of times a token should appear to be included in vocab (Default: `3.0`).
    *
    * @group param
    */
  val minCount = new Param[Double](
    this,
    "minCount",
    "Min number of times a token should appear to be included in vocab.")

  /** @group setParam */
  def setMinCount(threshold: Double): this.type = set(minCount, threshold)

  /** Min number of times a compound word should appear to be included in vocab (Default: `5`).
    *
    * @group param
    */
  val compoundCount = new Param[Int](
    this,
    "compoundCount",
    "Min number of times a compound word should appear to be included in vocab.")

  /** @group setParam */
  def setCompoundCount(k: Int): this.type = set(compoundCount, k)

  /** Tradeoff between the cost of a word error and a transition in the language model (Default:
    * `18.0f`).
    *
    * @group param
    */
  val tradeoff = new Param[Float](
    this,
    "tradeoff",
    "Tradeoff between the cost of a word error and a transition in the language model.")

  /** @group setParam */
  def setTradeoff(alpha: Float): this.type = set(tradeoff, alpha)

  /** Min number of times the word need to appear in corpus to not be considered of a special
    * class (Default: `15.0`).
    *
    * @group param
    */
  val classCount = new Param[Double](
    this,
    "classCount",
    "Min number of times the word need to appear in corpus to not be considered of a special class.")

  /** @group setParam */
  def setClassCount(t: Double): this.type = set(classCount, t)

  /** The path to the file containing the weights for the levenshtein distance.
    *
    * @group param
    */
  val weightedDistPath = new Param[String](
    this,
    "weightedDistPath",
    "The path to the file containing the weights for the levenshtein distance.")

  /** @group setParam */
  def setWeightedDistPath(filePath: String): this.type = set(weightedDistPath, filePath)

  /** Maximum size for the window used to remember history prior to every correction (Default:
    * `5`).
    *
    * @group param
    */
  val maxWindowLen = new IntParam(
    this,
    "maxWindowLen",
    "Maximum size for the window used to remember history prior to every correction.")

  /** @group setParam */
  def setMaxWindowLen(w: Int): this.type = set(maxWindowLen, w)

  /** Configproto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group param
    */
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): ContextSpellCheckerApproach.this.type =
    set(this.configProtoBytes, bytes)

  /** @group getParam */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** Maximum length for a sentence - internal use during training (Default: `250`)
    *
    * @group param
    */
  val maxSentLen = new IntParam(
    this,
    "maxSentLen",
    "Maximum length for a sentence - internal use during training")

  /** Folder path that contain external graph files
    *
    * @group setParam
    */
  val graphFolder =
    new Param[String](this, "graphFolder", "Folder path that contain external graph files")

  /** Folder path that contain external graph files
    *
    * @group setParam
    */
  def setGraphFolder(path: String): this.type = set(this.graphFolder, path)

  setDefault(
    minCount -> 3.0,
    specialClasses -> List(new DateToken, new NumberToken),
    wordMaxDistance -> 3,
    maxCandidates -> 6,
    languageModelClasses -> 2000,
    compoundCount -> 5,
    tradeoff -> 18.0f,
    maxWindowLen -> 5,
    inputCols -> Array("token"),
    epochs -> 2,
    batchSize -> 24,
    classCount -> 15.0,
    initialRate -> .7f,
    finalRate -> 0.0005f,
    validationFraction -> .1f,
    maxSentLen -> 250,
    caseStrategy -> CandidateStrategy.ALL,
    errorThreshold -> 10f)

  private var broadcastGraph: Option[Broadcast[Array[Byte]]] = None

  /** Adds a new class of words to correct, based on a vocabulary.
    *
    * @param usrLabel
    *   Name of the class
    * @param vocabList
    *   Vocabulary as a list
    * @param userDist
    *   Maximal distance to the word
    * @return
    */
  def addVocabClass(
      usrLabel: String,
      vocabList: util.ArrayList[String],
      userDist: Int = 3): ContextSpellCheckerApproach.this.type = {
    import scala.collection.JavaConverters._
    val vocab = vocabList.asScala.to[collection.mutable.Set]
    val nc = new GenericVocabParser(vocab, usrLabel, userDist)
    setSpecialClasses(getOrDefault(specialClasses) :+ nc)
  }

  /** Adds a new class of words to correct, based on regex.
    *
    * @param usrLabel
    *   Name of the class
    * @param usrRegex
    *   Regex to add
    * @param userDist
    *   Maximal distance to the word
    * @return
    */
  def addRegexClass(
      usrLabel: String,
      usrRegex: String,
      userDist: Int = 3): ContextSpellCheckerApproach.this.type = {
    val nc = new GenericRegexParser(usrRegex, usrLabel, userDist)
    setSpecialClasses(getOrDefault(specialClasses) :+ nc)
  }

  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): ContextSpellCheckerModel = {

    val (vocabulary, classes) = genVocab(dataset)
    val word2ids = vocabulary.keys.toList.sorted.zipWithIndex.toMap
    val encodedClasses = classes.map { case (word, cwid) => word2ids(word) -> cwid }

    // split in validation and train
    val trainFraction = 1.0 - getOrDefault(validationFraction)

    val Array(validation, train) =
      dataset.randomSplit(Array(getOrDefault(validationFraction), trainFraction))

    val graph =
      findAndLoadGraph(getOrDefault(languageModelClasses), vocabulary.size, dataset.sparkSession)

    // create transducers for special classes
    val specialClassesTransducers = getOrDefault(specialClasses).par.map { t =>
      t.setTransducer(t.generateTransducer)
    }.seq

    // training
    val tf = new TensorflowWrapper(
      Variables(Array.empty[Array[Byte]], Array.empty[Byte]),
      graph.toGraphDef.toByteArray)
    val model = new TensorflowSpell(tf, Verbose.Silent)
    model.train(
      encodeCorpus(train, word2ids, encodedClasses),
      encodeCorpus(validation, word2ids, encodedClasses),
      getOrDefault(epochs),
      getOrDefault(batchSize),
      getOrDefault(initialRate),
      getOrDefault(finalRate))

    val contextModel = new ContextSpellCheckerModel()
      .setVocabFreq(vocabulary.toMap)
      .setVocabIds(word2ids)
      .setClasses(classes.map { case (k, v) => (word2ids(k), v) })
      .setVocabTransducer(createTransducer(vocabulary.keys.toList))
      .setSpecialClassesTransducers(specialClassesTransducers)
      .setModelIfNotSet(dataset.sparkSession, tf)
      .setInputCols(getOrDefault(inputCols))
      .setWordMaxDistance($(wordMaxDistance))
      .setErrorThreshold($(errorThreshold))

    if (get(configProtoBytes).isDefined)
      contextModel.setConfigProtoBytes($(configProtoBytes))

    get(weightedDistPath)
      .map(path => contextModel.setWeights(loadWeights(path)))
      .getOrElse(contextModel)
  }

  def computeClasses(
      vocab: mutable.HashMap[String, Double],
      total: Double,
      k: Int): Map[String, (Int, Int)] = {
    val sorted = vocab.toList.sortBy(_._2).reverse
    val binMass = total / k

    var acc = 0.0
    var currBinLimit = binMass
    var currClass = 0
    var currWordId = 0

    var classes = Map[String, (Int, Int)]()
    var maxWid = 0
    for (word <- sorted) {
      if (acc < currBinLimit) {
        acc += word._2
        classes = classes.updated(word._1, (currClass, currWordId))
        currWordId += 1
      } else {
        acc += word._2
        currClass += 1
        currBinLimit = (currClass + 1) * binMass
        classes = classes.updated(word._1, (currClass, 0))
        currWordId = 1
      }
      if (currWordId > maxWid)
        maxWid = currWordId
    }

    require(
      maxWid < k,
      "The language model is unbalanced, pick a larger" +
        s" number of classes using setLanguageModelClasses(), current value is ${getOrDefault(languageModelClasses)}.")

    logger.info(s"Max num of words per class: $maxWid")
    classes
  }

  /*
   *  here we do some pre-processing of the training data, and generate the vocabulary
   * */
  def genVocab(
      dataset: Dataset[_]): (mutable.HashMap[String, Double], Map[String, (Int, Int)]) = {

    import dataset.sparkSession.implicits._
    // for every sentence we have one end and one begining
    val eosBosCount = dataset.count()

    var vocab = collection.mutable.HashMap(
      dataset
        .select(getInputCols.head)
        .as[Seq[Annotation]]
        .flatMap(identity)
        .groupByKey(_.result)
        .mapGroups { case (token, insts) => (token, insts.length.toDouble) }
        .collect(): _*)

    // words appearing less that minCount times will be unknown
    val unknownCount = vocab.filter(_._2 < getOrDefault(minCount)).values.sum

    // remove unknown tokens
    vocab = vocab.filter(_._2 >= getOrDefault(minCount))

    // second pass: identify tokens that belong to special classes, and replace with a label
    vocab.foreach { case (word, count) =>
      getOrDefault(specialClasses).foreach { specialClass =>
        // check word is in vocabulary and the word is uncommon
        if (specialClass.inVocabulary(word)) {
          if (count < getOrDefault(classCount)) {
            logger.debug(s"Recognized $word as ${specialClass.label}")
            vocab
              .get(word)
              .map { count =>
                vocab.update(specialClass.label, count + 1.0)
              }
              .getOrElse(vocab.update(specialClass.label, 1.0))

            // remove the token from global vocabulary, now it's covered by the class
            vocab.remove(word)
          } else {
            specialClass match {
              // remove the word from the class, it's already covered by the global vocabulary
              case p: VocabParser => p.vocab.remove(word)
              case _ =>
            }
          }
        }
      }
    }

    /* Blacklists {fwis, hyphen, slash} */
    // words that appear with first letter capitalized (e.g., at the beginning of sentence)
    val fwis = vocab
      .filter(_._1.length > 1)
      .filter(_._1.head.isUpper)
      .filter(w => vocab.contains(w._1.head.toLower + w._1.tail))
      .keys

    // Remove compound words for which components parts are in vocabulary, keep only the high frequent ones
    val compoundWords = Seq("-", "/").flatMap { separator =>
      vocab.filter { case (word, weight) =>
        val splits = word.split(separator)
        splits.length == 2 && vocab.contains(splits(0)) && vocab.contains(splits(1)) &&
        weight < getOrDefault(compoundCount)
      }.keys
    }

    val blacklist = fwis ++ compoundWords
    blacklist.foreach {
      vocab.remove
    }

    vocab.update("_BOS_", eosBosCount)
    vocab.update("_EOS_", eosBosCount)
    vocab.update("_UNK_", unknownCount)

    // count all occurrences of all tokens
    var totalCount = vocab.values.sum + unknownCount
    val classes = computeClasses(vocab, totalCount, getOrDefault(languageModelClasses))

    // compute frequencies - logarithmic
    totalCount = math.log(totalCount)
    for (key <- vocab.keys) {
      vocab.update(key, math.log(vocab(key)) - totalCount)
    }
    logger.info(s"Vocabulary size: ${vocab.size}")
    (vocab, classes)
  }

  private def persistVocab(v: List[(String, Double)], fileName: String) = {
    // vocabulary + frequencies
    val vocabFile = new File(fileName)
    val bwVocab = new BufferedWriter(new FileWriter(vocabFile))

    v.foreach { case (word, freq) =>
      bwVocab.write(s"""$word|$freq""")
      bwVocab.newLine()
    }
    bwVocab.close()
    v
  }

  /*
   * creates the transducer for the vocabulary
   *
   * */
  private def createTransducer(vocab: List[String]) = {
    import scala.collection.JavaConverters._
    new TransducerBuilder()
      .dictionary(vocab.sorted.asJava, true)
      .algorithm(Algorithm.TRANSPOSITION)
      .defaultMaxDistance(getOrDefault(wordMaxDistance))
      .includeDistance(true)
      .build[Candidate]
  }

  /*
   *  receives the corpus, the vocab mappings, the class info, and returns an encoded
   *  version of the training dataset
   */
  private def encodeCorpus(
      corpus: Dataset[_],
      vMap: Map[String, Int],
      classes: Map[Int, (Int, Int)]): Iterator[Array[LangModelSentence]] = {

    object DatasetIterator extends Iterator[Array[LangModelSentence]] {

      import com.johnsnowlabs.nlp.annotators.common.DatasetHelpers._
      import corpus.sparkSession.implicits._

      // Send batches, don't collect(), only keeping a single batch in memory anytime
      val it: util.Iterator[Array[Annotation]] = corpus
        .select(getInputCols.head)
        .randomize // to improve training
        .as[Array[Annotation]]
        .toLocalIterator()

      // create a batch
      override def next(): Array[LangModelSentence] = {
        var count = 0
        var thisBatch = Array.empty[LangModelSentence]

        while (it.hasNext && count < getOrDefault(batchSize)) {
          count += 1
          val next = it.next
          val ids = Array(vMap("_BOS_")) ++ next.map { token =>
            var tmp = token.result
            // identify tokens that belong to special classes, and replace with a label
            getOrDefault(specialClasses).foreach { specialClass =>
              tmp = specialClass.replaceWithLabel(tmp)
            }
            // replace the word by it's id(or the word class' id)
            vMap.getOrElse(tmp, vMap("_UNK_"))
          } ++ Array(vMap("_EOS_"))

          val cids = ids.map(id => classes(id)._1).tail
          val cwids = ids.map(id => classes(id)._2).tail
          val len = ids.length

          thisBatch = thisBatch :+ LangModelSentence(
            ids.dropRight(1).fixSize,
            cids.fixSize,
            cwids.fixSize,
            len)
        }
        thisBatch
      }

      override def hasNext: Boolean = it.hasNext
    }
    DatasetIterator
  }

  private val graphFilePattern = ".*nlm_([0-9]{3})_([0-9]{1,2})_([0-9]{2,4})_([0-9]{3,6})\\.pb".r

  private def findAndLoadGraph(requiredClassCount: Int, vocabSize: Int, spark: SparkSession) = {

    val graph = new Graph()

    if (broadcastGraph.isEmpty) {
      val availableGraphs = getGraphFiles(get(graphFolder))
      // get the one that better matches the class count
      val candidates = availableGraphs
        .map {
          // not looking into innerLayerSize or layerCount
          case filename @ graphFilePattern(_, _, classCount, vSize) =>
            val isValid = classCount.toInt >= requiredClassCount && vocabSize < vSize.toInt
            val score = classCount.toFloat / requiredClassCount
            (filename, score, isValid)
          case _ =>
            ("", 0f, false)
        }
        .filter(_._3)

      require(
        candidates.nonEmpty,
        s"We couldn't find any suitable graph for $requiredClassCount classes, vocabSize: $vocabSize")

      val bestGraph = candidates.minBy(_._2)._1
      val graphStream = ResourceHelper.getResourceStream(bestGraph)
      val graphBytesDef = IOUtils.toByteArray(graphStream)
      broadcastGraph = Some(spark.sparkContext.broadcast(graphBytesDef))
    }

    graph.importGraphDef(GraphDef.parseFrom(broadcastGraph.get.value))
    graph
  }

  implicit class ArrayHelper(array: Array[Int]) {
    def fixSize: Array[Int] = {
      val maxLen = getOrDefault(maxSentLen)
      if (array.length < maxLen)
        array.padTo(maxLen, 0)
      else
        array.dropRight(array.length - maxLen)
    }
  }

  private def getGraphFiles(localGraphPath: Option[String]): Seq[String] = {
    val graphFiles = localGraphPath
      .map(path =>
        ResourceHelper
          .listLocalFiles(ResourceHelper.copyToLocal(path))
          .map(_.getAbsolutePath))
      .getOrElse(ResourceHelper.listResourceDirectory("/spell_nlm"))

    graphFiles
  }

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("SPELL"))

  /** Input Annotator Types: TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.TOKEN)

  /** Output Annotator Types: TOKEN
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.TOKEN

}
