package com.johnsnowlabs.nlp.annotators.spell.context

import java.io.{BufferedWriter, File, FileWriter}
import java.util

import com.github.liblevenshtein.transducer.{Algorithm, Candidate, ITransducer}
import com.github.liblevenshtein.transducer.factory.TransducerBuilder
import com.johnsnowlabs.ml.tensorflow.{TensorflowSpell, TensorflowWrapper, Variables}
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.annotators.spell.context.parser._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach, AnnotatorType, HasFeatures}
import org.apache.commons.io.IOUtils
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.slf4j.LoggerFactory
import org.tensorflow.Graph

import scala.collection.JavaConversions._
import scala.collection.mutable



case class LangModelSentence(ids: Array[Int], cids: Array[Int], cwids:Array[Int], len: Int)

object CandidateStrategy {
  val ALL_UPPER_CASE = 0
  val FIRST_LETTER_CAPITALIZED = 1
  val ALL = 2
}

class ContextSpellCheckerApproach(override val uid: String) extends
  AnnotatorApproach[ContextSpellCheckerModel]
  with HasFeatures
  with WeightedLevenshtein {

  override val description: String = "Context Spell Checker"

  private val logger = LoggerFactory.getLogger("ContextSpellCheckerApproach")

  val specialClasses = new Param[List[SpecialClassParser]](this, "specialClasses", "List of parsers for special classes.")
  def setSpecialClasses(parsers: List[SpecialClassParser]):this.type = set(specialClasses, parsers)

  val languageModelClasses = new Param[Int](this, "languageModelClasses", "Number of classes to use during factorization of the softmax output in the LM.")
  def setLMClasses(k: Int):this.type = set(languageModelClasses, k)

  val wordMaxDistance = new IntParam(this, "wordMaxDistance", "Maximum distance for the generated candidates for every word.")
  def setWordMaxDist(k: Int):this.type = {
    require(k >= 1, "Please provided a minumum candidate distance of at least 1.")
    set(wordMaxDistance, k)
  }

  val maxCandidates = new IntParam(this, "maxCandidates", "Maximum number of candidates for every word.")
  def setMaxCandidates(k: Int):this.type = set(maxCandidates, k)

  val caseStrategy = new IntParam(this, "caseStrategy", "What case combinations to try when generating candidates.")
  def setCaseStrategy(k: Int):this.type = set(caseStrategy, k)

  val errorThreshold = new FloatParam(this, "errorThreshold", "Threshold perplexity for a word to be considered as an error.")
  def setErrorThreshold(t: Float):this.type = set(errorThreshold, t)

  val epochs = new IntParam(this, "epochs", "Number of epochs to train the language model.")
  def setEpochs(k: Int):this.type = set(epochs, k)

  val batchSize = new IntParam(this, "batchSize", "Batch size for the training in NLM.")
  def setBatchSize(k: Int):this.type = set(batchSize, k)

  val initialRate = new FloatParam(this, "initialRate", "Initial learning rate for the LM.")
  def setInitialLearningRate(r: Float):this.type = set(initialRate, r)

  val finalRate = new FloatParam(this, "finalRate", "Final learning rate for the LM.")
  def setFinalLearningRate(r: Float):this.type = set(finalRate, r)

  val validationFraction = new FloatParam(this, "validationFraction", "percentage of datapoints to use for validation.")
  def setValidationFraction(r: Float):this.type = set(validationFraction, r)

  val minCount = new Param[Double](this, "minCount", "Min number of times a token should appear to be included in vocab.")
  def setMinCount(threshold: Double): this.type = set(minCount, threshold)

  val compoundCount = new Param[Int](this, "compoundCount", "Min number of times a compound word should appear to be included in vocab.")
  def setBlackListMinFreq(k: Int):this.type = set(compoundCount, k)

  val tradeoff = new Param[Float](this, "tradeoff", "Tradeoff between the cost of a word error and a transition in the language model.")
  def setTradeoff(alpha: Float):this.type = set(tradeoff, alpha)

  val classCount = new Param[Double](this, "classCount", "Min number of times the word need to appear in corpus to not be considered of a special class.")
  def setClassThreshold(t: Double):this.type = set(classCount, t)

  val weightedDistPath = new Param[String](this, "weightedDistPath", "The path to the file containing the weights for the levenshtein distance.")
  def setWeights(filePath:String):this.type = set(weightedDistPath, filePath)

  val maxWindowLen = new IntParam(this, "maxWindowLen", "Maximum size for the window used to remember history prior to every correction.")
  def setMaxWindowLen(w: Int):this.type = set(maxWindowLen, w)

  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")
  def setConfigProtoBytes(bytes: Array[Int]) = set(this.configProtoBytes, bytes)
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  val maxSentLen = new IntParam(this, "maxSentLen", "Maximum length for a sentence - internal use during training")



  setDefault(minCount -> 3.0,
    specialClasses -> List(DateToken, NumberToken),
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
    errorThreshold -> 10f
  )

  def addVocabClass(usrLabel:String, vocabList:util.ArrayList[String], userDist: Int=3) = {
    val newClass = new VocabParser with Serializable {
      override var vocab: mutable.Set[String] = scala.collection.mutable.Set(vocabList.toArray.map(_.toString): _*)
      override val label: String = usrLabel
      override var transducer: ITransducer[Candidate] = generateTransducer
      override val maxDist: Int = userDist
    }
    setSpecialClasses(getOrDefault(specialClasses):+newClass)
  }

  def addRegexClass(usrLabel:String, usrRegex:String, userDist: Int=3) = {
    val newClass = new RegexParser with Serializable {
      override var regex: String = usrRegex
      override val label: String = usrLabel
      override var transducer: ITransducer[Candidate] = generateTransducer
      override val maxDist: Int = userDist
    }
    setSpecialClasses(getOrDefault(specialClasses):+newClass)
  }

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): ContextSpellCheckerModel = {

    val (vocabulary, classes) = genVocab(dataset)
    val word2ids = vocabulary.keys.toList.sorted.zipWithIndex.toMap
    val encodedClasses = classes.map { case (word, cwid) => word2ids.get(word).get -> cwid }

    // split in validation and train
    val trainFraction = 1.0 - getOrDefault(validationFraction)
    val Array(validation, train) = dataset.randomSplit(Array(getOrDefault(validationFraction), trainFraction))
    val graph = findAndLoadGraph(getOrDefault(languageModelClasses), vocabulary.size)

    // create transducers for special classes
    val specialClassesTransducers = getOrDefault(specialClasses).
      par.map{t => t.setTransducer(t.generateTransducer)}.seq

    // training
    val tf = new TensorflowWrapper(Variables(Array.empty[Byte], Array.empty[Byte]), graph.toGraphDef.toByteArray)
    val model = new TensorflowSpell(tf, Verbose.Silent)
    model.train(encodeCorpus(train, word2ids, encodedClasses), encodeCorpus(validation, word2ids, encodedClasses),
      getOrDefault(epochs), getOrDefault(batchSize), getOrDefault(initialRate), getOrDefault(finalRate))

    val contextModel = new ContextSpellCheckerModel().
      setVocabFreq(vocabulary.toMap).
      setVocabIds(word2ids).
      setClasses(classes.map{case (k,v) => (word2ids(k),  v)}.toMap).
      setVocabTransducer(createTransducer(vocabulary.keys.toList)).
      setSpecialClassesTransducers(specialClassesTransducers).
      setModelIfNotSet(dataset.sparkSession, tf).
      setInputCols(getOrDefault(inputCols)).
      setWordMaxDist($(wordMaxDistance))
      setErrorThreshold($(errorThreshold))

    if (get(configProtoBytes).isDefined)
      contextModel.setConfigProtoBytes($(configProtoBytes))

    get(weightedDistPath).map(path => contextModel.setWeights(loadWeights(path))).
    getOrElse(contextModel)
  }


  def computeClasses(vocab: mutable.HashMap[String, Double], total:Double, k:Int) = {
    val sorted = vocab.toList.sortBy(_._2).reverse
    val binMass = total / k

    var acc = 0.0
    var currBinLimit = binMass
    var currClass = 0
    var currWordId = 0

    var classes = Map[String, (Int, Int)]()
    var maxWid = 0
    for(word <-sorted) {
      if(acc < currBinLimit){
        acc += word._2
        classes = classes.updated(word._1, (currClass, currWordId))
        currWordId += 1
      }
      else{
        acc += word._2
        currClass += 1
        currBinLimit = (currClass + 1) * binMass
        classes = classes.updated(word._1, (currClass, 0))
        currWordId = 1
      }
      if (currWordId > maxWid)
        maxWid = currWordId
    }

    require(maxWid  < k, "The language model is unbalanced, pick a larger" +
        s" number of classes using setLMClasses(), current value is ${getOrDefault(languageModelClasses)}.")

    logger.info(s"Max num of words per class: $maxWid")
    classes
  }

  /*
  *  here we do some pre-processing of the training data, and generate the vocabulary
  * */

  def genVocab(dataset: Dataset[_]) = {

    import dataset.sparkSession.implicits._
    // for every sentence we have one end and one begining
    val eosBosCount = dataset.count()

    var vocab = collection.mutable.HashMap(dataset.select(getInputCols.head).as[Seq[Annotation]].
      flatMap(identity).
      groupByKey(_.result).
      mapGroups{case (token, insts) => (token, insts.length.toDouble)}.
      collect(): _*)

    // words appearing less that minCount times will be unknown
    val unknownCount = vocab.filter(_._2 < getOrDefault(minCount)).values.sum

    // remove unknown tokens
    vocab = vocab.filter(_._2 >= getOrDefault(minCount))

    // second pass: identify tokens that belong to special classes, and replace with a label
    vocab.foreach { case (word, count) =>
      getOrDefault(specialClasses).foreach { specialClass =>
        // check word is in vocabulary and the word is uncommon
        if(specialClass.inVocabulary(word)){
          if(count < getOrDefault(classCount)) {
            logger.debug(s"Recognized $word as ${specialClass.label}")
            vocab.get(word).map { count => vocab.update(specialClass.label, count + 1.0)
            }.getOrElse(vocab.update(specialClass.label, 1.0))

            // remove the token from global vocabulary, now it's covered by the class
            vocab.remove(word)
          } else {
            specialClass match {
              // remove the word from the class, it's already covered by the global vocabulary
              case p:VocabParser => p.vocab.remove(word)
              case _ =>
            }
          }
       }
      }
    }

    /* Blacklists {fwis, hyphen, slash} */
    // words that appear with first letter capitalized (e.g., at the beginning of sentence)
    val fwis = vocab.filter(_._1.length > 1).filter(_._1.head.isUpper).
      filter(w => vocab.contains(w._1.head.toLower + w._1.tail)).keys

    // Remove compound words for which components parts are in vocabulary, keep only the high frequent ones
    val compoundWords = Seq("-", "/").flatMap { separator =>
      vocab.filter {
        case (word, weight) =>
          val splits = word.split(separator)
          splits.length == 2 && vocab.contains(splits(0)) && vocab.contains(splits(1)) &&
            weight < getOrDefault(compoundCount)
      }.keys
    }

    val blacklist = fwis ++ compoundWords
    blacklist.foreach{vocab.remove}

    vocab.update("_BOS_", eosBosCount)
    vocab.update("_EOS_", eosBosCount)
    vocab.update("_UNK_", unknownCount)

    // count all occurrences of all tokens
    var totalCount = vocab.values.sum + unknownCount
    val classes = computeClasses(vocab, totalCount, getOrDefault(languageModelClasses))

    // compute frequencies - logarithmic
    totalCount = math.log(totalCount)
    for (key <- vocab.keys){
      vocab.update(key, math.log(vocab(key)) - totalCount)
    }
    logger.info(s"Vocabulary size: ${vocab.size}")
    (vocab, classes)
  }


  private def persistVocab(v: List[(String, Double)], fileName:String) = {
    // vocabulary + frequencies
    val vocabFile = new File(fileName)
    val bwVocab = new BufferedWriter(new FileWriter(vocabFile))

    v.foreach{case (word, freq) =>
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
  private def createTransducer(vocab:List[String]) = {
    import scala.collection.JavaConversions._
    new TransducerBuilder().
      dictionary(vocab.sorted, true).
      algorithm(Algorithm.TRANSPOSITION).
      defaultMaxDistance(getOrDefault(wordMaxDistance)).
      includeDistance(true).
      build[Candidate]
  }

  /*
    *  receives the corpus, the vocab mappings, the class info, and returns an encoded
    *  version of the training dataset
    */
  private def encodeCorpus(corpus: Dataset[_], vMap: Map[String, Int],
    classes:Map[Int, (Int, Int)]):Iterator[Array[LangModelSentence]] = {

    object DatasetIterator extends Iterator[Array[LangModelSentence]] {
      import corpus.sparkSession.implicits._
      import com.johnsnowlabs.nlp.annotators.common.DatasetHelpers._

      // Send batches, don't collect(), only keeping a single batch in memory anytime
      val it = corpus.select(getInputCols.head)
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
          val ids = Array(vMap("_BOS_")) ++ next.map { case token =>
            var tmp = token.result
            // identify tokens that belong to special classes, and replace with a label
            getOrDefault(specialClasses).foreach { specialClass =>
              tmp = specialClass.replaceWithLabel(tmp)
            }
            // replace the word by it's id(or the word class' id)
            vMap.getOrElse(tmp, vMap("_UNK_"))
          } ++ Array(vMap("_EOS_"))

          val cids = ids.map(id => classes.get(id).get._1).tail
          val cwids = ids.map(id => classes.get(id).get._2).tail
          val len = ids.length

          thisBatch = thisBatch :+ LangModelSentence(ids.dropRight(1).fixSize, cids.fixSize, cwids.fixSize, len)
        }
        thisBatch
      }

      override def hasNext: Boolean = it.hasNext
    }
    DatasetIterator
  }

  private val graphFilePattern = "spell_nlm\\/nlm_([0-9]{3})_([0-9]{1,2})_([0-9]{2,4})_([0-9]{3,6})\\.pb".r
  private def findAndLoadGraph(requiredClassCount: Int, vocabSize: Int) = {
    val availableGraphs = ResourceHelper.listResourceDirectory("/spell_nlm")

    // get the one that better matches the class count
    val candidates = availableGraphs.map { filename =>
      filename match {
        // not looking into innerLayerSize or layerCount
        case graphFilePattern(innerLayerSize, layerCount, classCount, vSize) =>
          val isValid = classCount.toInt >= requiredClassCount && vocabSize < vSize.toInt
          val score = classCount.toFloat / requiredClassCount
          (filename, score, isValid)
      } // keep the valid, and pick the best
    }.filter(_._3)

    require(!candidates.isEmpty, s"We couldn't find any suitable graph for $requiredClassCount classes.")

    val bestGraph = candidates.minBy(_._2)._1
    val graph = new Graph()
    val graphStream = ResourceHelper.getResourceStream(bestGraph)
    val graphBytesDef = IOUtils.toByteArray(graphStream)
    //graph.importGraphDef(graphBytesDef)
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

  def this() = this(Identifiable.randomUID("SPELL"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.TOKEN)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.TOKEN

}
