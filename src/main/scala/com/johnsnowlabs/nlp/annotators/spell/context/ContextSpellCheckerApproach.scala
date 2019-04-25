package com.johnsnowlabs.nlp.annotators.spell.context

import java.io.{BufferedWriter, File, FileWriter}

import com.github.liblevenshtein.transducer.{Algorithm, Candidate}
import com.github.liblevenshtein.transducer.factory.TransducerBuilder
import com.johnsnowlabs.ml.tensorflow.TensorflowWrapper
import com.johnsnowlabs.nlp.annotators.common.{PrefixedToken, SuffixedToken}
import com.johnsnowlabs.nlp.annotators.spell.context.parser._
import com.johnsnowlabs.nlp.serialization.ArrayFeature
import com.johnsnowlabs.nlp.{AnnotatorApproach, AnnotatorType, HasFeatures}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{IntParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset
import org.slf4j.LoggerFactory
import org.tensorflow.{Graph, Session}

import scala.collection.mutable
import scala.io.Codec

case class OpenClose(open:String, close:String)

class ContextSpellCheckerApproach(override val uid: String) extends
  AnnotatorApproach[ContextSpellCheckerModel]
  with HasFeatures
  with WeightedLevenshtein {

  override val description: String = "Context Spell Checker"

  private val logger = LoggerFactory.getLogger("ContextSpellCheckerApproach")

  val trainCorpusPath = new Param[String](this, "trainCorpusPath", "Path to the training corpus text file.")
  def setTrainCorpusPath(path: String): this.type = set(trainCorpusPath, path)

  val specialClasses = new Param[List[SpecialClassParser]](this, "specialClasses", "List of parsers for special classes.")
  def setSpecialClasses(parsers: List[SpecialClassParser]):this.type = set(specialClasses, parsers)

  val languageModelClasses = new Param[Int](this, "languageModelClasses", "Number of classes to use during factorization of the softmax output in the LM.")
  def setLMClasses(k: Int):this.type = set(languageModelClasses, k)

  val prefixes = new ArrayFeature[String](this, "prefixes")
  def setPrefixes(p: Array[String]):this.type = set(prefixes, p.sortBy(_.length).reverse)

  val suffixes = new ArrayFeature[String](this, "suffixes")
  def setSuffixes(s: Array[String]):this.type = set(suffixes, s.sortBy(_.length).reverse)

  val wordMaxDistance = new IntParam(this, "wordMaxDistance", "Maximum distance for the generated candidates for every word.")
  def setWordMaxDist(k: Int):this.type = {
    require(k >= 1, "Please provided a minumum candidate distance of at least 1.")
    set(wordMaxDistance, k)
  }

  val maxCandidates = new IntParam(this, "maxCandidates", "Maximum number of candidates for every word.")
  def setMaxCandidates(k: Int):this.type = set(maxCandidates, k)

  val minCount = new Param[Double](this, "minCount", "Min number of times a token should appear to be included in vocab.")
  def setMinCount(threshold: Double): this.type = set(minCount, threshold)

  val blacklistMinFreq = new Param[Int](this, "blacklistMinFreq", "Minimun number of occurrences for a word not to be blacklisted.")
  def setBlackListMinFreq(k: Int):this.type = set(blacklistMinFreq, k)

  val tradeoff = new Param[Float](this, "tradeoff", "Tradeoff between the cost of a word and a transition in the language model.")
  def setTradeoff(alpha: Float):this.type = set(tradeoff, alpha)

  val weightedDistPath = new Param[String](this, "weightedDistPath", "The path to the file containing the weights for the levenshtein distance.")
  def setWeights(filePath:String):this.type = set(weightedDistPath, filePath)

  val maxWindowLen = new IntParam(this, "maxWindowLen", "Maximum size for the window used to remember history prior to every correction.")
  def setMaxWindowLen(w: Int):this.type = set(maxWindowLen, w)


  setDefault(minCount -> 3.0,
    specialClasses -> List(DateToken, NumberToken),
    wordMaxDistance -> 3,
    maxCandidates -> 6,
    languageModelClasses -> 2000,
    blacklistMinFreq -> 5,
    tradeoff -> 18.0f,
    maxWindowLen -> 5
  )

  setDefault(prefixes, () => Array("'"))
  setDefault(suffixes, () => Array(".", ":", "%", ",", ";", "?", "'"))

  val openClose = List(OpenClose("(", ")"), OpenClose("[", "]"), OpenClose("\"", "\""))

  private lazy val firstPass = Seq(SuffixedToken($$(suffixes) ++ openClose.map(_.close)),
    PrefixedToken($$(prefixes) ++ openClose.map(_.open)))

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): ContextSpellCheckerModel = {

    val graph = new Graph()
    //val config = Array[Byte](56, 1)
    //val config = Array[Byte](50, 2, 32, 1, 56, 1, 64, 1)
    val config = Array[Byte](50, 2, 32, 1, 56, 1)
    val session = new Session(graph, config)
    //val tf = new TensorflowWrapper(session, graph)

    // extract vocabulary
    require(isDefined(trainCorpusPath), "Train corpus must be set before training")
    val rawTextPath = getOrDefault(trainCorpusPath)
    val vPath = s"${getOrDefault(trainCorpusPath)}.vocab"

    val (vocabFreq, word2ids, classes) =
      // either we load vocab, word2id and classes, or create them from scratch
      if (new File(vPath).exists())
        loadVocab(vPath)
      else {
        val (vocabulary, classes) = genVocab(rawTextPath)
        persistVocab(vocabulary, vPath)
        val w2i = vocabulary.map(_._1).sorted.zipWithIndex.toMap
        encodeCorpus(rawTextPath, w2i)
        (vocabulary.toMap, w2i, classes.map{case (k,v) => w2i.apply(k) -> v})
      }

    // create transducers for special classes
    val specialClassesTransducers = getOrDefault(specialClasses).
      par.map{t => t.setTransducer(t.generateTransducer)}.seq

    val model = new ContextSpellCheckerModel().
      setVocabFreq(vocabFreq.toMap).
      setVocabIds(word2ids.toMap).
      setClasses(classes).
      setVocabTransducer(createTransducer(vocabFreq.keys.toList)).
      setSpecialClassesTransducers(specialClassesTransducers).
      //setModelIfNotSet(dataset.sparkSession, tf).
      setInputCols(getOrDefault(inputCols)).
      setWordMaxDist($(wordMaxDistance))

    get(weightedDistPath).map(path => model.setWeights(loadWeights(path))).
    getOrElse(model)
  }

  private def loadVocab(path: String) = {
    // store individual frequencies of words
    val vocabFreq = mutable.HashMap[String, Double]()

    // store word ids
    val vocabIdxs = mutable.HashMap[String, Int]()

    scala.io.Source.fromFile(path).getLines.zipWithIndex.foreach { case (line, idx) =>
       val lineFields = line.split("\\|")
       vocabFreq += (lineFields(0)-> lineFields.last.toDouble)
       vocabIdxs += (lineFields(0)-> idx)
    }

    val classes = scala.io.Source.fromFile(s"${getOrDefault(trainCorpusPath)}.classes").getLines.map{line =>
      val chunks = line.split("\\|")
      val key = chunks(0).toInt
      val cid = chunks(1).toInt
      val wcid = chunks(2).toInt
      (key, (cid, wcid))
    }.toMap

    (vocabFreq, vocabIdxs, classes)
  }

  def computeAndPersistClasses(vocab: mutable.HashMap[String, Double], total:Double, k:Int) = {
    val filePath = s"${getOrDefault(trainCorpusPath)}.classes"
    val sorted = vocab.toList.sortBy(_._2).reverse
    val word2id = vocab.toList.sortBy(_._1).map(_._1).zipWithIndex.toMap
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

    logger.info(s"Max num of words per class: $maxWid")

    val classesFile = new File(filePath)
    val bwClassesFile = new BufferedWriter(new FileWriter(classesFile))

    classes.foreach{case (word, (cid, wid)) =>
      bwClassesFile.write(s"""${word2id.apply(word)}|$cid|$wid""")
      bwClassesFile.newLine()
    }
    bwClassesFile.close()
    classes
  }

  /*
  *  here we do some pre-processing of the training data, and generate the vocabulary
  * */

  def genVocab(rawDataPath: String) = {
    var vocab = mutable.HashMap[String, Double]()

    implicit val codec: Codec = Codec.UTF8

    // for every sentence we have one end and one begining
    val eosBosCount = scala.io.Source.fromFile(rawDataPath).getLines.size

    scala.io.Source.fromFile(rawDataPath).getLines.foreach { line =>
      // TODO remove crazy encodings of space(clean the dataset itself before input it here)
      line.split(" ").flatMap(_.split(" ")).flatMap(_.split(" ")).filter(_!=" ").foreach { token =>
        var tmp = Seq(token)

        // first pass: separate suffixes, prefixes, etc
        firstPass.foreach{ parser =>
          tmp = tmp.flatMap(_.split(" ").map(_.trim)).map(parser.separate).flatMap(_.split(" "))
        }

        // second pass: identify tokens that belong to special classes, and replace with a label
        getOrDefault(specialClasses).foreach { specialClass =>
          tmp = tmp.map(specialClass.replaceWithLabel)
        }

        // count frequencies
        tmp.foreach {cleanToken =>
          val currCount = vocab.getOrElse(cleanToken, 0.0)
          vocab.update(cleanToken, currCount + 1.0)
        }
      }
    }

    // words appearing less that minCount times will be unknown
    val unknownCount = vocab.filter(_._2 < getOrDefault(minCount)).values.sum

    // remove unknown tokens
    vocab = vocab.filter(_._2 >= getOrDefault(minCount))

    // Blacklists {fwis, hyphen, slash}
    // words that appear with first letter capitalized (e.g., at the beginning of sentence)
    val fwis = vocab.filter(_._1.length > 1).filter(_._1.head.isUpper).
      filter(w => vocab.contains(w._1.head.toLower + w._1.tail)).keys

    val hyphen = vocab.filter {
      case (word, weight) =>
        val splits = word.split("-")
        splits.length == 2 && vocab.contains(splits(0)) && vocab.contains(splits(1)) &&
          vocab.get(word).forall(_ < getOrDefault(blacklistMinFreq))
    }.keys

    val slash = vocab.filter {
      case (word, weight) =>
        val splits = word.split("/")
        splits.length == 2 && vocab.contains(splits(0)) && vocab.contains(splits(1)) &&
          vocab.get(word).forall(_ < getOrDefault(blacklistMinFreq))
    }.keys

    val blacklist = fwis ++ hyphen ++ slash
    blacklist.foreach{vocab.remove}

    vocab.update("_BOS_", eosBosCount)
    vocab.update("_EOS_", eosBosCount)
    vocab.update("_UNK_", unknownCount)

    // count all occurrences of all tokens
    var totalCount = vocab.values.sum + eosBosCount * 2 + unknownCount
    val classes = computeAndPersistClasses(vocab, totalCount, getOrDefault(languageModelClasses))

    // compute frequencies - logarithmic
    totalCount = math.log(totalCount)
    for (key <- vocab.keys){
      vocab.update(key, math.log(vocab(key)) - totalCount)
    }
    (vocab.toList.sortBy(_._1), classes)
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
      algorithm(Algorithm.STANDARD).
      defaultMaxDistance(getOrDefault(wordMaxDistance)).
      includeDistance(true).
      build[Candidate]
  }


  private def encodeCorpus(rawTextPath: String, vMap: Map[String, Int]) = {

    // path to the encoded corpus
    val bw = new BufferedWriter(new FileWriter(new File(rawTextPath + ".ids")))

    scala.io.Source.fromFile(rawTextPath).getLines.foreach { line =>
      // TODO removing crazy encodings of space and replacing with standard one - should be done outside Scala
      val text  = line.split(" ").flatMap(_.split(" ")).flatMap(_.split(" ")).filter(_!=" ").flatMap { token =>
        var tmp = token
        firstPass.foreach{ parser =>
          tmp = parser.separate(tmp)
        }
        // second pass identify tokens that belong to special classes, and replace with a label
        getOrDefault(specialClasses).foreach { specialClass =>
          tmp = specialClass.replaceWithLabel(tmp)
        }

        tmp.split(" ").filter(_ != " ").map {cleanToken =>
          s"""${vMap.getOrElse(cleanToken, vMap("_UNK_")).toString}"""
        }
      }.mkString(" ")
      bw.write(s"""${vMap("_BOS_")} $text ${vMap("_EOS_")}\n""")
    }
    bw.close()
    vMap
  }

  def this() = this(Identifiable.randomUID("SPELL"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.TOKEN)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.TOKEN

}
