package com.johnsnowlabs.nlp.annotators.spell.context

import java.util
import com.github.liblevenshtein.transducer.{Candidate, ITransducer}
import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.serialization._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.spell.context.parser.{RegexParser, SpecialClassParser, TransducerSeqFeature, VocabParser}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, FloatParam, IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, SparkSession}
import org.slf4j.LoggerFactory


class ContextSpellCheckerModel(override val uid: String) extends AnnotatorModel[ContextSpellCheckerModel] with WithAnnotate[ContextSpellCheckerModel]
  with WeightedLevenshtein
  with WriteTensorflowModel
  with ParamsAndFeaturesWritable
  with HasTransducerFeatures {

  private val logger = LoggerFactory.getLogger("ContextSpellCheckerModel")

  val transducer = new TransducerFeature(this, "mainVocabularyTransducer")
  def setVocabTransducer(trans:ITransducer[Candidate]): this.type = {
    set(transducer, trans)
  }

  val specialTransducers = new TransducerSeqFeature(this, "specialClassesTransducers")
  def setSpecialClassesTransducers(transducers: Seq[SpecialClassParser]): this.type = {
    set(specialTransducers, transducers.toArray)
  }

  val vocabFreq  = new MapFeature[String, Double](this, "vocabFreq")
  def setVocabFreq(v: Map[String, Double]): this.type = set(vocabFreq,v)

  val idsVocab = new MapFeature[Int, String](this, "idsVocab")
  val vocabIds = new MapFeature[String, Int](this, "vocabIds")

  def setVocabIds(v: Map[String, Int]): this.type = {
    set(idsVocab, v.map(_.swap))
    set(vocabIds, v)
  }

  val classes: MapFeature[Int, (Int, Int)] = new MapFeature(this, "classes")
  def setClasses(c:Map[Int, (Int, Int)]): this.type = set(classes, c)

  val wordMaxDistance = new IntParam(this, "wordMaxDistance", "Maximum distance for the generated candidates for every word, minimum 1.")
  def setWordMaxDist(k: Int):this.type = set(wordMaxDistance, k)

  val maxCandidates = new IntParam(this, "maxCandidates", "Maximum number of candidates for every word.")
  def setMaxCandidates(k: Int):this.type = set(maxCandidates, k)

  val caseStrategy = new IntParam(this, "caseStrategy", "What case combinations to try when generating candidates.")
  def setCaseStrategy(k: Int):this.type = set(caseStrategy, k)

  val errorThreshold = new FloatParam(this, "errorThreshold", "Threshold perplexity for a word to be considered as an error.")
  def setErrorThreshold(t: Float):this.type = set(errorThreshold, t)

  val tradeoff = new FloatParam(this, "tradeoff", "Tradeoff between the cost of a word and a transition in the language model.")
  def setTradeOff(lambda: Float):this.type = set(tradeoff, lambda)

  val gamma = new FloatParam(this, "gamma", "Controls the influence of individual word frequency in the decision.")
  def setGamma(g: Float):this.type = set(gamma, g)

  val weights: MapFeature[String, Map[String, Float]] = new MapFeature[String, Map[String, Float]](this, "levenshteinWeights")
  def setWeights(w:Map[String, Map[String, Float]]): this.type = set(weights, w)

  // for Python access
  def setWeights(w:util.HashMap[String, util.HashMap[String, Double]]): this.type = {
    import scala.collection.JavaConverters._
    val ws = w.asScala.mapValues(_.asScala.mapValues(_.toFloat).toMap).toMap
    set(weights, ws)
  }

  val useNewLines = new BooleanParam(this, "trim", "When set to true new lines will be treated as any other character, when set to false" +
    " correction is applied on paragraphs as defined by newline characters.")
  def setUseNewLines(useIt: Boolean):this.type = set(useNewLines, useIt)

  val maxWindowLen = new IntParam(this, "maxWindowLen", "Maximum size for the window used to remember history prior to every correction.")
  def setMaxWindowLen(w: Int):this.type = set(maxWindowLen, w)

  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")
  def setConfigProtoBytes(bytes: Array[Int]) = set(this.configProtoBytes, bytes)
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  val correctSymbols: BooleanParam = new BooleanParam(this, "correctSymbols", "Whether to correct special symbols or skip spell checking for them")
  def setCorrectSymbols(value: Boolean): this.type = set(correctSymbols, value)
  setDefault(
    correctSymbols -> false
  )

  val compareLowcase: BooleanParam = new BooleanParam(this, "compareLowcase", "If true will compare tokens in low case with vocabulary")
  def setCompareLowcase(value: Boolean): this.type = set(compareLowcase, value)
  setDefault(
    compareLowcase -> false
  )

  def getWordClasses() = $$(specialTransducers).map {
    case transducer:RegexParser =>
     (transducer.label, "RegexParser")
    case transducer:VocabParser =>
      (transducer.label, "VocabParser")
  }

  /* update a regex class */
  def updateRegexClass(label: String, regex:String) = {
    val classes = $$(specialTransducers)
    require(classes.count(_.label == label) == 1,
      s"Not found regex class $label. You can only update existing classes.")

    classes.filter(_.label.equals(label)).head match {
      case r:RegexParser =>
        r.regex = regex
        r.generateTransducer
        r.transducer = r.generateTransducer
      case _ => require(false, s"Class $label is not a regex class.")
    }
  }

  /* update a vocabulary class */
  def updateVocabClass(label: String, vocabList:util.ArrayList[String], append:Boolean=true) = {
    val vocab =  scala.collection.mutable.Set(vocabList.toArray.map(_.toString): _*)
    val classes = $$(specialTransducers)
    require(classes.count(_.label == label) == 1,
      s"Not found regex class $label. You can only update existing classes.")

    classes.filter(_.label.equals(label)).head match {
      case v:VocabParser =>
        val newSet = if(append) v.vocab ++ vocab else vocab
        v.vocab = newSet
        v.transducer = v.generateTransducer
      case _ => require(false, s"Class $label is not a vocabulary class.")
    }
  }

  setDefault(tradeoff -> 18.0f,
    gamma -> 120.0f,
    useNewLines -> false,
    maxCandidates -> 6,
    maxWindowLen -> 5,
    caseStrategy -> CandidateStrategy.ALL
  )

  // the scores for the EOS (end of sentence), and BOS (beginning of sentence)
  private val eosScore = .01
  private val bosScore = 1.0


  private var _model: Option[Broadcast[TensorflowSpell]] = None

  def getModelIfNotSet: TensorflowSpell = _model.get.value

  def setModelIfNotSet(spark: SparkSession, tensorflow: TensorflowWrapper): this.type = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowSpell(
            tensorflow,
            Verbose.Silent)
        )
      )
    }
    this
  }


  /* trellis goes like (label, weight, candidate)*/
  def decodeViterbi(trellis: Array[Array[(String, Double, String)]]):(Array[String], Double) = {

    // encode words with ids
    val encTrellis = Array(Array(($$(vocabIds)("_BOS_"), bosScore, "_BOS_"))) ++
      trellis.map(_.map{case (label, weight, cand) =>
        // at this point we keep only those candidates that are in the vocabulary
        ($$(vocabIds).get(label), weight, cand)}.filter(_._1.isDefined).map{case (x,y,z) => (x.get, y, z)}) ++
      Array(Array(($$(vocabIds)("_EOS_"), eosScore, "_EOS_")))

    // init
    var pathsIds = Array(Array($$(vocabIds)("_BOS_")))
    var pathWords = Array(Array("_BOS_"))
    var costs = Array(bosScore) // cost for each of the paths

    for(i <- 1 until encTrellis.length if pathsIds.forall(_.nonEmpty)) {

      var newPaths:Array[Array[Int]] = Array()
      var newWords: Array[Array[String]] = Array()
      var newCosts = Array[Double]()

      /* compute all the costs for all transitions in current step */
      val expPaths = pathsIds.
        map{p => p :+ p.head}. // we need a placeholder, put the head.
        map(_.takeRight($(maxWindowLen)))
      val cids = expPaths.map(_.map{id => $$(classes).apply(id)._1})
      val cwids = expPaths.map(_.map{id => $$(classes).apply(id)._2})

      val candCids = encTrellis(i).map(_._1).map{id => $$(classes).apply(id)._1}
      val candWids = encTrellis(i).map(_._1).map{id => $$(classes).apply(id)._2}
      val expPathsCosts_ = getModelIfNotSet.predict_(pathsIds.map(_.takeRight($(maxWindowLen))), cids, cwids, candCids, candWids, configProtoBytes=getConfigProtoBytes).toArray


      for {((state, wcost, cand), idx) <- encTrellis(i).zipWithIndex} {
        var minCost = Double.MaxValue
        var minPath = Array[Int]()
        var minWords = Array[String]()

        val z = (pathsIds, costs, pathWords).zipped.toList

        for (((path, pathCost, cands), pi) <- z.zipWithIndex) {
          // compute cost to arrive to this 'state' coming from that 'path'
          val mult = if (i > 1) costs.length else 0
          val ppl_ = expPathsCosts_(encTrellis(i).size * pi + idx)

          val cost = pathCost + ppl_
          logger.debug(s"${$$(idsVocab).apply(path.last)} -> $cand, $ppl_, $cost")

          if (cost < minCost){
            minCost = cost
            minPath = path :+ state
            minWords = cands :+ cand
          }
        }
        newPaths = newPaths :+ minPath
        newWords = newWords :+ minWords
        newCosts = newCosts :+ minCost + wcost * getOrDefault(tradeoff)
      }
      pathsIds = newPaths
      pathWords = newWords
      costs = newCosts

      // log paths and costs
      pathWords.zip(costs).foreach{ case (path, cost) =>
        logger.debug(s"${path.toList}, $cost")
      }

    }
    // return the path with the lowest cost, and the cost
    val (minPath, minCost) = pathWords.zip(costs).minBy(_._2)

    if (minPath.nonEmpty)
      (minPath.tail.dropRight(1), minCost)
    else
      (minPath, minCost)
  }

  def getClassCandidates(transducer: ITransducer[Candidate], token:String, label:String, maxDist:Int, limit:Int = 2) = {
    import scala.collection.JavaConversions._
    transducer.transduce(token, maxDist).map {cand =>

      // if weights are available, we use them
      val weight = weights.get.
        map(ws => wLevenshteinDist(cand.term, token, ws)).
        getOrElse(cand.distance.toFloat)

      (cand.term, label, weight)
    }.toSeq.sortBy(_._3).take(limit)
  }

  def getVocabCandidates(token: String, maxDist:Int) = {
    import scala.collection.JavaConversions._
    val trans = $$(transducer)
    // we use all case information as it comes
    val plainCandidates = trans.transduce(token, maxDist).
      toList.map(c => (c.term, c.term, c.distance.toFloat))

    // We evaluate some case variations
    val tryUpperCase = getOrDefault(caseStrategy) == CandidateStrategy.ALL_UPPER_CASE ||
      getOrDefault(caseStrategy) == CandidateStrategy.ALL

    val tryFirstCapitalized = getOrDefault(caseStrategy) == CandidateStrategy.FIRST_LETTER_CAPITALIZED ||
      getOrDefault(caseStrategy) == CandidateStrategy.ALL

    val caseCandidates = if (token.isUpperCase && tryUpperCase) {
      trans.transduce(token.toLowerCase).toList.map(
        c => (c.term.toUpperCase, c.term, c.distance.toFloat)
      )
    } else if(token.isFirstLetterCapitalized && tryFirstCapitalized) {
      trans.transduce(token.toLowerCase).toList.map(
        c => (c.term.capitalizeFirstLetter, c.term, c.distance.toFloat)
      )
    } else Seq.empty

    plainCandidates ++ caseCandidates
  }

  implicit class StringTools(s: String) {
    def isUpperCase() = s.toUpperCase.equals(s)
    def isLowerCase() = s.toLowerCase.equals(s)
    def isFirstLetterCapitalized() =
      s.headOption.map{fl => fl.isUpper && s.tail.isLowerCase}.
        getOrElse(false)
    def capitalizeFirstLetter() = s.head.toUpper + s.tail
  }


  override def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    require(_model.isDefined, "Tensorflow model has not been initialized")
    dataset
  }

  /**
    * takes a document and annotations and produces new annotations of this annotator's annotation type
    *
    * @param annotations Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return any number of annotations processed for every input annotation. Not necessary one to one relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val decodedSentPaths = annotations.groupBy(_.metadata.getOrElse("sentence", "0")).mapValues{ sentTokens =>
      val (decodedPath, cost) = toOption(getOrDefault(useNewLines)).map { _ =>
        val idxs = Seq(-1) ++ sentTokens.zipWithIndex.filter { case (a, _) => a.result.equals(System.lineSeparator) || a.result.equals(System.lineSeparator*2) }.
          map(_._2) ++ Seq(annotations.length)
        idxs.zip(idxs.tail).map { case (s, e) =>
          decodeViterbi(computeTrellis(sentTokens.slice(s + 1, e), computeMask(sentTokens.slice(s + 1, e))))
        }.reduceLeft[(Array[String], Double)]({ case ((dPathA, pCostA), (dPathB, pCostB)) =>
          (dPathA ++ Seq(System.lineSeparator) ++ dPathB, pCostA + pCostB)
        })
      }.getOrElse(decodeViterbi(computeTrellis(sentTokens, computeMask(sentTokens))))
      //ToDo: This is a backup plan for empty DecodedPath -- fix me!!
      if (decodedPath.nonEmpty)
        sentTokens.zip(decodedPath).map{case (orig, correct) =>
          orig.copy(result = correct, metadata = orig.metadata.updated("cost", cost.toString))}
      else
        sentTokens.map(orig =>
          orig.copy(metadata = orig.metadata.updated("cost", "0"))
        )
    }

    decodedSentPaths.values.flatten.toSeq
  }

  def toOption(boolean:Boolean) = {
    if(boolean)
      Some(boolean)
    else
      None
  }

  /* detects which tokens need correction
   *
   * returns a mask with boolean flag for each word indicating whether it needs correction or not
   *
   * two causes for a word to need correction, 1. high perplexity or 2. out of vocabulary
   * */
  def computeMask(annotations:Seq[Annotation]): Array[Boolean] = {
    val threshold = getOrDefault(errorThreshold)
    val unkCode = $$(vocabIds).get("_UNK_").get

    /* try to decide whether words need correction or not */
    // first pass - perplexities
    val encodedSent = Array($$(vocabIds)("_BOS_"))  ++ annotations.map{ ann =>
      if ($(compareLowcase))
        $$(vocabIds).get(ann.result).getOrElse($$(vocabIds).get(ann.result.toLowerCase).getOrElse(unkCode))
      else
        $$(vocabIds).get(ann.result).getOrElse(unkCode)
    } ++ Array($$(vocabIds)("_EOS_"))

    val cids = encodedSent.map{id => $$(classes).apply(id)._1}
    val cwids = encodedSent.map{id => $$(classes).apply(id)._2}

    val perplexities = getModelIfNotSet.pplEachWord(Array(encodedSent), Array(cids), Array(cwids)).map(_ > threshold)

    perplexities.zip(perplexities.tail).zip(encodedSent.tail).
      // if the word to the right needs correction, this word needs it too and is word in vocabulary ?
      map {case ((needCorrection, nextNeedCorrection), code) =>
          if(nextNeedCorrection) true else needCorrection || code == unkCode
    }
  }

  def computeTrellis(annotations:Seq[Annotation], mask: Seq[Boolean]) = {
    annotations.zip(mask).map { case (annotation, needCorrection) =>
      val token = annotation.result
      var correctionCondition = needCorrection
      if (! $(correctSymbols))
        correctionCondition = needCorrection & token.replaceAll("[^A-Za-z0-9]+", "").length > 0

      if(correctionCondition) {
            // ask each token class for candidates, keep the one with lower cost
            var candLabelWeight = $$(specialTransducers).flatMap { specialParser =>
              if (specialParser.transducer == null)
                throw new RuntimeException(s"${specialParser.label}")
              getClassCandidates(specialParser.transducer, token, specialParser.label, getOrDefault(wordMaxDistance) - 1)
            } ++ getVocabCandidates(token, getOrDefault(wordMaxDistance) - 1)

            // now try to relax distance requirements for candidates
            if (token.length > 4 && candLabelWeight.isEmpty)
              candLabelWeight = $$(specialTransducers).flatMap { specialParser =>
                getClassCandidates(specialParser.transducer, token, specialParser.label, getOrDefault(wordMaxDistance))
              } ++ getVocabCandidates(token, getOrDefault(wordMaxDistance))

            if (candLabelWeight.isEmpty)
              candLabelWeight = Array((token, "_UNK_", 3.0f))

            // label is a dictionary word for the main transducer, or a label such as _NUM_ for special classes
            val labelWeightCand = candLabelWeight.map { case (term, label, dist) =>
              // optional re-ranking of candidates according to special distance
              val d = get(weights).map { w => wLevenshteinDist(term, token, w) }.getOrElse(dist)
              val weight = d - $$(vocabFreq).getOrElse(label, 0.0) / getOrDefault(gamma)
              (label, weight, term)
            }.sortBy(_._2).take(getOrDefault(maxCandidates))
            logger.debug(s"""$token -> ${labelWeightCand.toList.take(getOrDefault(maxCandidates))}""")
            labelWeightCand.toArray //[(String, Double, String)]
      } else {Array(("_UNK_", .2, token))}
 	}.toArray
  }

  def this() = this(Identifiable.randomUID("SPELL"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.TOKEN)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.TOKEN

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModel(path, spark, getModelIfNotSet.tensorflow, "_langmodeldl", ContextSpellCheckerModel.tfFile, configProtoBytes = getConfigProtoBytes)
  }
}


trait ReadsLanguageModelGraph extends ParamsAndFeaturesReadable[ContextSpellCheckerModel] with ReadTensorflowModel {

  override val tfFile = "tensorflow_lm"

  def readLanguageModelGraph(instance: ContextSpellCheckerModel, path: String, spark: SparkSession): Unit = {
    val tf = readTensorflowModel(path, spark, "_langmodeldl")
    instance.setModelIfNotSet(spark, tf)
  }

  addReader(readLanguageModelGraph)
}

trait ReadablePretrainedContextSpell extends ReadsLanguageModelGraph with HasPretrained[ContextSpellCheckerModel] {
  override val defaultModelName: Some[String] = Some("spellcheck_dl")
  /** Java compliant-overrides */
  override def pretrained(): ContextSpellCheckerModel = super.pretrained()
  override def pretrained(name: String): ContextSpellCheckerModel = super.pretrained(name)
  override def pretrained(name: String, lang: String): ContextSpellCheckerModel = super.pretrained(name, lang)
  override def pretrained(name: String, lang: String, remoteLoc: String): ContextSpellCheckerModel = super.pretrained(name, lang, remoteLoc)
}

object ContextSpellCheckerModel extends ReadablePretrainedContextSpell
