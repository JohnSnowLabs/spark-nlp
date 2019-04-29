package com.johnsnowlabs.nlp.annotators.spell.context

import com.github.liblevenshtein.transducer.{Candidate, ITransducer}
import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.serialization._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.spell.context.parser.SpecialClassParser
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, FloatParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, SparkSession}
import org.slf4j.LoggerFactory


class ContextSpellCheckerModel(override val uid: String) extends AnnotatorModel[ContextSpellCheckerModel]
  with ReadTensorflowModel
  with WeightedLevenshtein
  with WriteTensorflowModel
  with ParamsAndFeaturesWritable
  with HasTransducerFeatures {

  private val logger = LoggerFactory.getLogger("ContextSpellCheckerModel")

  override val tfFile: String = "bigone"

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

  val tradeoff = new FloatParam(this, "tradeoff", "Tradeoff between the cost of a word and a transition in the language model.")
  def setTradeOff(lambda: Float):this.type = set(tradeoff, lambda)

  val gamma = new FloatParam(this, "gamma", "Controls the influence of individual word frequency in the decision.")
  def setGamma(g: Float):this.type = set(tradeoff, g)

  val weights: MapFeature[String, Map[String, Float]] = new MapFeature[String, Map[String, Float]](this, "levenshteinWeights")
  def setWeights(w:Map[String, Map[String, Float]]): this.type = set(weights, w)

  val useNewLines = new BooleanParam(this, "trim", "When set to true new lines will be treated as any other character, when set to false" +
    " correction is applied on paragraphs as defined by newline characters.")
  def setUseNewLines(useIt: Boolean):this.type = set(useNewLines, useIt)

  val maxWindowLen = new IntParam(this, "maxWindowLen", "Maximum size for the window used to remember history prior to every correction.")
  def setMaxWindowLen(w: Int):this.type = set(maxWindowLen, w)

  setDefault(tradeoff -> 18.0f, gamma -> 120.0f, useNewLines -> false, maxCandidates -> 6, maxWindowLen -> 5)


  // the scores for the EOS (end of sentence), and BOS (beginning of sentence)
  private val eosScore = .01
  private val bosScore = 1.0


  /* reads the external TF model, keeping this until we can train from within spark */
  def readModel(path: String, spark: SparkSession, suffix: String, useBundle:Boolean): this.type = {
    val tf = readTensorflowModel(
      path,
      spark,
      suffix,
      zipped=false,
      useBundle,
      tags = Array("our-graph")
    )
    _model = None
    setModelIfNotSet(spark, tf)
  }

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

    for(i <- 1 until encTrellis.length) {

      var newPaths:Array[Array[Int]] = Array()
      var newWords: Array[Array[String]] = Array()
      var newCosts = Array[Double]()

      /* compute all the costs for all transitions in current step - use a batch */
      val expPaths = encTrellis(i).flatMap{ case (state, _, _) =>
        pathsIds.map { path =>
          path :+ state
        }
      }.map(_.takeRight($(maxWindowLen)))

      val cids = expPaths.map(_.map{id => $$(classes).apply(id)._1})
      val cwids = expPaths.map(_.map{id => $$(classes).apply(id)._2})
      val expPathsCosts = getModelIfNotSet.predict(expPaths, cids, cwids).toArray

      for {((state, wcost, cand), idx) <- encTrellis(i).zipWithIndex} {
        var minCost = Double.MaxValue
        var minPath = Array[Int]()
        var minWords = Array[String]()

        val z = (pathsIds, costs, pathWords).zipped.toList

        for (((path, pathCost, cands), pi) <- z.zipWithIndex) {
          // compute cost to arrive to this 'state' coming from that 'path'
          val mult = if (i > 1) costs.length else 0
          val ppl = expPathsCosts(idx * mult + pi)

          logger.debug(s"${$$(idsVocab).apply(path.last)} -> $cand, $ppl")

          val cost = pathCost + ppl
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

    (minPath.tail.dropRight(1), minCost)
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

  def getVocabCandidates(trans: ITransducer[Candidate], token: String, maxDist:Int) = {
    import scala.collection.JavaConversions._

    if(token.head.isUpper && token.tail.forall(_.isLower)) {
      trans.transduce(token.head.toLower + token.tail, maxDist).
        toList.map(c => (c.term.head.toUpper +  c.term.tail, c.term, c.distance.toFloat))
    }
    else
      trans.transduce(token, maxDist).
        toList.map(c => (c.term, c.term, c.distance.toFloat))
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
    // TODO still don't like the .apply() here
    val decodedSentPaths = annotations.groupBy(_.metadata.apply("sentence")).mapValues{ sentTokens =>
      val (decodedPath, cost) = toOption(getOrDefault(useNewLines)).map { _ =>
        val idxs = Seq(-1) ++ sentTokens.zipWithIndex.filter { case (a, _) => a.result.equals(System.lineSeparator) || a.result.equals(System.lineSeparator*2) }.
          map(_._2) ++ Seq(annotations.length)
        idxs.zip(idxs.tail).map { case (s, e) =>
          decodeViterbi(computeTrellis(sentTokens.slice(s + 1, e)))
        }.reduceLeft[(Array[String], Double)]({ case ((dPathA, pCostA), (dPathB, pCostB)) =>
          (dPathA ++ Seq(System.lineSeparator) ++ dPathB, pCostA + pCostB)
        })
      }.getOrElse(decodeViterbi(computeTrellis(sentTokens)))
      sentTokens.zip(decodedPath).map{case (orig, correct) => orig.copy(result = correct)}
    }

    decodedSentPaths.values.flatten.toSeq
  }

  def toOption(boolean:Boolean) = {
    if(boolean)
      Some(boolean)
    else
      None
  }


  def computeTrellis(annotations:Seq[Annotation]) = {

    annotations.map { annotation =>
      val token = annotation.result


      // ask each token class for candidates, keep the one with lower cost
      var candLabelWeight = $$(specialTransducers).flatMap { specialParser =>
        if(specialParser.transducer == null)
          throw new RuntimeException(s"${specialParser.label}")
        // println(s"special parser:::${specialParser.label}")
        // println(s"value: ${specialParser.transducer}")
        getClassCandidates(specialParser.transducer, token, specialParser.label, getOrDefault(wordMaxDistance) - 1)
      } ++ getVocabCandidates($$(transducer), token, getOrDefault(wordMaxDistance) -1)

      // now try to relax distance requirements for candidates
      if (token.length > 4 && candLabelWeight.isEmpty)
        candLabelWeight = $$(specialTransducers).flatMap { specialParser =>
          getClassCandidates(specialParser.transducer, token, specialParser.label, getOrDefault(wordMaxDistance))
        } ++ getVocabCandidates($$(transducer), token, getOrDefault(wordMaxDistance))

      if (candLabelWeight.isEmpty)
        candLabelWeight = Array((token, "_UNK_", 3.0f))

      // label is a dictionary word for the main transducer, or a label such as _NUM_ for special classes
      val labelWeightCand = candLabelWeight.map{ case (term, label, dist) =>
        // optional re-ranking of candidates according to special distance
        val d = get(weights).map{w => wLevenshteinDist(term, token, w)}.getOrElse(dist)
        val weight =  d - $$(vocabFreq).getOrElse(label, 0.0) / getOrDefault(gamma)
        (label, weight, term)
      }.sortBy(_._2).take(getOrDefault(maxCandidates))

      logger.debug(s"""$token -> ${labelWeightCand.toList.take(getOrDefault(maxCandidates))}""")
      labelWeightCand.toArray
    }.toArray

  }

  def this() = this(Identifiable.randomUID("SPELL"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.TOKEN)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.TOKEN

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModel(path, spark, getModelIfNotSet.tensorflow, "_langmodeldl", ContextSpellCheckerModel.tfFile)
  }
}


trait ReadsLanguageModelGraph extends ParamsAndFeaturesReadable[ContextSpellCheckerModel] with ReadTensorflowModel {

  override val tfFile = "bigone"

  def readLanguageModelGraph(instance: ContextSpellCheckerModel, path: String, spark: SparkSession): Unit = {
    val tf = readTensorflowModel(path, spark, "_langmodeldl")
    instance.setModelIfNotSet(spark, tf)
  }

  addReader(readLanguageModelGraph)
}

trait PretrainedSpellModel {
  def pretrained(name: String = "spellcheck_dl", language: Option[String] = Some("en"), remoteLoc: String = ResourceDownloader.publicLoc): ContextSpellCheckerModel =
    ResourceDownloader.downloadModel(ContextSpellCheckerModel, name, language, remoteLoc)
}

object ContextSpellCheckerModel extends ReadsLanguageModelGraph with PretrainedSpellModel