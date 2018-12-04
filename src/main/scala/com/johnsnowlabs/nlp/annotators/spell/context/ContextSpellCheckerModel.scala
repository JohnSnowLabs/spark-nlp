package com.johnsnowlabs.nlp.annotators.spell.context

import com.github.liblevenshtein.transducer.{Candidate, ITransducer}
import com.johnsnowlabs.ml.tensorflow.{ReadTensorflowModel, TensorflowSpell, TensorflowWrapper, WriteTensorflowModel}
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.serialization._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.spell.context.parser.SpecialClassParser
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import org.apache.spark.ml.param.{FloatParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory


class ContextSpellCheckerModel(override val uid: String) extends AnnotatorModel[ContextSpellCheckerModel]
  with ReadTensorflowModel
  with WeightedLevenshtein
  with WriteTensorflowModel
  with ParamsAndFeaturesWritable {

  private val logger = LoggerFactory.getLogger("ContextSpellCheckerModel")

  override val tfFile: String = "bigone"

  private var tensorflow:TensorflowWrapper = null

  val transducer = new TransducerFeature(this, "The transducer for the main vocabulary.")
  def setVocabTransducer(trans:ITransducer[Candidate]) = {
    transducer.setValue(Some(trans))
    this
  }

  val specialTransducers = new TransducerSeqFeature(this, "The transducers for special classes.")
  def setSpecialClassesTransducers(transducers: Seq[SpecialClassParser]) = {
    specialTransducers.setValue(Some(transducers))
    this
  }

  val vocabFreq  = new MapFeature[String, Double](this, "vocabFreq")
  def setVocabFreq(v: Map[String, Double]) = set(vocabFreq,v)

  val idsVocab = new MapFeature[Int, String](this, "idsVocab")
  val vocabIds = new MapFeature[String, Int](this, "vocabIds")

  def setVocabIds(v: Map[String, Int]) = {
    set(idsVocab, v.map(_.swap))
    set(vocabIds, v)
  }

  val classes: MapFeature[Int, (Int, Int)] = new MapFeature(this, "classes")
  def setClasses(c:Map[Int, (Int, Int)]) = set(classes, c)

  val wordMaxDistance = new IntParam(this, "wordMaxDistance", "Maximum distance for the generated candidates for every word, minimum 1.")
  def setWordMaxDist(k: Int):this.type = set(wordMaxDistance, k)

  val maxCandidates = new IntParam(this, "maxCandidates", "Maximum number of candidates for every word.")

  val tradeoff = new FloatParam(this, "tradeoff", "Tradeoff between the cost of a word and a transition in the language model.")
  def setTradeOfft(lambda: Float):this.type = set(tradeoff, lambda)

  val gamma = new FloatParam(this, "gamma", "Controls the influence of individual word frequency in the decision.")

  val weights: MapFeature[Char, Map[Char, Float]] = new MapFeature[Char, Map[Char, Float]](this, "levenshteinWeights")
  def setWeights(w:Map[Char, Map[Char, Float]]): this.type = set(weights, w)

  setDefault(tradeoff -> 18.0f, gamma -> 120.0f)

  // the scores for the EOS (end of sentence), and BOS (beginning of sentence)
  private val eosScore = .01
  private val bosScore = 1.0


  /* reads the external TF model, keeping this until we can train from within spark */
  def readModel(path: String, spark: SparkSession, suffix: String, useBundle:Boolean): this.type = {
    tensorflow = readTensorflowModel(path, spark, suffix, false, useBundle, tags = Array("our-graph"))
    this
  }

  @transient
  private var _model: TensorflowSpell = null

  def model: TensorflowSpell = {
    if (_model == null) {
      require(tensorflow != null, "Tensorflow must be set before usage. Use method setTensorflow() for it.")

      _model = new TensorflowSpell(
        tensorflow,
        Verbose.Silent)
    }
    _model
  }


  def setTensorflow(tf: TensorflowWrapper): ContextSpellCheckerModel = {
    tensorflow = tf
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

    for(i <- 1 to encTrellis.length - 1) {

      var newPaths:Array[Array[Int]] = Array()
      var newWords: Array[Array[String]] = Array()
      var newCosts = Array[Double]()

      /* compute all the costs for all transitions in current step - use a batch */
      val expPaths = encTrellis(i).flatMap{ case (state, _, _) =>
        pathsIds.map { path =>
          path :+ state
        }
      }

      val cids = expPaths.map(_.map{id => $$(classes).get(id).get._1})
      val cwids = expPaths.map(_.map{id => $$(classes).get(id).get._2})
      val expPathsCosts = model.predict(expPaths, cids, cwids).toArray

      for {((state, wcost, cand), idx) <- encTrellis(i).zipWithIndex} {
        var minCost = Double.MaxValue
        var minPath = Array[Int]()
        var minWords = Array[String]()

        val z = (pathsIds, costs, pathWords).zipped.toList

        for (((path, pathCost, cands), pi) <- z.zipWithIndex) {
          // compute cost to arrive to this 'state' coming from that 'path'
          val mult = if (i > 1) costs.length else 0
          val ppl = expPathsCosts(idx * mult + pi)

          logger.debug(s"${$$(idsVocab).get(path.last).get} -> $cand, $ppl")

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
        logger.debug(s"${path.toList}, $cost\n-----\n")
      }

    }
    // return the path with the lowest cost, and the cost
    pathWords.zip(costs).minBy(_._2)
  }

  def getClassCandidates(transducer: ITransducer[Candidate], token:String, label:String, maxDist:Int) = {
    import scala.collection.JavaConversions._
    transducer.transduce(token, maxDist).map {case cand =>

      // if weights are available, we use them
      val weight = weights.get.
        map(ws => wLevenshteinDist(cand.term, token, ws)).
        getOrElse(cand.distance.toFloat)

      (cand.term, label, weight)
    }.toSeq.sortBy(_._3).take(2) //getOrDefault(maxCandidates)
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

  /**
    * takes a document and annotations and produces new annotations of this annotator's annotation type
    *
    * @param annotations Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return any number of annotations processed for every input annotation. Not necessary one to one relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    import scala.collection.JavaConversions._

      val trellis:Array[Array[(String, Double, String)]] = annotations.map { annotation =>
        val token = annotation.result


        // ask each token class for candidates, keep the one with lower cost
        var candLabelWeight = specialTransducers.getOrDefault.flatMap { case specialParser =>
            getClassCandidates(specialParser.transducer, token, specialParser.label, getOrDefault(wordMaxDistance) - 1)
        } ++ getVocabCandidates(transducer.getOrDefault, token, getOrDefault(wordMaxDistance) -1)

        // now try to relax distance requirements for candidates
        if (token.length > 4 && candLabelWeight.isEmpty)
          candLabelWeight = specialTransducers.getOrDefault.flatMap { case specialParser =>
            getClassCandidates(specialParser.transducer, token, specialParser.label, getOrDefault(wordMaxDistance))
          } ++ getVocabCandidates(transducer.getOrDefault, token, getOrDefault(wordMaxDistance))

        if (candLabelWeight.isEmpty)
          candLabelWeight = Seq((token, "_UNK_", 3.0f))

        // label is a dictionary word for the main transducer, or a label such as _NUM_ for special classes
        val labelWeightCand = candLabelWeight.map{ case (term, label, dist) =>
          // optional re-ranking of candidates according to special distance
          val d = get(weights).map{w => wLevenshteinDist(term, token, w)}.getOrElse(dist)
          val weight =  d - $$(vocabFreq).getOrElse(label, 0.0) / getOrDefault(gamma)
            (label, weight, term)
        }.sortBy(_._2).take(getOrDefault(maxCandidates))

        logger.info(s"""$token -> ${labelWeightCand.toList.take(getOrDefault(maxCandidates))}""")
        labelWeightCand.toArray
      }.toArray

      val (decodedPath, cost) = decodeViterbi(trellis)

    decodedPath.tail.dropRight(1). // get rid of BOS and EOS
    map { word => Annotation(word)}

  }

  def this() = this(Identifiable.randomUID("SPELL"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val requiredAnnotatorTypes: Array[String] = Array(AnnotatorType.TOKEN)
  override val annotatorType: AnnotatorType = AnnotatorType.TOKEN

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModel(path, spark, tensorflow, "_langmodeldl", ContextSpellCheckerModel.tfFile)
  }
}


trait ReadsLanguageModelGraph extends ParamsAndFeaturesReadable[ContextSpellCheckerModel] with ReadTensorflowModel {

  override val tfFile = "bigone"

  def readLanguageModelGraph(instance: ContextSpellCheckerModel, path: String, spark: SparkSession): Unit = {
    val tf = readTensorflowModel(path, spark, "_langmodeldl")
    instance.setTensorflow(tf)
  }

  addReader(readLanguageModelGraph)
}

trait PretrainedSpellModel {
  def pretrained(name: String = "context_spell_gen", language: Option[String] = Some("en"), remoteLoc: String = ResourceDownloader.publicLoc): ContextSpellCheckerModel =
    ResourceDownloader.downloadModel(ContextSpellCheckerModel, name, language, remoteLoc)
}

object ContextSpellCheckerModel extends ReadsLanguageModelGraph with PretrainedSpellModel