package com.johnsnowlabs.nlp.annotators.spell.ocr

import com.github.liblevenshtein.transducer.{Candidate, ITransducer}
import com.johnsnowlabs.ml.tensorflow.{TensorflowSpell, TensorflowWrapper}
import com.johnsnowlabs.nlp.annotators.assertion.dl.{ReadTensorflowModel, WriteTensorflowModel}
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.serialization.{MapFeature, StructFeature, TransducerFeature}
import com.johnsnowlabs.nlp._
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

class OcrSpellCheckModel(override val uid: String) extends AnnotatorModel[OcrSpellCheckModel]
  with ReadTensorflowModel // --> remove
  with TokenClasses
  with WriteTensorflowModel
  with ParamsAndFeaturesWritable {

  override val tfFile: String = "bigone"

  private var tensorflow:TensorflowWrapper = null

  val transducer = new TransducerFeature(this, "The transducer for the main vocabulary.")
  def setVocabTransducer(trans:ITransducer[Candidate]) = {
    transducer.setValue(Some(trans))
    this
  }

  val specialTransducers = new StructFeature[Seq[(ITransducer[Candidate], String)]](this, "The transducers for special classes.")
  def setSpecialClassesTransducers(transducers: Seq[(ITransducer[Candidate], String)]) = set(specialTransducers, transducers)

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

  val maxCandidates = new Param[Int](this, "maxCandidates", "Maximum number of candidates for every word.")
  val tradeoff = new Param[Float](this, "tradeoff", "Tradeoff between the cost of a word and a transition in the language model.")


  // the scores for the EOS (end of sentence), and BOS (begining of sentence)
  private val eosScore = .01
  private val bosScore = 1.0

  private val gamma = 120.0

  def readModel(path: String, spark: SparkSession, suffix: String): this.type = {
    tensorflow = readTensorflowModel(path, spark, suffix, false)
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


  def setTensorflow(tf: TensorflowWrapper): OcrSpellCheckModel = {
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

          println(s"${$$(idsVocab).get(path.last).get} -> $cand, $ppl")

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

      /* TODO: create an optional log with this */
      pathWords.zip(costs).foreach{ case (path, cost) =>
          println(path.toList, cost)
      }
      println("----------")
    }
    // return the path with the lowest cost, and the cost
    pathWords.zip(costs).minBy(_._2)
  }

  def getClassCandidates(transducer: ITransducer[Candidate], token:String , k: Int , label:String) = {
    import scala.collection.JavaConversions._
    transducer.transduce(token, 2).map {case cand =>
      val weight = wLevenshteinDist(cand.term, token)
      (cand.term, label, weight)
    }.toSeq.sortBy(_._3).take(k)
  }

  def getVocabCandidates(trans: ITransducer[Candidate], token: String, maxDist:Int = 2) = {
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
    annotations.map{ annotation =>
      val trellis:Array[Array[(String, Double, String)]] = annotation.result.split(" ").map { token =>

        // ask each token class for candidates, keep the one with lower cost
        var candLabelWeight = $$(specialTransducers).flatMap { case (transducer, label) =>
            getClassCandidates(transducer, token, k = 2, label)
        } ++ getVocabCandidates(transducer.getOrDefault, token)

        // quick and dirty patch
        if (token.length > 4 && candLabelWeight.isEmpty)
          candLabelWeight = getVocabCandidates(transducer.getOrDefault, token, 3)

        // quick and dirty patch
        if (candLabelWeight.isEmpty)
          candLabelWeight = Seq((token, "_UNK_", 3.0f))

        //TODO: there's no reason to carry the distance until this point
        // optional re-ranking of candidates according to special distance
        val candLabelDist = candLabelWeight.map {case (cand, label, _) =>
          val weight = wLevenshteinDist(cand, token)
          (cand, label, weight)
        }

        //label is a dictionary word for the main transducer, or a label such as _NUM_ for special classes
        val labelWeightCand = candLabelDist.map{ case (term, label, dist) =>

          val weight =  - $$(vocabFreq).getOrElse(label, 0.0) / gamma + dist.toDouble
          (label, weight, term)
        }.sortBy(_._2).take(getOrDefault(maxCandidates))

        println(s"""$token -> ${labelWeightCand.toList.take(getOrDefault(maxCandidates))}""")
        labelWeightCand.toArray
      }

      val (decodedPath, cost) = decodeViterbi(trellis)

      annotation.copy(result = decodedPath.tail.dropRight(1) // get rid of BOS and EOS
        .mkString(" "),
        metadata = annotation.metadata + ("score" -> cost.toString))
    }
  }

  def this() = this(Identifiable.randomUID("SPELL"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val requiredAnnotatorTypes: Array[String] = Array(AnnotatorType.TOKEN)
  override val annotatorType: AnnotatorType = AnnotatorType.TOKEN

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModel(path, spark, tensorflow, "_langmodeldl")
  }

}


trait ReadsLanguageModelGraph extends ParamsAndFeaturesReadable[OcrSpellCheckModel] with ReadTensorflowModel {

  override val tfFile = "bigone"

  def readLanguageModelGraph(instance: OcrSpellCheckModel, path: String, spark: SparkSession): Unit = {
    val tf = readTensorflowModel(path, spark, "_langmodeldl")
    instance.setTensorflow(tf)
  }

  addReader(readLanguageModelGraph)
}

object OcrSpellCheckModel extends ReadsLanguageModelGraph