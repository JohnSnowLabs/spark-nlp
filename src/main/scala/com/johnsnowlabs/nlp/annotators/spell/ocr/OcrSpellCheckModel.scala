package com.johnsnowlabs.nlp.annotators.spell.ocr

import com.github.liblevenshtein.transducer.{Candidate, ITransducer}
import com.johnsnowlabs.ml.tensorflow.{TensorflowSpell, TensorflowWrapper}
import com.johnsnowlabs.nlp.annotators.assertion.dl.ReadTensorflowModel
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.serialization.{MapFeature, StructFeature}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

class OcrSpellCheckModel(override val uid: String) extends AnnotatorModel[OcrSpellCheckModel] with ReadTensorflowModel {

  override val tfFile: String = "good_model"

  private var tensorflow:TensorflowWrapper = null

  val transducer = new StructFeature[ITransducer[Candidate]](this, "The transducer for the main vocabulary.")
  def setVocabTransducer(trans:ITransducer[Candidate]) = set(transducer, trans)

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

  // the score for the EOS (end of sentence), and BOS (begining of sentence)
  private val eosScore = .01
  private val bosScore = 1.0
  private val gamma = 60.0

  def readModel(path: String, spark: SparkSession, suffix: String): this.type = {
    tensorflow = readTensorflowModel(path, spark, suffix)
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
          trellis.map(_.map{case (word, weight, cand) =>
            // at this point we keep only those candidates that are in the vocabulary
            ($$(vocabIds).get(word), weight, cand)}.filter(_._1.isDefined).map{case (x,y,z) => (x.get, y, z)}) ++
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
          // val ppl = model.predict(Array(path :+ state))
          val mult = if (i > 1) costs.length else 0
          val ppl = expPathsCosts(idx * mult + pi)

          val cost = pathCost + ppl * wcost
          if (cost < minCost){
            minCost = cost
            minPath = path :+ state
            minWords = cands :+ cand
          }
        }
        newPaths = newPaths :+ minPath
        newWords = newWords :+ minWords
        newCosts = newCosts :+ minCost
      }
      pathsIds = newPaths
      pathWords = newWords
      costs = newCosts

      /* TODO: create an optional log with this */
      pathsIds.zip(costs).foreach{ case (path, cost) =>
          println(path.map($$(idsVocab).get).map(_.get).toList, cost)
      }
      println("----------")
    }
    // return the path with the lowest cost, and the cost
    pathWords.zip(costs).minBy(_._2)
  }

  /**
    * takes a document and annotations and produces new annotations of this annotator's annotation type
    *
    * @param annotations Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return any number of annotations processed for every input annotation. Not necessary one to one relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    //val allTransducers = $$(specialTransducers) :+ $$(transducer)
    import scala.collection.JavaConversions._
    annotations.map{ annotation =>
      val trellis:Array[Array[(String, Double, String)]] = annotation.result.split(" ").map { token =>

        // ask each token class for candidates, keep the one with lower cost
        val candLabel = $$(specialTransducers).flatMap { case (transducer, label) =>
          // TODO: hardcoded maxDistance!
          transducer.transduce(token, 2).map((_, label))
        } ++ $$(transducer).transduce(token, 2).
          map { case cand => (cand, cand.term)}

        //label is a dictionary word for the main transducer, or a label such as _NUM_ for special classes
        val labelWeightCand = candLabel.map{ case (c, l) =>
          val weight =  - $$(vocabFreq).getOrElse(c.term, 0.0) / gamma +
                          c.distance.toDouble / token.size
          (l, weight, c.term)
        }.sortBy(_._2).take(getOrDefault(maxCandidates))

        println(s"""$token -> ${labelWeightCand.toList.take(getOrDefault(maxCandidates))}""")
        labelWeightCand.toArray
      }

      val (decodedPath, cost) = decodeViterbi(trellis)

      annotation.copy(result = decodedPath.mkString(" "),
        metadata = annotation.metadata + ("score" -> cost.toString))
    }
  }

  def this() = this(Identifiable.randomUID("SPELL"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val requiredAnnotatorTypes: Array[String] = Array(AnnotatorType.TOKEN)
  override val annotatorType: AnnotatorType = AnnotatorType.TOKEN

}
