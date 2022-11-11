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

import com.github.liblevenshtein.transducer.{Candidate, ITransducer}
import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.annotators.spell.context.parser._
import com.johnsnowlabs.nlp.serialization._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, FloatParam, IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, SparkSession}
import org.slf4j.LoggerFactory

import java.util
import scala.collection.JavaConverters._
import scala.collection.mutable

/** Implements a deep-learning based Noisy Channel Model Spell Algorithm. Correction candidates
  * are extracted combining context information and word information.
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
  * This is the instantiated model of the [[ContextSpellCheckerApproach]]. For training your own
  * model, please see the documentation of that class.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val spellChecker = ContextSpellCheckerModel.pretrained()
  *   .setInputCols("token")
  *   .setOutputCol("checked")
  * }}}
  * The default model is `"spellcheck_dl"`, if no name is provided. For available pretrained
  * models please see the [[https://nlp.johnsnowlabs.com/models?task=Spell+Check Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SPELL_CHECKER_EN.ipynb Spark NLP Workshop]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/spell/context/ContextSpellCheckerTestSpec.scala ContextSpellCheckerTestSpec]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellCheckerModel
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("doc")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols(Array("doc"))
  *   .setOutputCol("token")
  *
  * val spellChecker = ContextSpellCheckerModel
  *   .pretrained()
  *   .setTradeOff(12.0f)
  *   .setInputCols("token")
  *   .setOutputCol("checked")
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   tokenizer,
  *   spellChecker
  * ))
  *
  * val data = Seq("It was a cold , dreary day and the country was white with smow .").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.select("checked.result").show(false)
  * +--------------------------------------------------------------------------------+
  * |result                                                                          |
  * +--------------------------------------------------------------------------------+
  * |[It, was, a, cold, ,, dreary, day, and, the, country, was, white, with, snow, .]|
  * +--------------------------------------------------------------------------------+
  * }}}
  *
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingModel NorvigSweetingModel]] and
  *   [[com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteModel SymmetricDeleteModel]]
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
class ContextSpellCheckerModel(override val uid: String)
    extends AnnotatorModel[ContextSpellCheckerModel]
    with HasSimpleAnnotate[ContextSpellCheckerModel]
    with WeightedLevenshtein
    with WriteTensorflowModel
    with ParamsAndFeaturesWritable
    with HasTransducerFeatures
    with HasEngine {

  private val logger = LoggerFactory.getLogger("ContextSpellCheckerModel")

  val transducer = new TransducerFeature(this, "mainVocabularyTransducer")

  /** @group setParam */
  def setVocabTransducer(trans: ITransducer[Candidate]): this.type = {
    val main = new MainVocab()
    main.transducer = trans

    set(transducer, main)
  }

  val specialTransducers = new TransducerSeqFeature(this, "specialClassesTransducers")

  /** @group setParam */
  def setSpecialClassesTransducers(transducers: Seq[SpecialClassParser]): this.type = {
    set(specialTransducers, transducers.toArray)
  }

  /** Frequency words from the vocabulary
    *
    * @group param
    */
  val vocabFreq = new MapFeature[String, Double](this, "vocabFreq")

  /** @group setParam */
  def setVocabFreq(v: Map[String, Double]): this.type = set(vocabFreq, v)

  /** Mapping of ids to vocabulary
    *
    * @group param
    */
  val idsVocab = new MapFeature[Int, String](this, "idsVocab")

  /** Mapping of vocabulary to ids
    *
    * @group param
    */
  val vocabIds = new MapFeature[String, Int](this, "vocabIds")

  /** @group setParam */
  def setVocabIds(v: Map[String, Int]): this.type = {
    set(idsVocab, v.map(_.swap))
    set(vocabIds, v)
  }

  /** Classes the spell checker recognizes
    *
    * @group param
    */
  val classes: MapFeature[Int, (Int, Int)] = new MapFeature(this, "classes")

  /** @group setParam */
  def setClasses(c: Map[Int, (Int, Int)]): this.type = set(classes, c)

  /** Maximum distance for the generated candidates for every word, minimum 1.
    *
    * @group param
    */
  val wordMaxDistance = new IntParam(
    this,
    "wordMaxDistance",
    "Maximum distance for the generated candidates for every word, minimum 1.")

  /** @group setParam */
  def setWordMaxDistance(k: Int): this.type = set(wordMaxDistance, k)

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

  /** Threshold perplexity for a word to be considered as an error.
    *
    * @group param
    */
  val errorThreshold = new FloatParam(
    this,
    "errorThreshold",
    "Threshold perplexity for a word to be considered as an error.")

  /** @group setParam */
  def setErrorThreshold(t: Float): this.type = set(errorThreshold, t)

  /** Tradeoff between the cost of a word and a transition in the language model (Default:
    * `18.0f`).
    *
    * @group param
    */
  val tradeoff = new FloatParam(
    this,
    "tradeoff",
    "Tradeoff between the cost of a word and a transition in the language model.")

  /** @group setParam */
  def setTradeOff(lambda: Float): this.type = set(tradeoff, lambda)

  /** Controls the influence of individual word frequency in the decision (Default: `120.0f`).
    *
    * @group param
    */
  val gamma = new FloatParam(
    this,
    "gamma",
    "Controls the influence of individual word frequency in the decision.")

  /** @group setParam */
  def setGamma(g: Float): this.type = set(gamma, g)

  val weights: MapFeature[String, Map[String, Float]] =
    new MapFeature[String, Map[String, Float]](this, "levenshteinWeights")

  /** @group setParam */
  def setWeights(w: Map[String, Map[String, Float]]): this.type = set(weights, w)

  // for Python access

  /** @group setParam */
  def setWeights(w: util.HashMap[String, util.HashMap[String, Double]]): this.type = {

    val ws = w.asScala.mapValues(_.asScala.mapValues(_.toFloat).toMap).toMap
    set(weights, ws)
  }

  /** When set to true new lines will be treated as any other character (Default: `false`). When
    * set to false correction is applied on paragraphs as defined by newline characters.
    *
    * @group param
    */
  val useNewLines = new BooleanParam(
    this,
    "trim",
    "When set to true new lines will be treated as any other character, when set to false" +
      " correction is applied on paragraphs as defined by newline characters.")

  /** @group setParam */
  def setUseNewLines(useIt: Boolean): this.type = set(useNewLines, useIt)

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

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group param
    */
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): ContextSpellCheckerModel.this.type =
    set(this.configProtoBytes, bytes)

  /** @group getParam */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** Whether to correct special symbols or skip spell checking for them
    *
    * @group param
    */
  val correctSymbols: BooleanParam = new BooleanParam(
    this,
    "correctSymbols",
    "Whether to correct special symbols or skip spell checking for them")

  /** @group setParam */
  def setCorrectSymbols(value: Boolean): this.type = set(correctSymbols, value)

  setDefault(correctSymbols -> false)

  /** If true will compare tokens in low case with vocabulary (Default: `false`)
    *
    * @group param
    */
  val compareLowcase: BooleanParam = new BooleanParam(
    this,
    "compareLowcase",
    "If true will compare tokens in low case with vocabulary")

  /** @group setParam */
  def setCompareLowcase(value: Boolean): this.type = set(compareLowcase, value)

  setDefault(compareLowcase -> false)

  /** @group getParam */
  def getWordClasses: Seq[(String, AnnotatorType)] = $$(specialTransducers).map {
    case transducer: RegexParser =>
      (transducer.label, "RegexParser")
    case transducer: VocabParser =>
      (transducer.label, "VocabParser")
  }

  /* update a regex class */
  def updateRegexClass(label: String, regex: String): ContextSpellCheckerModel = {
    val classes = $$(specialTransducers)
    require(
      classes.count(_.label == label) == 1,
      s"Not found regex class $label. You can only update existing classes.")

    classes.filter(_.label.equals(label)).head match {
      case r: RegexParser =>
        r.regex = regex
        r.transducer = r.generateTransducer
      case _ => require(requirement = false, s"Class $label is not a regex class.")
    }
    this
  }

  /* update a vocabulary class */
  def updateVocabClass(
      label: String,
      vocabList: util.ArrayList[String],
      append: Boolean = true): ContextSpellCheckerModel = {
    val vocab = scala.collection.mutable.Set(vocabList.toArray.map(_.toString): _*)
    val classes = $$(specialTransducers)
    require(
      classes.count(_.label == label) == 1,
      s"Not found vocab class $label. You can only update existing classes.")

    classes.filter(_.label.equals(label)).head match {
      case v: VocabParser =>
        if (v.vocab.eq(null)) v.vocab = mutable.Set.empty[String]

        val newSet = if (append) v.vocab ++ vocab else vocab
        v.vocab = newSet
        v.transducer = v.generateTransducer
      case _ => require(requirement = false, s"Class $label is not a vocabulary class.")
    }
    this
  }

  setDefault(
    tradeoff -> 18.0f,
    gamma -> 120.0f,
    useNewLines -> false,
    maxCandidates -> 6,
    maxWindowLen -> 5,
    caseStrategy -> CandidateStrategy.ALL)

  // the scores for the EOS (end of sentence), and BOS (beginning of sentence)
  private val eosScore = .01
  private val bosScore = 1.0

  private var _model: Option[Broadcast[TensorflowSpell]] = None

  def getModelIfNotSet: TensorflowSpell = _model.get.value

  def setModelIfNotSet(spark: SparkSession, tensorflow: TensorflowWrapper): this.type = {
    if (_model.isEmpty) {
      _model = Some(spark.sparkContext.broadcast(new TensorflowSpell(tensorflow, Verbose.Silent)))
    }
    this
  }

  /* trellis goes like (label, weight, candidate)*/
  def decodeViterbi(trellis: Array[Array[(String, Double, String)]]): (Array[String], Double) = {

    // encode words with ids
    val encTrellis = Array(Array(($$(vocabIds)("_BOS_"), bosScore, "_BOS_"))) ++
      trellis.map(_.map { case (label, weight, cand) =>
        // at this point we keep only those candidates that are in the vocabulary
        ($$(vocabIds).get(label), weight, cand)
      }.filter(_._1.isDefined).map { case (x, y, z) => (x.get, y, z) }) ++
      Array(Array(($$(vocabIds)("_EOS_"), eosScore, "_EOS_")))

    // init
    var pathsIds = Array(Array($$(vocabIds)("_BOS_")))
    var pathWords = Array(Array("_BOS_"))
    var costs = Array(bosScore) // cost for each of the paths

    for (i <- 1 until encTrellis.length if pathsIds.forall(_.nonEmpty)) {

      var newPaths: Array[Array[Int]] = Array()
      var newWords: Array[Array[String]] = Array()
      var newCosts = Array[Double]()

      /* compute all the costs for all transitions in current step */
      val expPaths = pathsIds
        .map { p =>
          p :+ p.head
        }
        . // we need a placeholder, put the head.
        map(_.takeRight($(maxWindowLen)))
      val cids = expPaths.map(_.map { id =>
        $$(classes).apply(id)._1
      })
      val cwids = expPaths.map(_.map { id =>
        $$(classes).apply(id)._2
      })

      val candCids = encTrellis(i).map(_._1).map { id =>
        $$(classes).apply(id)._1
      }
      val candWids = encTrellis(i).map(_._1).map { id =>
        $$(classes).apply(id)._2
      }
      val expPathsCosts_ = getModelIfNotSet
        .predict_(
          pathsIds.map(_.takeRight($(maxWindowLen))),
          cids,
          cwids,
          candCids,
          candWids,
          configProtoBytes = getConfigProtoBytes)
        .toArray

      for { ((state, wcost, cand), idx) <- encTrellis(i).zipWithIndex } {
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

          if (cost < minCost) {
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
      pathWords.zip(costs).foreach { case (path, cost) =>
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

  def getClassCandidates(
      transducer: ITransducer[Candidate],
      token: String,
      label: String,
      maxDist: Int,
      limit: Int = 2) = {
    transducer
      .transduce(token, maxDist)
      .asScala
      .map { cand =>
        // if weights are available, we use them
        val weight = weights.get
          .map(ws => wLevenshteinDist(cand.term, token, ws))
          .getOrElse(cand.distance.toFloat)

        (cand.term, label, weight)
      }
      .toSeq
      .sortBy(_._3)
      .take(limit)
  }

  def getVocabCandidates(token: String, maxDist: Int) = {
    val trans = $$(transducer).transducer
    // we use all case information as it comes
    val plainCandidates =
      trans
        .transduce(token, maxDist)
        .asScala
        .toList
        .map(c => (c.term, c.term, c.distance.toFloat))

    // We evaluate some case variations
    val tryUpperCase = getOrDefault(caseStrategy) == CandidateStrategy.ALL_UPPER_CASE ||
      getOrDefault(caseStrategy) == CandidateStrategy.ALL

    val tryFirstCapitalized =
      getOrDefault(caseStrategy) == CandidateStrategy.FIRST_LETTER_CAPITALIZED ||
        getOrDefault(caseStrategy) == CandidateStrategy.ALL

    val caseCandidates = if (token.isUpperCase && tryUpperCase) {
      trans
        .transduce(token.toLowerCase)
        .asScala
        .toList
        .map(c => (c.term.toUpperCase, c.term, c.distance.toFloat))
    } else if (token.isFirstLetterCapitalized && tryFirstCapitalized) {
      trans
        .transduce(token.toLowerCase)
        .asScala
        .toList
        .map(c => (c.term.capitalizeFirstLetter, c.term, c.distance.toFloat))
    } else Seq.empty

    plainCandidates ++ caseCandidates
  }

  implicit class StringTools(s: String) {
    def isUpperCase() = s.toUpperCase.equals(s)

    def isLowerCase() = s.toLowerCase.equals(s)

    def isFirstLetterCapitalized() =
      s.headOption
        .map { fl =>
          fl.isUpper && s.tail.isLowerCase
        }
        .getOrElse(false)

    def capitalizeFirstLetter() = s.head.toUpper + s.tail
  }

  override def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    require(_model.isDefined, "Tensorflow model has not been initialized")
    dataset
  }

  /** takes a document and annotations and produces new annotations of this annotator's annotation
    * type
    *
    * @param annotations
    *   Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return
    *   any number of annotations processed for every input annotation. Not necessary one to one
    *   relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val decodedSentPaths =
      annotations.groupBy(_.metadata.getOrElse("sentence", "0").toInt).mapValues { sentTokens =>
        val (decodedPath, cost) = toOption(getOrDefault(useNewLines))
          .map { _ =>
            val idxs = Seq(-1) ++ sentTokens.zipWithIndex
              .filter { case (a, _) =>
                a.result.equals(System.lineSeparator) || a.result.equals(System.lineSeparator * 2)
              }
              .map(_._2) ++ Seq(annotations.length)
            idxs
              .zip(idxs.tail)
              .map { case (s, e) =>
                decodeViterbi(
                  computeTrellis(
                    sentTokens.slice(s + 1, e),
                    computeMask(sentTokens.slice(s + 1, e))))
              }
              .reduceLeft[(Array[String], Double)]({ case ((dPathA, pCostA), (dPathB, pCostB)) =>
                (dPathA ++ Seq(System.lineSeparator) ++ dPathB, pCostA + pCostB)
              })
          }
          .getOrElse(decodeViterbi(computeTrellis(sentTokens, computeMask(sentTokens))))
        // ToDo: This is a backup plan for empty DecodedPath -- fix me!!
        if (decodedPath.nonEmpty)
          sentTokens.zip(decodedPath).map { case (orig, correct) =>
            orig.copy(result = correct, metadata = orig.metadata.updated("cost", cost.toString))
          }
        else
          sentTokens.map(orig => orig.copy(metadata = orig.metadata.updated("cost", "0")))
      }

    decodedSentPaths.values.toList.reverse.flatten
  }

  def toOption(boolean: Boolean): Option[Boolean] = {
    if (boolean)
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
  def computeMask(annotations: Seq[Annotation]): Array[Boolean] = {
    val threshold = getOrDefault(errorThreshold)
    val unkCode = $$(vocabIds).get("_UNK_").get

    /* try to decide whether words need correction or not */
    // first pass - perplexities
    val encodedSent = Array($$(vocabIds)("_BOS_")) ++ annotations.map { ann =>
      if ($(compareLowcase))
        $$(vocabIds)
          .get(ann.result)
          .getOrElse($$(vocabIds).get(ann.result.toLowerCase).getOrElse(unkCode))
      else
        $$(vocabIds).get(ann.result).getOrElse(unkCode)
    } ++ Array($$(vocabIds)("_EOS_"))

    val cids = encodedSent.map { id =>
      $$(classes).apply(id)._1
    }
    val cwids = encodedSent.map { id =>
      $$(classes).apply(id)._2
    }

    val perplexities = getModelIfNotSet
      .pplEachWord(Array(encodedSent), Array(cids), Array(cwids))
      .map(_ > threshold)

    perplexities
      .zip(perplexities.tail)
      .zip(encodedSent.tail)
      .
      // if the word to the right needs correction, this word needs it too and is word in vocabulary ?
      map { case ((needCorrection, nextNeedCorrection), code) =>
        if (nextNeedCorrection) true else needCorrection || code == unkCode
      }
  }

  def computeTrellis(annotations: Seq[Annotation], mask: Seq[Boolean]) = {
    annotations
      .zip(mask)
      .map { case (annotation, needCorrection) =>
        val token = annotation.result
        var correctionCondition = needCorrection
        if (! $(correctSymbols))
          correctionCondition = needCorrection & token
            .replaceAll("[^A-Za-z0-9]+", "")
            .length > 0

        if (correctionCondition) {
          // ask each token class for candidates, keep the one with lower cost
          var candLabelWeight = $$(specialTransducers).flatMap { specialParser =>
            if (specialParser.transducer == null)
              throw new RuntimeException(s"${specialParser.label}")
            getClassCandidates(
              specialParser.transducer,
              token,
              specialParser.label,
              getOrDefault(wordMaxDistance) - 1)
          } ++ getVocabCandidates(token, getOrDefault(wordMaxDistance) - 1)

          // now try to relax distance requirements for candidates
          if (token.length > 4 && candLabelWeight.isEmpty)
            candLabelWeight = $$(specialTransducers).flatMap { specialParser =>
              getClassCandidates(
                specialParser.transducer,
                token,
                specialParser.label,
                getOrDefault(wordMaxDistance))
            } ++ getVocabCandidates(token, getOrDefault(wordMaxDistance))

          if (candLabelWeight.isEmpty)
            candLabelWeight = Array((token, "_UNK_", 3.0f))

          // label is a dictionary word for the main transducer, or a label such as _NUM_ for special classes
          val labelWeightCand = candLabelWeight
            .map { case (term, label, dist) =>
              // optional re-ranking of candidates according to special distance
              val d = get(weights)
                .map { w =>
                  wLevenshteinDist(term, token, w)
                }
                .getOrElse(dist)
              val weight = d - $$(vocabFreq).getOrElse(label, 0.0) / getOrDefault(gamma)
              (label, weight, term)
            }
            .sortBy(_._2)
            .take(getOrDefault(maxCandidates))
          logger.debug(
            s"""$token -> ${labelWeightCand.toList.take(getOrDefault(maxCandidates))}""")
          labelWeightCand.toArray // [(String, Double, String)]
        } else {
          Array(("_UNK_", .2, token))
        }
      }
      .toArray
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

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModel(
      path,
      spark,
      getModelIfNotSet.tensorflow,
      "_langmodeldl",
      ContextSpellCheckerModel.tfFile,
      configProtoBytes = getConfigProtoBytes)
  }
}

trait ReadsLanguageModelGraph
    extends ParamsAndFeaturesReadable[ContextSpellCheckerModel]
    with ReadTensorflowModel {

  override val tfFile = "tensorflow_lm"

  def readLanguageModelGraph(
      instance: ContextSpellCheckerModel,
      path: String,
      spark: SparkSession): Unit = {
    val tf = readTensorflowModel(path, spark, "_langmodeldl")
    instance.setModelIfNotSet(spark, tf)
  }

  addReader(readLanguageModelGraph)
}

trait ReadablePretrainedContextSpell
    extends ReadsLanguageModelGraph
    with HasPretrained[ContextSpellCheckerModel] {
  override val defaultModelName: Some[String] = Some("spellcheck_dl")

  /** Java compliant-overrides */
  override def pretrained(): ContextSpellCheckerModel = super.pretrained()

  override def pretrained(name: String): ContextSpellCheckerModel = super.pretrained(name)

  override def pretrained(name: String, lang: String): ContextSpellCheckerModel =
    super.pretrained(name, lang)

  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): ContextSpellCheckerModel = super.pretrained(name, lang, remoteLoc)
}

/** This is the companion object of [[ContextSpellCheckerModel]]. Please refer to that class for
  * the documentation.
  */
object ContextSpellCheckerModel extends ReadablePretrainedContextSpell
