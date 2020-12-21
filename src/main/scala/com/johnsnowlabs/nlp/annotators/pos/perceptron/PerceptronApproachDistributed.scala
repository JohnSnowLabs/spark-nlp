package com.johnsnowlabs.nlp.annotators.pos.perceptron

import com.johnsnowlabs.nlp.annotators.common.{IndexedTaggedWord, TaggedSentence}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach, AnnotatorType}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{IntParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.rand
import org.apache.spark.util.LongAccumulator

import scala.collection.mutable.{ListBuffer, Map => MMap}

/**
  * Distributed Averaged Perceptron model to tag words part-of-speech.
  *
  * Sets a POS tag to each word within a sentence. Its train data (train_pos) is a spark dataset of POS format values with Annotation columns.
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/pos/perceptron/DistributedPos.scala]] for further reference on how to use this APIs.
  *
  * @param uid internal uid required to generate writable annotators
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
  * @groupdesc Parameters A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  **/
class PerceptronApproachDistributed(override val uid: String) extends AnnotatorApproach[PerceptronModel]
  with PerceptronTrainingUtils
{

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Averaged Perceptron model to tag words part-of-speech */
  override val description: String = "Averaged Perceptron model to tag words part-of-speech"

  /** column of Array of POS tags that match tokens
    *
    * @group param
    **/
  val posCol = new Param[String](this, "posCol", "column of Array of POS tags that match tokens")
  /** POS tags delimited corpus. Needs 'delimiter' in options
    *
    * @group param
    **/
  val corpus = new ExternalResourceParam(this, "corpus", "POS tags delimited corpus. Needs 'delimiter' in options")
  /** Number of iterations in training, converges to better accuracy
    *
    * @group param
    **/
  val nIterations = new IntParam(this, "nIterations", "Number of iterations in training, converges to better accuracy")

  setDefault(nIterations, 5)

  /** Column containing an array of POS Tags matching every token on the line.
    *
    * @group setParam
    **/
  def setPosColumn(value: String): this.type = set(posCol, value)

  /** POS tags delimited corpus. Needs 'delimiter' in options
    *
    * @group setParam
    **/
  def setCorpus(value: ExternalResource): this.type = {
    require(value.options.contains("delimiter"), "PerceptronApproach needs 'delimiter' in options to associate words with tags")
    set(corpus, value)
  }

  /** POS tags delimited corpus. Needs 'delimiter' in options
    *
    * @group setParam
    **/
  def setCorpus(path: String,
                delimiter: String,
                readAs: ReadAs.Format = ReadAs.SPARK,
                options: Map[String, String] = Map("format" -> "text")): this.type =
    set(corpus, ExternalResource(path, readAs, options ++ Map("delimiter" -> delimiter)))

  /** Number of iterations for training. May improve accuracy but takes longer. Default 5.
    *
    * @group setParam
    **/
  def setNIterations(value: Int): this.type = set(nIterations, value)

  def this() = this(Identifiable.randomUID("POS"))

  /** Output annotator types : POS
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = POS
  /** Input annotator types : TOKEN, DOCUMENT
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)

  /**
    * Finds very frequent tags on a word in training, and marks them as non ambiguous based on tune parameters
    * ToDo: Move such parameters to configuration
    *
    * @param taggedSentences    Takes entire tagged sentences to find frequent tags
    * @param frequencyThreshold How many times at least a tag on a word to be marked as frequent
    * @param ambiguityThreshold How much percentage of total amount of words are covered to be marked as frequent
    */
  def buildTagBook(
                            taggedSentences: Dataset[TaggedSentence],
                            frequencyThreshold: Int = 20,
                            ambiguityThreshold: Double = 0.97
                          ): Map[String, String] = {
    import ResourceHelper.spark.implicits._
    val tagFrequenciesByWord = taggedSentences
      .flatMap(_.taggedWords)
      .groupByKey(tw => tw.word.toLowerCase)
      .mapGroups{case (lw, tw) => (lw, tw.toSeq.groupBy(_.tag).mapValues(_.length))}
      .filter { lwtw =>
        val (_, mode) = lwtw._2.maxBy(t => t._2)
        val n = lwtw._2.values.sum
        n >= frequencyThreshold && (mode / n.toDouble) >= ambiguityThreshold
      }

    tagFrequenciesByWord.map { case (word, tagFrequencies) =>
      val (tag, _) = tagFrequencies.maxBy(_._2)
      logger.debug(s"TRAINING: Ambiguity discarded on: << $word >> set to: << $tag >>")
      (word, tag)
    }.collect.toMap
  }

  private[pos] def averageWeights(
                                   tags: Broadcast[Array[String]],
                                   taggedWordBook: Broadcast[Map[String, String]],
                                   featuresWeight: StringMapStringDoubleAccumulator,
                                   updateIteration: LongAccumulator,
                                   timetotals: TupleKeyLongDoubleMapAccumulator
                                 ): AveragedPerceptron = {
    val fw = featuresWeight.value
    val uiv = updateIteration.value
    val totals = timetotals.value
    featuresWeight.reset()
    updateIteration.reset()
    timetotals.reset()
    val finalfw = fw.map { case (feature, weights) =>
      (feature, weights.map { case (tag, weight) =>
        val param = (feature, tag)
        val total = totals.get(param).map(_._2).getOrElse(0.0) + ((uiv - totals.get(param).map(_._1).getOrElse(0L)) * weight)
        (tag, total / uiv.toDouble)
      })
    }
    val apr = AveragedPerceptron(
      tags.value,
      taggedWordBook.value,
      finalfw
    )
    taggedWordBook.destroy()
    tags.destroy()
    apr
  }

  /**
    * Trains a model based on a provided CORPUS
    *
    * @return A trained averaged model
    */
  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): PerceptronModel = {

    val featuresWeightAcc = new StringMapStringDoubleAccumulator()
    val timeTotalsAcc = new TupleKeyLongDoubleMapAccumulator()
    val updateIterationAcc = new LongAccumulator()
    dataset.sparkSession.sparkContext.register(featuresWeightAcc)
    dataset.sparkSession.sparkContext.register(timeTotalsAcc)
    dataset.sparkSession.sparkContext.register(updateIterationAcc)

    /**
      * Generates TagBook, which holds all the word to tags mapping that are not ambiguous
      */
    val taggedSentences: Dataset[TaggedSentence] = if (get(posCol).isDefined) {
      import ResourceHelper.spark.implicits._
      val tokenColumn = dataset.schema.fields
        .find(f => f.metadata.contains("annotatorType") && f.metadata.getString("annotatorType") == AnnotatorType.TOKEN)
        .map(_.name).get
      dataset.select(tokenColumn, $(posCol))
        .as[(Array[Annotation], Array[String])]
        .map{
          case (annotations, posTags) =>
            lazy val strTokens = annotations.map(_.result).mkString("#")
            lazy val strPosTags = posTags.mkString("#")
            require(annotations.length == posTags.length, s"Cannot train from $posCol since there" +
              s" is a row with different amount of tags and tokens:\n$strTokens\n$strPosTags")
            TaggedSentence(annotations.zip(posTags)
              .map{case (annotation, posTag) => IndexedTaggedWord(annotation.result, posTag, annotation.begin, annotation.end)}
            )
        }
    } else {
      ResourceHelper.parseTupleSentencesDS($(corpus))
    }

    val nPartitions = $(corpus).options.get("repartition").map(_.toInt).getOrElse(0)
    val doCache = $(corpus).options.get("cache").exists(_.toBoolean == true)
    val repartitioned = if (nPartitions > 0 && nPartitions != taggedSentences.rdd.partitions.length)
      taggedSentences.repartition(nPartitions)
    else
      taggedSentences

    val cachedSentences = if (doCache)
      repartitioned.cache
    else
      repartitioned

    val taggedWordBook = dataset.sparkSession.sparkContext.broadcast(buildTagBook(taggedSentences))
    /** finds all distinct tags and stores them */
    val classes = {
      import ResourceHelper.spark.implicits._
      dataset.sparkSession.sparkContext.broadcast(taggedSentences.flatMap(_.tags).distinct.collect)
    }

    /**
      * Iterates for training
      */
    (1 to $(nIterations)).foreach { iteration => {
      logger.debug(s"TRAINING: Iteration n: $iteration")

      val iterationTimestamps = if (iteration == 1 ) {
        dataset.sparkSession.sparkContext.broadcast(Map.empty[(String, String), Long])
      } else {
        dataset.sparkSession.sparkContext.broadcast(timeTotalsAcc.value.mapValues(_._1))
      }

      val iterationWeights = if (iteration == 1 ) {
        dataset.sparkSession.sparkContext.broadcast(Map.empty[String, Map[String, Double]])
      } else {
        dataset.sparkSession.sparkContext.broadcast(featuresWeightAcc.value)
      }

      val iterationUpdateCount = if (iteration == 1 ) {
        dataset.sparkSession.sparkContext.broadcast[Long](0L)
      } else {
        dataset.sparkSession.sparkContext.broadcast[Long](updateIterationAcc.value)
      }

      val sortedSentences = cachedSentences.sort(rand()).sortWithinPartitions(rand())

      /** Cache of iteration datasets does not show any improvements, try sample? */

      sortedSentences.foreachPartition(partition => {

        val _temp1 = ListBuffer.empty[((String, String), Long)]
        iterationTimestamps.value.copyToBuffer(_temp1)
        val newPartitionTimeTotals = MMap.empty[(String, String), (Long, Double)]
        val partitionTimestamps = _temp1.toMap
        _temp1.clear()

        val _temp2 = ListBuffer.empty[(String, Map[String, Double])]
        iterationWeights.value.copyToBuffer(_temp2)
        val newPartitionWeights = MMap.empty[String, MMap[String, Double]]
        val partitionWeights = _temp2.toMap
        _temp2.clear()

        var partitionUpdateCount: Long = iterationUpdateCount.value
        val partitionUpdateCountOriginal = partitionUpdateCount

        val partitionTotals: MMap[(String, String), Double] = MMap.empty[(String, String), Double]

        val twb = taggedWordBook.value
        val cls = classes.value

        def update(
                    truth: String,
                    guess: String,
                    features: Iterable[String]): Unit = {
          def updateFeature(tag: String, feature: String, weight: Double, value: Double): Unit = {
            /**
              * update totals and timestamps
              */
            val param = (feature, tag)
            val newTimestamp = partitionUpdateCount
            partitionTotals.update(param, partitionTotals.getOrElse(param, 0.0) + ((newTimestamp - newPartitionTimeTotals.get(param).map(_._1).getOrElse(partitionTimestamps.getOrElse(param, 0L))) * weight))
            newPartitionTimeTotals.update(param, (newTimestamp, partitionTotals(param)))
            /**
              * update weights
              */
            val newWeights = newPartitionWeights.getOrElse(feature, MMap()) ++ MMap(tag -> (weight + value))
            newPartitionWeights.update(feature, newWeights)
          }
          /**
            * if prediction was wrong, take all features and for each feature get feature's current tags and their weights
            * congratulate if success and punish for wrong in weight
            */
          if (truth != guess) {
            features.foreach{feature =>
              val weights = newPartitionWeights.get(feature).map(pw => partitionWeights.getOrElse(feature, Map()) ++ pw).orElse(partitionWeights.get(feature)).getOrElse(Map())
              updateFeature(truth, feature, weights.getOrElse(truth, 0.0), 1.0)
              updateFeature(guess, feature, weights.getOrElse(guess, 0.0), -1.0)
            }
          }
        }

        def predict(features: Map[String, Int]): String = {
          /**
            * scores are used for feature scores, which are all by default 0
            * if a feature has a relevant score, look for all its possible tags and their scores
            * multiply their weights per the times they appear
            * Return highest tag by score
            *
            */
          val scoresByTag = features
            .filter{case (feature, value) => (partitionWeights.contains(feature) || newPartitionWeights.contains(feature)) && value != 0}
            .map{case (feature, value ) =>
              newPartitionWeights.get(feature).map(pw => partitionWeights.getOrElse(feature, Map()) ++ pw).getOrElse(partitionWeights(feature))
                .map{ case (tag, weight) =>
                  (tag, value * weight)
                }
            }.aggregate(Map[String, Double]())(
            (tagsScores, tagScore) => tagScore ++ tagsScores.map{case(tag, score) => (tag, tagScore.getOrElse(tag, 0.0) + score)},
            (pTagScore, cTagScore) => pTagScore.map{case (tag, score) => (tag, cTagScore.getOrElse(tag, 0.0) + score)}
          )
          /**
            * ToDo: Watch it here. Because of missing training corpus, default values are made to make tests pass
            * Secondary sort by tag simply made to match original python behavior
            */
          cls.maxBy{ tag => (scoresByTag.getOrElse(tag, 0.0), tag)}
        }

        /**
          * In a shuffled sentences list, try to find tag of the word, hold the correct answer
          */
        partition.foreach { taggedSentence =>

          /**
            * Defines a sentence context, with room to for look back
            */
          var prev = START(0)
          var prev2 = START(1)
          val context = START ++: taggedSentence.words.map(w => normalized(w)) ++: END
          taggedSentence.words.zipWithIndex.foreach { case (word, i) =>
            val guess =
              twb.getOrElse(word.toLowerCase, {
                val features = getFeatures(i, word, context, prev, prev2)
                val guess = predict(features)
                partitionUpdateCount += 1L
                update(taggedSentence.tags(i), guess, features.keys)
                guess
              })

            /**
              * shift the context
              */
            prev2 = prev
            prev = guess
          }

        }
        featuresWeightAcc.addMany(newPartitionWeights)
        timeTotalsAcc.updateMany(newPartitionTimeTotals)
        updateIterationAcc.add(partitionUpdateCount - partitionUpdateCountOriginal)
      })
      if (doCache) {sortedSentences.unpersist()}
      iterationTimestamps.unpersist(true)
      iterationWeights.unpersist(true)
      iterationUpdateCount.unpersist(true)
    }}
    logger.debug("TRAINING: Finished all iterations")
    new PerceptronModel().setModel(averageWeights(classes, taggedWordBook, featuresWeightAcc, updateIterationAcc, timeTotalsAcc))
  }
}

object PerceptronApproachDistributed extends DefaultParamsReadable[PerceptronApproachDistributed]