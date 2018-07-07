package com.johnsnowlabs.nlp.annotators.pos.perceptron

import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach, AnnotatorType}
import com.johnsnowlabs.nlp.annotators.common.{IndexedTaggedWord, TaggedSentence}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{IntParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset
import org.apache.spark.util.LongAccumulator
import org.apache.spark.sql.functions.rand

import scala.collection.mutable.{ListBuffer, Map => MMap}

/**
  * Created by Saif Addin on 5/17/2017.
  * Inspired on Averaged Perceptron by Matthew Honnibal
  * https://explosion.ai/blog/part-of-speech-pos-tagger-in-python
  */
class PerceptronApproach(override val uid: String) extends AnnotatorApproach[PerceptronModel] with PerceptronUtils {

  import com.johnsnowlabs.nlp.AnnotatorType._

  override val description: String = "Averaged Perceptron model to tag words part-of-speech"

  val posCol = new Param[String](this, "posCol", "column of Array of POS tags that match tokens")
  val corpus = new ExternalResourceParam(this, "corpus", "POS tags delimited corpus. Needs 'delimiter' in options")
  val nIterations = new IntParam(this, "nIterations", "Number of iterations in training, converges to better accuracy")

  setDefault(nIterations, 5)

  def setPosColumn(value: String): this.type = set(posCol, value)

  def setCorpus(value: ExternalResource): this.type = {
    require(value.options.contains("delimiter"), "PerceptronApproach needs 'delimiter' in options to associate words with tags")
    set(corpus, value)
  }

  def setCorpus(path: String,
                delimiter: String,
                readAs: ReadAs.Format = ReadAs.SPARK_DATASET,
                options: Map[String, String] = Map("format" -> "text")): this.type =
    set(corpus, ExternalResource(path, readAs, options ++ Map("delimiter" -> delimiter)))

  def setNIterations(value: Int): this.type = set(nIterations, value)

  def this() = this(Identifiable.randomUID("POS"))

  override val annotatorType: AnnotatorType = POS

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)

  /**
    * Finds very frequent tags on a word in training, and marks them as non ambiguous based on tune parameters
    * ToDo: Move such parameters to configuration
    *
    * @param taggedSentences    Takes entire tagged sentences to find frequent tags
    * @param frequencyThreshold How many times at least a tag on a word to be marked as frequent
    * @param ambiguityThreshold How much percentage of total amount of words are covered to be marked as frequent
    */
  private def buildTagBook(
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

  private def update(
              truth: String,
              guess: String,
              features: Iterable[String],
              updateIteration: Long,
              partitionWeights: MMap[String, MMap[String, Double]],
              partitionTotals: MMap[(String, String), Double],
              partitionTimestamps: MMap[(String, String), Long]): Unit = {
    def updateFeature(tag: String, feature: String, weight: Double, value: Double): Unit = {
      /**
        * update totals and timestamps
        */
      val param = (feature, tag)
      partitionTotals.update(param, partitionTotals.getOrElse(param, 0.0) + ((updateIteration - partitionTimestamps.getOrElse(param, 0L)) * weight))
      partitionTimestamps.update(param, updateIteration)
      /**
        * update weights
        */
      partitionWeights.update(feature, partitionWeights.getOrElse(feature, MMap()) ++ MMap(tag -> (weight + value)))
    }
    /**
      * if prediction was wrong, take all features and for each feature get feature's current tags and their weights
      * congratulate if success and punish for wrong in weight
      */
    if (truth != guess) {
      features.foreach{feature =>
        val weights = partitionWeights.getOrElse(feature, MMap())
        updateFeature(truth, feature, weights.getOrElse(truth, 0.0), 1.0)
        updateFeature(guess, feature, weights.getOrElse(guess, 0.0), -1.0)
      }
    }
  }

  private[pos] def averageWeights(
                                   tags: Broadcast[Array[String]],
                                   taggedWordBook: Broadcast[Map[String, String]],
                                   featuresWeight: StringMapStringDoubleAccumulator,
                                   updateIteration: LongAccumulator,
                                   totals: StringTupleDoubleAccumulator,
                                   timestamps: TupleKeyLongMapAccumulator
                                   ): AveragedPerceptron = {
    val fw = featuresWeight.value
    val uiv = updateIteration.value
    val tv = totals.value
    val tmv = timestamps.value
    featuresWeight.reset()
    updateIteration.reset()
    totals.reset()
    timestamps.reset()
    val finalfw = fw.map { case (feature, weights) =>
      (feature, weights.map { case (tag, weight) =>
        val param = (feature, tag)
        val total = tv.getOrElse(param, 0.0) + ((uiv - tmv.getOrElse(param, 0L)) * weight)
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
    val timestampsAcc = new TupleKeyLongMapAccumulator()
    val updateIterationAcc = new LongAccumulator()
    val totalsAcc = new StringTupleDoubleAccumulator()
    dataset.sparkSession.sparkContext.register(featuresWeightAcc)
    dataset.sparkSession.sparkContext.register(timestampsAcc)
    dataset.sparkSession.sparkContext.register(updateIterationAcc)
    dataset.sparkSession.sparkContext.register(totalsAcc)

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
        dataset.sparkSession.sparkContext.broadcast(timestampsAcc.value)
      }

      val iterationWeights = if (iteration == 1 ) {
        dataset.sparkSession.sparkContext.broadcast(Map.empty[String, Map[String, Double]])
      } else {
        dataset.sparkSession.sparkContext.broadcast(featuresWeightAcc.value)
      }

      val iterationTotals = if (iteration == 1 ) {
        dataset.sparkSession.sparkContext.broadcast(Map.empty[(String, String), Double])
      } else {
        dataset.sparkSession.sparkContext.broadcast(totalsAcc.value)
      }

      val iterationUpdateCount = if (iteration == 1 ) {
        dataset.sparkSession.sparkContext.broadcast[Long](0L)
      } else {
        dataset.sparkSession.sparkContext.broadcast[Long](updateIterationAcc.value)
      }

      val sortedSentences = taggedSentences.sort(rand())
        .repartition($(corpus).options.get("repartition").map(_.toInt)
          .getOrElse(8))
        /** Cache of iteration datasets does not show any improvements, try sample? */
        //.cache

      sortedSentences.foreachPartition(partition => {

        val _temp1 = ListBuffer.empty[((String, String), Long)]
        iterationTimestamps.value.copyToBuffer(_temp1)
        val partitionTimestamp = MMap(_temp1:_*)
        _temp1.clear()

        val _temp2 = ListBuffer.empty[(String, Map[String, Double])]
        iterationWeights.value.copyToBuffer(_temp2)
        val partitionWeights = MMap(_temp2.map{case (k,v) => (k, MMap(v.toSeq:_*))}:_*)
        _temp2.clear()

        val _temp3 = ListBuffer.empty[((String, String), Double)]
        iterationTotals.value.copyToBuffer(_temp3)
        val partitionTotals = MMap(_temp3:_*)
        _temp3.clear()

        var partitionUpdateCount: Long = iterationUpdateCount.value

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
              taggedWordBook.value.getOrElse(word.toLowerCase, {
                val features = getFeatures(i, word, context, prev, prev2)
                val model = TrainingPerceptron(classes.value, partitionWeights)
                val guess = model.predict(features)
                partitionUpdateCount += 1L
                update(taggedSentence.tags(i), guess, features.keys, partitionUpdateCount, partitionWeights, partitionTotals, partitionTimestamp)
                guess
              })

            /**
              * shift the context
              */
            prev2 = prev
            prev = guess
          }

        }
        synchronized {
          featuresWeightAcc.addMany(partitionWeights)
          timestampsAcc.updateMany(partitionTimestamp)
          totalsAcc.updateMany(partitionTotals)
          updateIterationAcc.add(partitionUpdateCount)
        }
      })
      //sortedSentences.unpersist()
      iterationTimestamps.unpersist()
      iterationWeights.unpersist()
      iterationUpdateCount.unpersist()
    }}
    logger.debug("TRAINING: Finished all iterations")
    new PerceptronModel().setModel(averageWeights(classes, taggedWordBook, featuresWeightAcc, updateIterationAcc, totalsAcc, timestampsAcc))
  }
}

object PerceptronApproach extends DefaultParamsReadable[PerceptronApproach]