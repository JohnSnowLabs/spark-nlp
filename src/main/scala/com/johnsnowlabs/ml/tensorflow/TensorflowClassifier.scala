package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.apache.spark.ml.util.Identifiable

import scala.util.Random

class TensorflowClassifier(val tensorflow: TensorflowWrapper, val encoder: ClassifierDatasetEncoder)
  extends Serializable {

  private val inputKey = "inputs:0"
  private val labelKey = "labels:0"
  private val learningRateKey = "lr:0"
  private val dropouttKey = "dp:0"

  private val numClasses: Int = encoder.params.tags.length

  private val predictionKey = s"softmax_output_$numClasses/Softmax:0"
  private val optimizer = s"optimizer_adam_$numClasses/Adam/Assign:0"
  private val cost = s"loss_$numClasses/softmax_cross_entropy_with_logits_sg:0"
  private val accuracy = s"accuracy_$numClasses/mean_accuracy:0"

  def train(
             inputs: Array[Array[Float]],
             labels: Array[String],
             lr: Float = 5e-3f,
             batchSize: Int = 64,
             dropout: Float = 0.5f,
             startEpoch: Int = 0,
             endEpoch: Int = 30,
             configProtoBytes: Option[Array[Byte]] = None,
             validationSplit: Float = 0.0f,
             evaluationLogExtended: Boolean = false,
             uuid: String = Identifiable.randomUID("annotator")
           ): Unit = {

    val encodedLabels = encoder.encodeTags(labels)
    val zippedInputsLabels = inputs.zip(encodedLabels).toSeq

    println(s"Training started - $endEpoch max Epoch - $batchSize batch size - ${zippedInputsLabels.length} training examples")

    for (epoch <- startEpoch until endEpoch) {

      val time = System.nanoTime()
      var batches = 0
      var loss = 0f
      var acc = 0f
      val learningRate = lr / (1 + 0.5 * epoch)
      val trainingDataset = Random.shuffle(zippedInputsLabels).toArray

      for (batch <- trainingDataset.grouped(batchSize)) {
        val tensors = new TensorResources()

        val inputArrays = batch.map(x => x._1)
        val labelsArray = batch.map(x => x._2)

        val inputTensor = tensors.createTensor(inputArrays)
        val labelTensor = tensors.createTensor(labelsArray)
        val lrTensor = tensors.createTensor(learningRate.toFloat)
        val dpTensor = tensors.createTensor(dropout.toFloat)

        val calculated = tensorflow
          .getSession(configProtoBytes = configProtoBytes)
          .runner
          .feed(inputKey, inputTensor)
          .feed(labelKey, labelTensor)
          .feed(learningRateKey, lrTensor)
          .feed(dropouttKey, dpTensor)
          .fetch(predictionKey)
          .fetch(optimizer)
          .fetch(cost)
          .fetch(accuracy)
          .run()

        loss += TensorResources.extractFloats(calculated.get(2))(0)
        acc += TensorResources.extractFloats(calculated.get(3))(0)
        batches += 1

        tensors.clearTensors()
      }
      acc /= (inputs.length / batchSize)

      println(s"Epoch $epoch, done in ${(System.nanoTime() - time)/1e9} accuracy: $acc loss: $loss, batches: $batches")

    }
  }

  def predict(docs: Seq[(Int, Seq[Annotation])], configProtoBytes: Option[Array[Byte]] = None): Seq[Annotation] = {

    val tensors = new TensorResources()

    //FixMe: (.head) Document or sentence as inputCols
    val inputs = docs.map(x => x._2.head.embeddings).toArray

    val calculated = tensorflow
      .getSession(configProtoBytes = configProtoBytes)
      .runner
      .feed(inputKey, tensors.createTensor(inputs))
      .fetch(predictionKey)
      .run()

    val tagsId = TensorResources.extractFloats(calculated.get(0)).grouped(numClasses).toArray
    val tagsName = encoder.decodeOutputData(tagIds = tagsId)
    tensors.clearTensors()

    docs.flatMap { sentence =>
      sentence._2.zip(tagsName).map {
        case (content, score) =>
          val label = score.find(_._1 == score.maxBy(_._2)._1).map(_._1).getOrElse("NA")

          Annotation(
            annotatorType = AnnotatorType.CATEGORY,
            begin = content.begin,
            end = content.end,
            result = label,
            metadata = Map("sentence" -> sentence._1.toString) ++ score.flatMap(x => Map(x._1 -> x._2.toString))
          )
      }

    }

  }

}
