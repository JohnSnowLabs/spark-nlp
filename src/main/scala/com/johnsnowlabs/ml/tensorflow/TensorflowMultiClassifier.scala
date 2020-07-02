package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.apache.spark.ml.util.Identifiable

import scala.collection.mutable
import scala.util.Random

class TensorflowMultiClassifier(
                                 val tensorflow: TensorflowWrapper,
                                 val encoder: ClassifierDatasetEncoder,
                                 override val verboseLevel: Verbose.Value
                               )
  extends Serializable with Logging {

  private val inputKey = "inputs:0"
  private val labelKey = "labels:0"
  private val sequenceLengthKey = "sequence_length:0"
  private val learningRateKey = "lr:0"
  private val dropouttKey = "dp:0"

  private val numClasses: Int = encoder.params.tags.length

  private val predictionKey = s"sigmoid_output_$numClasses/Sigmoid:0"
  private val optimizer = s"optimizer_adam_$numClasses/Adam/Assign:0"
  private val cost = s"loss_$numClasses/mean_cost:0"
  private val accuracy = s"accuracy_$numClasses/mean_accuracy:0"
  private val accuracyPerEntity = s"accuracy_$numClasses/mean_accuracy_per_entity:0"
  private val initKey = "init_all_tables"

  def reshapeInputFeatures(batch: Array[Array[Array[Float]]]): Array[Array[Array[Float]]] = {
    val sequencesLength = batch.map(x => x.length)
    val maxSentenceLength = sequencesLength.max
    //    val maxSentenceLength = 300
    val dimension = batch(0).head.length
    batch.map { sentence =>
      if (sentence.length >= maxSentenceLength) {
        sentence.take(maxSentenceLength)
      }else {
        val diff = maxSentenceLength - sentence.length
        sentence ++ Array.fill(diff)(Array.fill(dimension)(0.0f))
      }
    }
  }

  def train(
             inputs: Array[Array[Array[Float]]],
             labels: Array[Array[String]],
             classNum: Int,
             lr: Float = 5e-3f,
             batchSize: Int = 64,
             dropout: Float = 0.5f,
             startEpoch: Int = 0,
             endEpoch: Int = 10,
             configProtoBytes: Option[Array[Byte]] = None,
             validationSplit: Float = 0.0f,
             enableOutputLogs: Boolean = false,
             outputLogsPath: String,
             threshold: Float = 0.5f,
             uuid: String = Identifiable.randomUID("classifierdl")
           ): Unit = {

    // Initialize
    if (startEpoch == 0)
      tensorflow.createSession(configProtoBytes=configProtoBytes).runner.addTarget(initKey).run()

    val encodedLabels = encoder.encodeTagsMultiLabel(labels)
    val zippedInputsLabels = inputs.zip(encodedLabels).toSeq
    val trainingDataset = Random.shuffle(zippedInputsLabels)

    val sample: Int = (trainingDataset.length*validationSplit).toInt

    val (trainDatasetSeq, validateDatasetSample) = if (validationSplit > 0f) {
      val (trainingSample, trainingSet) = trainingDataset.splitAt(sample)
      (trainingSet.toArray, trainingSample.toArray)
    } else {
      // No validationSplit has been set so just use the entire training Dataset
      val emptyValid: Seq[(Array[Array[Float]], Array[Float])] = Seq((Array.empty, Array.empty))
      (trainingDataset.toArray, emptyValid.toArray)
    }

    println(s"Training started - total epochs: $endEpoch - learning rate: $lr - batch size: $batchSize - training examples: ${trainDatasetSeq.length} - classes: $classNum")
    outputLog(s"Training started - total epochs: $endEpoch - learning rate: $lr - batch size: $batchSize - training examples: ${trainDatasetSeq.length} - classes: $classNum",
      uuid, enableOutputLogs, outputLogsPath)

    for (epoch <- startEpoch until endEpoch) {

      val time = System.nanoTime()
      var batches = 0
      var loss = 0f
      var acc = 0f
      var accEntity = 0f
      val learningRate = lr / (1 + dropout * epoch)

      for (batch <- trainDatasetSeq.grouped(batchSize)) {
        val tensors = new TensorResources()

        val sequenceLengthArrays = batch.map(x => x._1.length)
        val inputArrays = reshapeInputFeatures(batch.map(x => x._1))
        val labelsArray = batch.map(x => x._2)

        val inputTensor = tensors.createTensor(inputArrays)
        val labelTensor = tensors.createTensor(labelsArray)
        val sequenceLengthTensor = tensors.createTensor(sequenceLengthArrays)
        val lrTensor = tensors.createTensor(learningRate.toFloat)
        val dpTensor = tensors.createTensor(dropout.toFloat)

        val calculated = tensorflow.getSession(configProtoBytes = configProtoBytes).runner
          .feed(inputKey, inputTensor)
          .feed(labelKey, labelTensor)
          .feed(sequenceLengthKey, sequenceLengthTensor)
          .feed(learningRateKey, lrTensor)
          .feed(dropouttKey, dpTensor)
          .fetch(predictionKey)
          .fetch(optimizer)
          .fetch(cost)
          .fetch(accuracy)
          .fetch(accuracyPerEntity)
          .run()

        loss += TensorResources.extractFloats(calculated.get(2))(0)
        acc += TensorResources.extractFloats(calculated.get(3))(0)
        accEntity += TensorResources.extractFloats(calculated.get(4))(0)

        batches += 1

        tensors.clearTensors()

      }
      acc /= (trainDatasetSeq.length / batchSize)
      accEntity /= (trainDatasetSeq.length / batchSize)

      if (validationSplit > 0.0) {
        val validationAccuracy = measure(validateDatasetSample, (s: String) => log(s, Verbose.Epochs), threshold=threshold)
        val endTime = (System.nanoTime() - time)/1e9
        println(f"Epoch ${epoch+1}/$endEpoch - $endTime%.2fs - loss: $loss - accuracy: $acc - accuracy_entity: $accEntity - validation: $validationAccuracy - batches: $batches")
        outputLog(s"Epoch $epoch/$endEpoch - $endTime%.2fs - loss: $loss - accuracy: $acc - accuracy_entity: $accEntity - validation: $validationAccuracy - batches: $batches", uuid, enableOutputLogs, outputLogsPath)
      }else{
        val endTime = (System.nanoTime() - time)/1e9
        println(f"Epoch ${epoch+1}/$endEpoch - $endTime%.2fs - loss: $loss - accuracy: $acc - accuracy_entity: $accEntity - batches: $batches")
        outputLog(s"Epoch $epoch/$endEpoch - $endTime%.2fs - loss: $loss - accuracy: $acc - accuracy_entity: $accEntity - batches: $batches", uuid, enableOutputLogs, outputLogsPath)
      }

    }
  }

  def predict(docs: Seq[(Int, Seq[Annotation])], threshold: Float = 0.5f, configProtoBytes: Option[Array[Byte]] = None): Seq[Annotation] = {

    val tensors = new TensorResources()

    val inputs = encoder.extractSentenceEmbeddingsMultiLabelPredict(docs)

    val sequenceLengthArrays = inputs.map(x => x.length)
    val inputsReshaped = reshapeInputFeatures(inputs)

    val calculated = tensorflow
      .getSession(configProtoBytes = configProtoBytes)
      .runner
      .feed(inputKey, tensors.createTensor(inputsReshaped))
      .feed(sequenceLengthKey, tensors.createTensor(sequenceLengthArrays))
      .fetch(predictionKey)
      .run()

    val tagsId = TensorResources.extractFloats(calculated.get(0)).grouped(numClasses).toArray
    val tagsName = encoder.decodeOutputData(tagIds = tagsId)
    tensors.clearTensors()

    tagsName.map { case (score) =>
      val labels = score.filter(x=>x._2 >= threshold).map(x=>x._1).mkString(" ")
      val documentBegin = docs.head._2.head.begin
      val documentEnd = docs.last._2.last.end

      Annotation(
        annotatorType = AnnotatorType.CATEGORY,
        begin = documentBegin,
        end = documentEnd,
        result = labels,
        metadata = Map("sentence" -> "0") ++ score.flatMap(x => Map(x._1 -> x._2.toString))
      )
    }
  }

  def internalPredict(inputs: Array[Array[Array[Float]]], threshold: Float = 0.5f, configProtoBytes: Option[Array[Byte]] = None): Array[Array[Float]] = {

    val tensors = new TensorResources()

    val sequenceLengthArrays = inputs.map(x => x.length)
    val inputsReshaped = reshapeInputFeatures(inputs)

    val calculated = tensorflow
      .getSession(configProtoBytes = configProtoBytes)
      .runner
      .feed(inputKey, tensors.createTensor(inputsReshaped))
      .feed(sequenceLengthKey, tensors.createTensor(sequenceLengthArrays))
      .fetch(predictionKey)
      .run()

    val tagsId = TensorResources.extractFloats(calculated.get(0)).grouped(numClasses).toArray
    val scores = tagsId.map{score=>
      score.zipWithIndex.map(x => x._1 >= threshold).map { y =>
        if(y) 1.0f else 0.0f
      }
    }
    tensors.clearTensors()
    scores
  }

  def measure(labeled: Array[(Array[Array[Float]], Array[Float])],
              log: String => Unit,
              extended: Boolean = false,
              batchSize: Int = 100,
              threshold: Float = 0.5f
             ): Float = {

    var correct = 0
    var totalLabels = 0

    for (batch <- labeled.grouped(batchSize)) {

      val originalEmbeddings = batch.map(x => x._1)
      val originalLabels = batch.map(x => x._2)

      val predictedLabels = internalPredict(originalEmbeddings, threshold = 0.5f)
      val labeledPredicted = predictedLabels.zip(originalLabels)
      totalLabels += originalLabels.map(x=>x.count(x => x != 0)).sum

      for (i <- labeledPredicted) {
        val predict = i._1
        val original = i._2

        predict.zip(original).map {case (pred, orig)=>
          if(orig != 0.0f && pred == orig) correct+=1
        }
      }
    }

    (correct.toFloat / totalLabels.toFloat) * 100
  }

}
