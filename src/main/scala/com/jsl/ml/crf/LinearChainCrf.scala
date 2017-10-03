package com.jsl.ml.crf

import VectorMath._
import org.slf4j.{Logger, LoggerFactory}

import scala.util.Random

// ToDo Make c0 estimation before training
class LinearChainCrf(val params: TrainParams) {

  private val logger = LoggerFactory.getLogger("CRF")

  def log(value: => String, minLevel: Verbose.Level): Unit = {
    if (minLevel >= params.verbose) {
      logger.info(value)
    }
  }

  def trainSGD(dataset: CrfDataset): LinearChainCrfModel = {
    val metadata = dataset.metadata
    val weights = Vector(dataset.metadata.attrFeatures.size + dataset.metadata.transitions.size)
    val labels = dataset.metadata.labels.size

    if (params.randomSeed.isDefined)
      Random.setSeed(params.randomSeed.get)

    // 1. Calc max sentence Length
    val maxLength = dataset.instances.map(w => w._2.items.size).max

    log(s"labels: $labels", Verbose.TrainingStat)
    log(s"instances: ${dataset.instances.size}", Verbose.TrainingStat)
    log(s"features: ${weights.size}", Verbose.TrainingStat)
    log(s"maxLength: $maxLength", Verbose.TrainingStat)


    // 2. Allocate reusable space
    val context = new FbCalculator(maxLength, metadata)

    var bestW = Vector(weights.size, 0f)
    var bestLoss = Float.MaxValue
    var lastLoss = Float.MaxValue

    var notImprovedEpochs = 0

    val decayStrategy = new L2DecayStrategy(dataset.instances.size, params.l2, params.c0)

    for (epoch <- 0 until params.maxEpochs
         if notImprovedEpochs < 10 || epoch < params.minEpochs) {

      var loss = 0f

      log(s"\nEpoch: $epoch, eta: ${decayStrategy.eta}", Verbose.Epochs)
      val started = System.nanoTime()

      val shuffled = Random.shuffle(dataset.instances)
      for ((labels, sentence) <- shuffled) {
        decayStrategy.nextStep()

        // 1. Calculate values for further usage
        context.calculate(sentence, weights, decayStrategy.getScale)

        // 2. Make one gradient step
        doSgdStep(sentence, labels, decayStrategy.alpha, weights, context)

        // 3. Calculate loss
        loss += getLoss(sentence, labels, context)
      }

      // Return weights to normal values
      decayStrategy.reset(weights)

      val l2Loss = params.l2 * weights.map(w => w*w).sum

      val totalLoss = loss + l2Loss

      log(s"finished, time: ${(System.nanoTime() - started)/1e9}", Verbose.Epochs)
      log(s"Loss = $totalLoss, logLoss = $loss, l2Loss = $l2Loss", Verbose.Epochs)

      // Update best solution if loss is lower
      if (totalLoss < bestLoss) {
        bestLoss = totalLoss
        copy(weights, bestW)

        if ((bestLoss - totalLoss)/totalLoss < params.lossEps)
          notImprovedEpochs = 0
        else
          notImprovedEpochs += 1
      }
      else
        notImprovedEpochs += 1

      lastLoss = totalLoss
    }

    new LinearChainCrfModel(bestW, metadata)
  }

  private def getLoss(sentence: Instance, labels: InstanceLabels, context: FbCalculator): Float = {
    val length = sentence.items.length

    var prevLabel = 0
    var result = 0f
    for (i <- 0 until length) {
      result -= context.logPhi(i)(prevLabel)(labels.labels(i))
      prevLabel = labels.labels(i)

      result += Math.log(context.c(i)).toFloat
    }

    if (result >= 0) {
      assert(result >= 0)
    }

    result
  }

  // Step for minimizing model Log Likelihood
  def doSgdStep(sentence: Instance,
                labels: InstanceLabels,
                a: Float,
                weights: Array[Float],
                context: FbCalculator): Unit = {

    // Make Gradient Step
    // Minimizing -log likelihood
    // Gradient = [Observed Expectation] - [Model Expectations]
    // Weights = Weights + a*Gradient

    // 1. Plus Observed Expectation
    context.addObservedExpectations(weights, sentence, labels, a)

    // 2. Minus Model Expectations
    context.addModelExpectations(weights, sentence, -a)
  }
}

class L2DecayStrategy(val instances: Int,
                      val l2: Float,
                      val c0: Float = 1000
                     ) {

  // Correct weights is equal weights * scale
  private var scale: Float = 1f

  // Number of step SGD
  private var step = 0

  // Regularization for one instance
  private val lambda = 2f*l2 / instances

  def getScale: Float = scale

  // Scaled coefficient for Gradient step
  def alpha: Float = eta / scale

  // Real coefficient for Gradient step
  def eta: Float = 1f / (lambda * (step + c0))

  def nextStep(): Unit = {
    step += 1
    scale = scale * (1f - eta * lambda)
  }

  def reset(weights: Vector): Unit = {
    VectorMath.multiply(weights, scale)
    scale = 1f
  }

}


object Verbose extends Enumeration {
  type Level = Value

  val All = Value(0)
  val PerStep = Value(1)
  val Epochs = Value(2)
  val TrainingStat = Value(3)
  val Silent = Value(4)
}

case class TrainParams
(
  minEpochs: Int = 10,
  maxEpochs: Int = 1000,
  l2: Float = 1f,
  verbose: Verbose.Level = Verbose.Silent,
  randomSeed: Option[Int] = None,
  lossEps: Float = 1e-4f,
  c0: Int = 1500000
)