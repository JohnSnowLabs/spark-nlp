package com.johnsnowlabs.ml.lstm

import java.util
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import scala.collection.JavaConversions._


/**
  *  Created by jose on 21/08/17.
  *
  *  @param maxSampleLength: max length (leftCtx + targetSize + rightCtx)
  *  @param embeddings: this is an array of variable length sequences of embeddings
  *                   (which are in turn also arrays, but fixed length)
  */

/* TODO refactor Windowing -> FixedWindowFeatures
*                new Class -> VariableLengthFeatures (to produce embeddings: Array[Array[Array[Double]]])
* */

class LSTMRecordIterator(embeddings: Array[Array[Array[Float]]],
                         labels: Array[Int], maxSampleLength:Int = 250) extends DataSetIterator {

  var cursor = 0

  /* batch size, how many samples at the same time? */
  val batchSize = 64
  val datasetSize = embeddings.length
  val vectorSize = embeddings.head.head.size

  /* how many classes we have for classification */
  val classes = labels.distinct.length

  override def next(num: Int): DataSet = {
    if (cursor >= datasetSize) throw new NoSuchElementException
    nextDataSet(num)
  }

  /* looks for the next batch of size 'num' */
  private def nextDataSet(num : Int) : DataSet = {
    // pick the minimum from what is requested and what we have
    val possibleNum = Math.min(num, datasetSize - cursor)

    val features = Nd4j.create(Array(possibleNum, vectorSize, maxSampleLength), 'f')
    val labels = Nd4j.create(Array(possibleNum, classes, maxSampleLength), 'f')

    //Because we are dealing with sentences of different lengths and only one output at the final time step: use padding arrays
    //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
    val featuresMask = Nd4j.zeros(possibleNum, maxSampleLength)
    val labelsMask = Nd4j.zeros(possibleNum, maxSampleLength)
    var temp = Array.fill(2)(0)

    var i = 0


    while (i < possibleNum && cursor < datasetSize) {
      temp = Array(i, temp(1))
      val sentenceVectors = embeddings(cursor)

      //Get word vectors for each word in sentence, and put them in the training data
      sentenceVectors.zipWithIndex.foreach { case(vector, j) =>
        features.put(Array(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)), Nd4j.create(vector))
        temp = Array(temp(0), j)
        //Word is present (not padding) for this example + time step -> 1.0 in features mask
        featuresMask.putScalar(temp, 1.0);
      }

      val lastIdx = Math.min(sentenceVectors.size, maxSampleLength)
      if(this.labels.size > 0) {
        val idx = this.labels(cursor)
        //Set label: one-hot representation, location of the '1.0' is controlled by idx
        labels.putScalar(Array(i, idx, lastIdx - 1), 1.0)
      }
      labelsMask.putScalar(Array(i, lastIdx - 1), 1.0) //Specify that an output exists at the final time step for this example
      cursor = cursor + 1
      i = i + 1
    }
    if(this.labels.size > 0)
      new DataSet(features,labels,featuresMask,labelsMask)
    else
      new DataSet(features,null, featuresMask, labelsMask)
  }


  override def batch(): Int = batchSize

  override def totalExamples(): Int = datasetSize

  override def resetSupported(): Boolean = true

  override def inputColumns(): Int = ???

  override def getPreProcessor: DataSetPreProcessor = ???

  override def setPreProcessor(preProcessor: DataSetPreProcessor): Unit = ???

  override def getLabels: util.List[String] = (0 to classes).map(_.toString)

  override def totalOutcomes(): Int = ???

  override def reset(): Unit = cursor = 0

  override def asyncSupported(): Boolean = false

  override def numExamples(): Int = embeddings.size

  override def next(): DataSet = next(batchSize)

  override def hasNext: Boolean =  cursor < datasetSize

}
