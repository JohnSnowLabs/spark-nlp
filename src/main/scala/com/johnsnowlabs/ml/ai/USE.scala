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

package com.johnsnowlabs.ml.ai

import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

import scala.collection.JavaConverters._

/** The Universal Sentence Encoder encodes text into high dimensional vectors that can be used for
  * text classification, semantic similarity, clustering and other natural language tasks.
  *
  * See
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/UniversalSentenceEncoderTestSpec.scala]]
  * for further reference on how to use this API.
  *
  * @param tensorflow
  *   USE Model wrapper with TensorFlow Wrapper
  * @param configProtoBytes
  *   Configuration for TensorFlow session
  *
  * Sources :
  *
  * [[https://arxiv.org/abs/1803.11175]]
  *
  * [[https://tfhub.dev/google/universal-sentence-encoder/2]]
  */
class USE(
    val tensorflow: TensorflowWrapper,
    configProtoBytes: Option[Array[Byte]] = None,
    loadSP: Boolean = false)
    extends Serializable {

  private val inputKey = "input"
  private val outPutKey = "output"

  private def sessionWarmup(): Unit = {
    val content = "Let's warmup the TF Session for the first inference."
    val dummyInput = Sentence(content, 0, content.length, 1, None)
    predict(Seq(dummyInput), 1)
  }

  sessionWarmup()

  def predict(sentences: Seq[Sentence], batchSize: Int): Seq[Annotation] = {

    sentences
      .grouped(batchSize)
      .flatMap { batch =>
        val tensors = new TensorResources()
        val batchSize = batch.length

        val sentencesContent = batch.map { x =>
          x.content
        }.toArray

        val sentenceTensors = tensors.createTensor(sentencesContent)

        val runner = tensorflow
          .getTFSessionWithSignature(configProtoBytes = configProtoBytes, loadSP = loadSP)
          .runner

        runner
          .feed(inputKey, sentenceTensors)
          .fetch(outPutKey)

        val outs = runner.run().asScala
        val allEmbeddings = TensorResources.extractFloats(outs.head)

        tensors.clearSession(outs)
        tensors.clearTensors()
        sentenceTensors.close()

        val dim = allEmbeddings.length / batchSize
        val embeddings = allEmbeddings.grouped(dim).toArray

        batch.zip(embeddings).map { case (sentence, vectors) =>
          Annotation(
            annotatorType = AnnotatorType.SENTENCE_EMBEDDINGS,
            begin = sentence.start,
            end = sentence.end,
            result = sentence.content,
            metadata = Map(
              "sentence" -> sentence.index.toString,
              "token" -> sentence.content,
              "pieceId" -> "-1",
              "isWordStart" -> "true"),
            embeddings = vectors)
        }
      }
  }.toSeq

}
