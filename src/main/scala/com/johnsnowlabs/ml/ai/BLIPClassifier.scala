/*
 * Copyright 2017-2024 John Snow Labs
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

import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.annotators.cv.util.io.ImageIOUtils
import com.johnsnowlabs.nlp.annotators.cv.util.transform.ImageResizeUtils
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.BertTokenizer
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.WordpieceEncoder
import com.johnsnowlabs.nlp.{Annotation, AnnotationImage}
import org.tensorflow.ndarray.buffer.{IntDataBuffer, LongDataBuffer}

import scala.collection.JavaConverters._

private[johnsnowlabs] class BLIPClassifier(
    val tensorflowWrapper: TensorflowWrapper,
    configProtoBytes: Option[Array[Byte]] = None,
    tokenizer: BertTokenizer,
    preprocessor: Preprocessor,
    signatures: Option[Map[String, String]] = None,
    vocabulary: Map[String, Int])
    extends Serializable {

  private val _tfBLIPSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())

  def predict(
      images: Array[AnnotationImage],
      questions: Seq[Annotation],
      maxSentenceLength: Int,
      batchSize: Int): Seq[Annotation] = {

    val sentences = SentenceSplit.unpack(questions).toArray
    val tokenizedSentences = TokenizedWithSentence.unpack(questions).toArray
    val inputIds = encodeTokenizedSentence(
      tokenizedSentences,
      sentences,
      batchSize,
      maxSentenceLength,
      caseSensitive = false)

    val pixelValues = images
      .grouped(batchSize)
      .flatMap { batch =>
        encodeImage(batch, preprocessor)
      }
      .toArray

    val outputs = generate(pixelValues, inputIds, maxSentenceLength)
    val decodedOutput = tokenizer.decodeTokens(outputs)
    Seq(Annotation(decodedOutput))
  }

  def generate(
      imagesBatch: Array[Array[Array[Array[Float]]]],
      inputsBatch: Array[Array[Int]],
      maxSentenceLength: Int): Array[Int] = {
    val tensors = new TensorResources()
    val imageTensors = tensors.createTensor(imagesBatch)

    val batchLength = inputsBatch.length
    // [nb of encoded sentences , maxSentenceLength]
    val shape = Array(imagesBatch.length.toLong, maxSentenceLength)

    val tokenBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
    val maskBuffers: LongDataBuffer = tensors.createLongBuffer(batchLength * maxSentenceLength)

    inputsBatch.zipWithIndex
      .foreach { case (sentence, idx) =>
        val offset = idx * maxSentenceLength
        tokenBuffers.offset(offset).write(sentence)
        maskBuffers.offset(offset).write(sentence.map(x => if (x == 0) 0L else 1L))
      }

    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensors.createLongBufferTensor(shape, maskBuffers)

    val runner = tensorflowWrapper
      .getTFSessionWithSignature(configProtoBytes = configProtoBytes, initAllTables = false)
      .runner

    runner
      .feed(
        _tfBLIPSignatures
          .getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_ids"),
        tokenTensors)
      .feed(
        _tfBLIPSignatures
          .getOrElse(ModelSignatureConstants.AttentionMask.key, "missing_input_mask_key"),
        maskTensors)
      .feed(
        _tfBLIPSignatures
          .getOrElse(ModelSignatureConstants.PixelValuesInput.key, "missing_pixel_values"),
        imageTensors)
      .fetch(_tfBLIPSignatures
        .getOrElse(ModelSignatureConstants.DecoderOutput.key, "missing_output"))

    val outs = runner.run().asScala
    val output = TensorResources.extractInts(outs.head)

    tensors.clearSession(outs)
    tensors.clearTensors()
    imageTensors.close()

    output
  }

  /** Calculate softmax from returned logits
    * @param scores
    *   logits output from output layer
    * @return
    */
  def calculateSoftmax(scores: Array[Float]): Array[Float] = {
    val exp = scores.map(x => math.exp(x))
    exp.map(x => x / exp.sum).map(_.toFloat)
  }

  private def encodeImage(
      annotations: Array[AnnotationImage],
      preprocessor: Preprocessor): Array[Array[Array[Array[Float]]]] = {

    val batchProcessedImages = annotations.map { annot =>
      val bufferedImage = ImageIOUtils.byteToBufferedImage(
        bytes = annot.result,
        w = annot.width,
        h = annot.height,
        nChannels = annot.nChannels)

      val resizedImage = if (preprocessor.do_resize) {
        ImageResizeUtils.resizeBufferedImage(
          width = preprocessor.size,
          height = preprocessor.size,
          preprocessor.resample)(bufferedImage)
      } else bufferedImage

      val normalizedImage =
        ImageResizeUtils.normalizeAndConvertBufferedImage(
          img = resizedImage,
          mean = preprocessor.image_mean,
          std = preprocessor.image_std,
          doNormalize = preprocessor.do_normalize,
          doRescale = preprocessor.do_rescale,
          rescaleFactor = preprocessor.rescale_factor)

      normalizedImage
    }

    batchProcessedImages

  }

  def encodeTokenizedSentence(
      tokenizedSentences: Seq[TokenizedSentence],
      sentences: Seq[Sentence],
      batchSize: Int,
      maxSentenceLength: Int,
      caseSensitive: Boolean): Array[Array[Int]] = {
    val wordPieceTokenizedSentences =
      tokenizeWithAlignment(tokenizedSentences, maxSentenceLength, caseSensitive)

    /*Run calculation by batches*/
    wordPieceTokenizedSentences
      .zip(sentences)
      .zipWithIndex
      .grouped(batchSize)
      .flatMap { batch =>
        val tokensBatch = batch.map(x => (x._1._1, x._2))
        tokenizer.encode(tokensBatch, maxSentenceLength)
      }
      .toArray
  }

  def tokenizeWithAlignment(
      sentences: Seq[TokenizedSentence],
      maxSeqLength: Int,
      caseSensitive: Boolean): Seq[WordpieceTokenizedSentence] = {

    val encoder = new WordpieceEncoder(vocabulary)

    sentences.map { tokenIndex =>
      // filter empty and only whitespace tokens
      val bertTokens =
        tokenIndex.indexedTokens.filter(x => x.token.nonEmpty && !x.token.equals(" ")).map {
          token =>
            val content = if (caseSensitive) token.token else token.token.toLowerCase()
            val sentenceBegin = token.begin
            val sentenceEnd = token.end
            val sentenceIndex = tokenIndex.sentenceIndex
            val result =
              tokenizer.tokenize(Sentence(content, sentenceBegin, sentenceEnd, sentenceIndex))
            if (result.nonEmpty) result.head else IndexedToken("")
        }
      val wordpieceTokens = bertTokens.flatMap(token => encoder.encode(token)).take(maxSeqLength)
      WordpieceTokenizedSentence(wordpieceTokens)
    }
  }

}
