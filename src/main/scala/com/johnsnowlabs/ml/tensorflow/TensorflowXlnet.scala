package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.ml.tensorflow.sentencepiece._
import com.johnsnowlabs.nlp.annotators.common._

import scala.collection.JavaConverters._
/** XlnetEmbeddings (XLNet): Generalized Autoregressive Pretraining for Language Understanding
  *
  * Note that this is a very computationally expensive module compared to word embedding modules that only perform embedding lookups.
  * The use of an accelerator is recommended.
  *
  * XLNet is a new unsupervised language representation learning method based on a novel generalized permutation language modeling objective. Additionally, XLNet employs Transformer-XL as the backbone model, exhibiting excellent performance for language tasks involving long context. Overall, XLNet achieves state-of-the-art (SOTA) results on various downstream language tasks including question answering, natural language inference, sentiment analysis, and document ranking.
  *
  * XLNet-Large     = [[https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip]]    | 24-layer, 1024-hidden, 16-heads
  * XLNet-Base    = [[https://storage.googleapis.com/xlnet/released_models/cased_L-12_H-768_A-12.zip]]   |  12-layer, 768-hidden, 12-heads. This model is trained on full data (different from the one in the paper).
  *
  * @param uid required internal uid for saving annotator
  *
  *            '''Sources :'''
  *
  *            [[ https://arxiv.org/abs/1906.08237]]
  *
  *            [[ https://github.com/zihangdai/xlnet]]
  *
  *            '''Paper abstract: '''
  *
  *            With the capability of modeling bidirectional contexts, denoising autoencoding based pretraining like BERT achieves better performance than pretraining approaches based on autoregressive language modeling. However, relying on corrupting the input with masks, BERT neglects dependency between the masked positions and suffers from a pretrain-finetune discrepancy. In light of these pros and cons, we propose XLNet, a generalized autoregressive pretraining method that (1) enables learning bidirectional contexts by maximizing the expected likelihood over all permutations of the factorization order and (2) overcomes the limitations of BERT thanks to its autoregressive formulation. Furthermore, XLNet integrates ideas from Transformer-XL, the state-of-the-art autoregressive model, into pretraining. Empirically, under comparable experiment settings, XLNet outperforms BERT on 20 tasks, often by a large margin, including question answering, natural language inference, sentiment analysis, and document ranking.
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
  */
class TensorflowXlnet(val tensorflow: TensorflowWrapper,
                      val spp: SentencePieceWrapper,
                      configProtoBytes: Option[Array[Byte]] = None
                     ) extends Serializable {

  // keys representing the input and output tensors of the XLNet model
  private val tokenIdsKey = "input_ids"
  private val maskIdsKey = "input_mask"
  private val segmentIdsKey = "segment_ids"
  private val outputSequenceKey = "module/seq_out"

  private val tokenSEPCLSIds = Array(4, 3)
  private val sentencePieceDelimiterId = 17

  def getSpecialTokens(token: String): Array[Int] = {
    spp.getSppModel.encodeAsIds(token)
  }

  def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] = {

    val tensors = new TensorResources()

    /* Actual size of each sentence to skip padding in the TF model */
    val sequencesLength = batch.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max

    val tokenBuffers = tensors.createIntBuffer(batch.length*maxSentenceLength)
    val maskBuffers = tensors.createFloatBuffer(batch.length*maxSentenceLength)
    val segmentBuffers = tensors.createIntBuffer(batch.length*maxSentenceLength)

    val shape = Array(batch.length.toLong, maxSentenceLength)

    batch.zipWithIndex.foreach { case(tokenIds, idx) =>
      val offset = idx * maxSentenceLength
      val diff = maxSentenceLength - tokenIds.length
      segmentBuffers.offset(offset).write(Array.fill(maxSentenceLength)(0))

      val padding = Array.fill(diff)(0)
      val newTokenIds = tokenIds ++ padding

      tokenBuffers.offset(offset).write(newTokenIds)
      maskBuffers.offset(offset).write(newTokenIds.map(x=> if (x == 0) 0f else 1f))
    }


    val tokenTensors = tensors.createIntBufferTensor(shape, tokenBuffers)
    val maskTensors = tensors.createFloatBufferTensor(shape, maskBuffers)
    val segmentTensors = tensors.createIntBufferTensor(shape, segmentBuffers)

    val runner = tensorflow.getTFHubSession(configProtoBytes = configProtoBytes).runner

    runner
      .feed(tokenIdsKey, tokenTensors)
      .feed(maskIdsKey, maskTensors)
      .feed(segmentIdsKey, segmentTensors)
      .fetch(outputSequenceKey)

    val outs = runner.run().asScala
    val embeddings = TensorResources.extractFloats(outs.head)

    tensors.clearSession(outs)
    tensors.clearTensors()
    tokenTensors.close()
    maskTensors.close()
    segmentTensors.close()

    val dim = embeddings.length / (batch.length * maxSentenceLength)
    val shrinkedEmbeddings: Array[Array[Array[Float]]] = embeddings.grouped(dim).toArray.grouped(maxSentenceLength).toArray

    val emptyVector = Array.fill(dim)(0f)

    batch.zip(shrinkedEmbeddings).map { case (ids, embeddings) =>
      if (ids.length > embeddings.length) {
        embeddings.take(embeddings.length - 1) ++
          Array.fill(embeddings.length - ids.length)(emptyVector) ++
          Array(embeddings.last)
      } else {
        embeddings
      }
    }

  }

  def calculateEmbeddings(sentences: Seq[TokenizedSentence],
                          batchSize: Int,
                          maxSentenceLength: Int,
                          caseSensitive: Boolean
                         ): Seq[WordpieceEmbeddingsSentence] = {

    sentences.grouped(batchSize).toArray.flatMap { batch =>

      val tokensPiece = tokenize(batch, maxSentenceLength, caseSensitive)
      val tokenIds = tokensPiece.map { sentence =>
        sentence.flatMap(x => x.tokens.find(_.pieceId != sentencePieceDelimiterId).map(x => x.pieceId)) ++ tokenSEPCLSIds
      }
      val vectors = tag(tokenIds)
      val tokenIdsVectors = tokenIds.zip(vectors).map { x =>
        x._1.zip(x._2).toMap
      }

      tokensPiece.zipWithIndex.zip(tokenIdsVectors).map { case (tokens, vectors) =>

        val tokensWithEmbeddings =  tokens._1.map{ token =>
          /* 17 is the id for '▁' token if appears alone */
          val subWord:TokenPiece = token.tokens.find(_.pieceId != sentencePieceDelimiterId).getOrElse(token.tokens.head)
          TokenPieceEmbeddings(
            subWord.wordpiece,
            subWord.token,
            subWord.pieceId,
            isWordStart = true,
            isOOV = false,
            vectors.apply(subWord.pieceId),
            subWord.begin,
            subWord.end
          )
        }
        WordpieceEmbeddingsSentence(tokensWithEmbeddings, tokens._2)
      }
    }

  }

  def tokenize(sentences: Seq[TokenizedSentence], maxSeqLength: Int, caseSensitive: Boolean):
  Seq[Array[WordpieceTokenizedSentence]] = {

    val sentenceTokenPieces = sentences.map { s =>
      // Account for one [SEP] & one [CLS]
      val shrinkedSentence = s.indexedTokens.take(maxSeqLength)
      shrinkedSentence.map{
        case(token) =>
          val tokenContent = if (caseSensitive) token.token else token.token.toLowerCase()
          val tokenPieces = spp.getSppModel.encodeAsPieces(tokenContent).toArray.map(x=>x.toString)
          val tokenIds = spp.getSppModel.encodeAsIds(tokenContent)
          WordpieceTokenizedSentence(
            tokenPieces.zip(tokenIds).map(x=> TokenPiece(x._1, token.token, x._2, isWordStart = false, token.begin, token.end))
          )
      }
    }
    sentenceTokenPieces
  }

}
