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

package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT, SENTENCE_EMBEDDINGS, WORD_EMBEDDINGS}
import com.johnsnowlabs.nlp.annotators.common.WordpieceEmbeddingsSentence
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType, HasSimpleAnnotate}
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}

/** Produces contextual chunk-level embeddings using the Late Chunking technique described in
  * [[https://arxiv.org/abs/2409.04701 Jin et al. (2024)]].
  *
  * Unlike [[ChunkEmbeddings]], which embeds each chunk in isolation, `LateChunkEmbeddings`
  * expects that the upstream token-embedding stage (e.g. `ModernBertEmbeddings` or
  * `LongformerEmbeddings`) has already processed the **full document** in a single forward pass,
  * producing contextual token representations. This annotator then locates the tokens that fall
  * within each chunk's character span and mean-pools them into a single `SENTENCE_EMBEDDINGS`
  * vector — so every chunk embedding is informed by the complete document context rather than
  * being isolated.
  *
  * ==Ordering requirement==
  * `LateChunkEmbeddings` '''must''' appear '''after''' the token-embedding stage in the pipeline.
  * Placing it before the embedding stage will raise a runtime error.
  *
  * ==Context-window limitation==
  * The contextual benefit is bounded by the upstream model's maximum sequence length (e.g. 8 192
  * tokens for `ModernBertEmbeddings`). Documents that exceed this limit are truncated before
  * embedding, reducing cross-chunk context for tokens near the end of very long documents.
  *
  * ==Warning: upstream model must process the full document==
  * The upstream embedding stage '''must''' use `DOCUMENT` (not `SENTENCE`) as its input
  * annotation type, processing the entire document in a single forward pass. If a
  * `SentenceDetector` is placed before the embedding model, each sentence is embedded
  * independently and the contextual benefit of late chunking is lost — the annotator will still
  * run without error but will produce embeddings equivalent to naive `ChunkEmbeddings`.
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.{DocumentAssembler, Doc2Chunk}
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.embeddings.{ModernBertEmbeddings, LateChunkEmbeddings}
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols(Array("document"))
  *   .setOutputCol("token")
  *
  * val tokenEmbeddings = ModernBertEmbeddings.pretrained("modernbert-base", "en")
  *   .setInputCols("document", "token")
  *   .setOutputCol("token_embeddings")
  *   .setMaxSentenceLength(8192)
  *
  * val chunker = new Doc2Chunk()
  *   .setInputCols(Array("document"))
  *   .setChunkCol("chunks")
  *   .setIsArray(true)
  *   .setOutputCol("chunk")
  *
  * val lateChunkEmbeddings = new LateChunkEmbeddings()
  *   .setInputCols("document", "chunk", "token_embeddings")
  *   .setOutputCol("late_chunk_embeddings")
  *   .setPoolingStrategy("AVERAGE")
  *
  * val pipeline = new Pipeline()
  *   .setStages(Array(
  *     documentAssembler,
  *     tokenizer,
  *     tokenEmbeddings,
  *     chunker,
  *     lateChunkEmbeddings
  *   ))
  *
  * val data = Seq((
  *   "AcmeDrug was prescribed for migraine in March. The patient took two doses.\n\n" +
  *   "It caused severe nausea the next day, and therapy was stopped.",
  *   Array(
  *     "AcmeDrug was prescribed for migraine in March. The patient took two doses.",
  *     "It caused severe nausea the next day, and therapy was stopped.")
  * )).toDF("text", "chunks")
  *
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(late_chunk_embeddings) as r")
  *   .select("r.annotatorType", "r.result", "r.embeddings")
  *   .show(5, 80)
  * // +-------------------+--------------------------------------------------------------------------+--------------------------------------------------------------------------------+
  * // |      annotatorType|                                                                    result|                                                                      embeddings|
  * // +-------------------+--------------------------------------------------------------------------+--------------------------------------------------------------------------------+
  * // |sentence_embeddings|AcmeDrug was prescribed for migraine in March. The patient took two doses.|[0.050471008, -0.07595207, 0.031268876, 0.15105441, -0.013697156, 0.08131724,...|
  * // |sentence_embeddings|            It caused severe nausea the next day, and therapy was stopped.|[0.0735685, 0.0060829176, 0.12051964, 0.22399232, 0.055884164, 0.066795066, 0...|
  * // +-------------------+--------------------------------------------------------------------------+--------------------------------------------------------------------------------+
  * }}}
  *
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class LateChunkEmbeddings(override val uid: String)
    extends AnnotatorModel[LateChunkEmbeddings]
    with HasSimpleAnnotate[LateChunkEmbeddings]
    with HasEmbeddingsProperties
    with HasStorageRef {

  /** Output annotator type : SENTENCE_EMBEDDINGS
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = SENTENCE_EMBEDDINGS

  /** Input annotator types : DOCUMENT, CHUNK, WORD_EMBEDDINGS
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] =
    Array(DOCUMENT, CHUNK, WORD_EMBEDDINGS)

  /** Number of embedding dimensions. Inferred automatically from the upstream token embeddings on
    * the first annotation.
    *
    * @group param
    */
  override val dimension = new IntParam(this, "dimension", "Number of embedding dimensions")

  /** Number of embedding dimensions
    *
    * @group getParam
    */
  override def getDimension: Int = $(dimension)

  /** Strategy used to aggregate token embeddings within each chunk span. Either `"AVERAGE"`
    * (default) or `"SUM"`.
    *
    * @group param
    */
  val poolingStrategy = new Param[String](
    this,
    "poolingStrategy",
    "Strategy to aggregate token embeddings into a chunk embedding: AVERAGE or SUM")

  /** Whether to skip OOV (out-of-vocabulary) token embeddings when pooling (Default: `true`).
    * When true, default zero-vectors for OOV tokens are excluded from the pool.
    *
    * @group param
    */
  val skipOOV = new BooleanParam(
    this,
    "skipOOV",
    "Whether to discard default vectors for OOV words from the aggregation / pooling")

  /** Sets pooling strategy. Must be `"AVERAGE"` or `"SUM"`.
    *
    * @group setParam
    */
  def setPoolingStrategy(strategy: String): this.type = {
    strategy.toLowerCase() match {
      case "average" => set(poolingStrategy, "AVERAGE")
      case "sum" => set(poolingStrategy, "SUM")
      case _ => throw new MatchError("poolingStrategy must be either AVERAGE or SUM")
    }
  }

  /** Sets whether to skip OOV token embeddings during pooling.
    *
    * @group setParam
    */
  def setSkipOOV(value: Boolean): this.type = set(skipOOV, value)

  /** Returns the current pooling strategy.
    *
    * @group getParam
    */
  def getPoolingStrategy: String = $(poolingStrategy)

  /** Returns whether OOV token embeddings are skipped during pooling.
    *
    * @group getParam
    */
  def getSkipOOV: Boolean = $(skipOOV)

  setDefault(
    inputCols -> Array(DOCUMENT, CHUNK, WORD_EMBEDDINGS),
    outputCol -> "late_chunk_embeddings",
    poolingStrategy -> "AVERAGE",
    skipOOV -> true,
    dimension -> 0)

  /** Internal constructor to submit a random UID */
  def this() = this(Identifiable.randomUID("LATE_CHUNK_EMBEDDINGS"))

  /** Pools a matrix of token embeddings (rows = tokens, cols = dimensions) into a single vector
    * according to the configured pooling strategy.
    *
    * NOTE: `setDimension` is intentionally NOT called here. This method runs inside a Spark UDF
    * on an executor. In cluster mode, any param changes made on the executor are made to a
    * serialized copy of the annotator and do NOT propagate back to the driver. The dimension is
    * instead read from the upstream WORD_EMBEDDINGS column metadata in `beforeAnnotate`, which
    * runs on the driver before the UDF is built. This ensures that `afterAnnotate` — also
    * driver-side — sees the correct value when it writes metadata to the output column schema.
    *
    * Precondition: `matrix` must be non-empty (enforced by the caller).
    */
  private def calculateChunkEmbeddings(matrix: Array[Array[Float]]): Array[Float] = {
    require(matrix.nonEmpty, "calculateChunkEmbeddings called with empty matrix")
    val res = Array.ofDim[Float](matrix(0).length)
    matrix(0).indices.foreach { j =>
      matrix.indices.foreach { i =>
        res(j) += matrix(i)(j)
      }
      if ($(poolingStrategy) == "AVERAGE")
        res(j) /= matrix.length
    }
    res
  }

  /** Takes a document and annotations and produces new annotations of this annotator's annotation
    * type.
    *
    * The method:
    *   1. Collects all contextual token embeddings produced by the upstream model across the
    *      entire document (flattened across all internal sentence buckets). 2. For each `CHUNK`
    *      annotation, selects the token embeddings whose character offsets fall within the
    *      chunk's `[begin, end]` span. 3. Mean-pools the selected embeddings into a single
    *      vector. 4. Emits one `SENTENCE_EMBEDDINGS` annotation per chunk, preserving the chunk's
    *      text, offsets, and metadata.
    *
    * @param annotations
    *   Annotations from all configured input columns (DOCUMENT, CHUNK, WORD_EMBEDDINGS).
    * @return
    *   One `SENTENCE_EMBEDDINGS` annotation per chunk that has at least one overlapping token
    *   embedding. Chunks with no matching tokens are silently dropped.
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    // NOTE: DOCUMENT annotations are intentionally not consumed here. They are required in
    // inputAnnotatorTypes solely as a pipeline contract (ensuring a full-document source col
    // is present), not for any algorithmic purpose in this method.
    val chunkAnnotations = annotations.filter(_.annotatorType == CHUNK)

    // Unpack ALL token embeddings from the full document, flattening across sentence buckets.
    // This is the key difference from ChunkEmbeddings: we never restrict the search to a single
    // sentence, allowing every chunk to benefit from full-document contextual representations.
    val embeddingsSentences = WordpieceEmbeddingsSentence.unpack(annotations)
    val allTokenEmbeddings = embeddingsSentences.flatMap(_.tokens)

    chunkAnnotations.flatMap { chunk =>
      val sentenceIdx = chunk.metadata.getOrElse("sentence", "0")
      val chunkIdx = chunk.metadata.getOrElse("chunk", "0")

      // Select tokens whose character span falls inside the chunk span
      val tokensInSpan = allTokenEmbeddings.filter { token =>
        token.begin >= chunk.begin && token.end <= chunk.end
      }

      // Respect skipOOV: if all selected tokens are OOV and skipOOV=true, fall back to using
      // all of them (same fallback as ChunkEmbeddings) so we never return an empty embedding.
      val validEmbeddings = tokensInSpan.flatMap { t =>
        if (!t.isOOV || ! $(skipOOV)) Some(t.embeddings) else None
      }

      val finalEmbeddings =
        if (validEmbeddings.nonEmpty) validEmbeddings else tokensInSpan.map(_.embeddings)

      // If there are truly no tokens in this chunk span (e.g. model was truncated), emit nothing
      if (finalEmbeddings.isEmpty)
        None
      else
        Some(
          Annotation(
            annotatorType = SENTENCE_EMBEDDINGS,
            begin = chunk.begin,
            end = chunk.end,
            result = chunk.result,
            metadata = chunk.metadata ++ Map(
              "sentence" -> sentenceIdx,
              "chunk" -> chunkIdx,
              "token" -> chunk.result,
              "pieceId" -> "-1",
              "isWordStart" -> "true"),
            embeddings = calculateChunkEmbeddings(finalEmbeddings.toArray)))
    }
  }

  /** Reads the storage reference and embedding dimension from the upstream WORD_EMBEDDINGS column
    * metadata before any executor UDF runs.
    *
    * ===Why dimension is set here, not inside annotate()===
    * `AnnotatorModel._transform` calls `beforeAnnotate` eagerly on the '''driver''', then builds
    * a lazy `withColumn(...udf...)` plan, then calls `afterAnnotate` — also on the driver —
    * before any executor has executed the UDF. This means `setDimension()` called inside
    * `annotate()` (executor context) would happen '''after''' `afterAnnotate` has already baked
    * the dimension into the output column metadata. By reading the dimension from the Spark
    * schema field metadata here (set by the upstream model's own `afterAnnotate`), we guarantee
    * `$(dimension)` is correct when `afterAnnotate` calls `wrapSentenceEmbeddingsMetadata`.
    */
  override protected def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    val ref =
      HasStorageRef.getStorageRefFromInput(dataset, $(inputCols), AnnotatorType.WORD_EMBEDDINGS)
    if (get(storageRef).isEmpty)
      setStorageRef(ref)

    // Read the embedding dimension from the upstream WORD_EMBEDDINGS column's schema metadata.
    // Every transformer embedding model writes this via wrapEmbeddingsMetadata in its own
    // afterAnnotate (e.g. ModernBertEmbeddings, LongformerEmbeddings, WordEmbeddingsModel).
    if ($(dimension) == 0) {
      val embeddingsField =
        Annotation.getColumnByType(dataset, $(inputCols), WORD_EMBEDDINGS)
      if (embeddingsField.metadata.contains("dimension"))
        setDimension(embeddingsField.metadata.getLong("dimension").toInt)
    }

    dataset
  }

  /** Attaches Spark column metadata that marks the output as `SENTENCE_EMBEDDINGS` with the
    * correct dimension and storage reference — enabling downstream use with `EmbeddingsFinisher`
    * or sentence-level classifiers.
    */
  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    dataset.withColumn(
      getOutputCol,
      wrapSentenceEmbeddingsMetadata(
        dataset.col(getOutputCol),
        $(dimension),
        Some($(storageRef))))
  }
}

/** This is the companion object of [[LateChunkEmbeddings]]. Please refer to that class for the
  * documentation.
  */
object LateChunkEmbeddings extends DefaultParamsReadable[LateChunkEmbeddings]
