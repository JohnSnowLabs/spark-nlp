/*
 * Copyright 2017-2025 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.similarity

import com.johnsnowlabs.nlp.AnnotatorType.{SENTENCE_EMBEDDINGS, VECTOR_SIMILARITY}
import com.johnsnowlabs.nlp.{
  Annotation,
  AnnotatorModel,
  AnnotatorType,
  HasSimpleAnnotate,
  ParamsAndFeaturesReadable,
  ParamsAndFeaturesWritable
}
import org.apache.spark.ml.param.{Param, ParamValidators}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.StructType

/** Computes pairwise vector similarity between two sets of sentence embeddings.
  *
  * The annotator takes **two** `SENTENCE_EMBEDDINGS` input columns (e.g. query embeddings and
  * document embeddings already joined on the same row) and, for every row, scores all N×M pairs
  * between the embeddings in column A and the embeddings in column B. Each pair produces one
  * `VECTOR_SIMILARITY` output annotation whose `result` holds the score as a String, and whose
  * `metadata` holds typed fields for easy extraction.
  *
  * ==Sign conventions==
  * {{{
  * method        range         higher means
  * ----------    ----------    ------------
  * cosine        [-1.0, 1.0]   more similar
  * dotProduct    (−∞, +∞)      more similar
  * euclidean     (−∞, 0.0]     more similar (0.0 = identical vectors)
  * }}}
  * The `euclidean` method returns the **negative** L2 distance so that "higher is better" holds
  * uniformly across all three methods. A score of `0.0` means the two vectors are identical; more
  * negative values indicate less similarity.
  *
  * ==Important: input data shape==
  * Each input column should contain **exactly one** embedding per row for standard document
  * retrieval. If a column contains N > 1 embeddings (e.g. produced by `SentenceDetector` +
  * embedder), all N×M cross-pairs are scored and returned as separate annotations. To compare
  * queries against a corpus, join them first:
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.nlp.annotators.similarity.PairwiseVectorSimilarity
  * import org.apache.spark.sql.functions.{col, desc, explode}
  *
  * // Assume embeddingPipeline produces a "embeddings" column of SENTENCE_EMBEDDINGS.
  * val queryDf  = embeddingPipeline.transform(queries).select(col("embeddings").as("query_emb"))
  * val corpusDf = embeddingPipeline.transform(corpus).select(col("embeddings").as("doc_emb"), col("id"))
  *
  * // CrossJoin to get one (query, document) pair per row, then score.
  * val paired = queryDf.crossJoin(corpusDf)
  *
  * val pvs = new PairwiseVectorSimilarity()
  *   .setInputCols("query_emb", "doc_emb")
  *   .setOutputCol("similarity")
  *   .setSimilarityMethod("cosine")
  *
  * pvs.transform(paired)
  *   .select(explode(col("similarity")).as("s"))
  *   .select(
  *     col("s.metadata")("sentence_a_text").as("query"),
  *     col("s.metadata")("sentence_b_text").as("document"),
  *     col("s.result").cast("double").as("score"))
  *   .orderBy(desc("score"))
  *   .show(false)
  * }}}
  *
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio setParam  4
  * @groupprio getParam  5
  */
class PairwiseVectorSimilarity(override val uid: String)
    extends AnnotatorModel[PairwiseVectorSimilarity]
    with HasSimpleAnnotate[PairwiseVectorSimilarity]
    with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("PairwiseVectorSimilarity"))

  /** Input annotator types: two SENTENCE_EMBEDDINGS columns (e.g. queries and documents).
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] =
    Array(SENTENCE_EMBEDDINGS, SENTENCE_EMBEDDINGS)

  /** Output annotator type: VECTOR_SIMILARITY
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = VECTOR_SIMILARITY

  /** Similarity function to use when scoring pairs.
    *
    * Supported values:
    *   - `"cosine"` (default) — cosine similarity in `[-1.0, 1.0]`; higher = more similar.
    *   - `"dotProduct"` — dot-product score in `(−∞, +∞)`; higher = more similar.
    *   - `"euclidean"` — **negative** L2 distance in `(−∞, 0.0]`; 0.0 = identical, more negative
    *     \= less similar. The sign is negated so that "higher is better" holds uniformly across
    *     all three methods.
    *
    * @group param
    */
  val similarityMethod: Param[String] = new Param[String](
    this,
    "similarityMethod",
    """Similarity function: "cosine" (default), "dotProduct", or "euclidean" """ +
      """(negative L2 distance — 0.0 = identical, more negative = less similar).""",
    ParamValidators.inArray(Array("cosine", "dotProduct", "euclidean")))

  /** @group setParam */
  def setSimilarityMethod(value: String): this.type = set(similarityMethod, value)

  /** @group getParam */
  def getSimilarityMethod: String = $(similarityMethod)

  setDefault(
    inputCols -> Array(SENTENCE_EMBEDDINGS, SENTENCE_EMBEDDINGS),
    outputCol -> VECTOR_SIMILARITY,
    similarityMethod -> "cosine")

  /** Verifies that both named input columns exist in the schema with the correct annotator type.
    *
    * The base `RawAnnotator.validate` uses `exists` (any column of the right type satisfies each
    * entry in `inputAnnotatorTypes`), which would pass validation even when only one
    * SENTENCE_EMBEDDINGS column is present. This override checks each column by name.
    */
  override protected def validate(schema: StructType): Boolean =
    getInputCols.forall { colName =>
      schema.exists(f =>
        f.name == colName &&
          f.metadata.contains("annotatorType") &&
          f.metadata.getString("annotatorType") == SENTENCE_EMBEDDINGS)
    }

  /** Scores all N×M pairs between the two input annotation sets.
    *
    * Each pair `(i, j)` produces one output annotation with:
    *   - `result` — score as a full-precision String
    *   - `metadata("sentence_a_idx")` / `metadata("sentence_b_idx")` — 0-based pair indices
    *   - `metadata("sentence_a_text")` / `metadata("sentence_b_text")` — source `.result` text
    *   - `metadata("similarity")` — score (same as `result`, for named access)
    *   - `metadata("similarityMethod")` — the method used
    */
  private def computeSimilarities(
      setA: Seq[Annotation],
      setB: Seq[Annotation]): Seq[Annotation] = {
    val method = $(similarityMethod)
    for {
      (annA, i) <- setA.zipWithIndex
      (annB, j) <- setB.zipWithIndex
    } yield {
      val vecA = annA.embeddings.map(_.toDouble)
      val vecB = annB.embeddings.map(_.toDouble)
      validateSameDimension(vecA, vecB, i, j)
      val score = method match {
        case "cosine" => cosineSimilarity(vecA, vecB)
        case "dotProduct" => dotProduct(vecA, vecB)
        case "euclidean" => negativeEuclidean(vecA, vecB)
        case other => throw new IllegalArgumentException(s"Unknown method: $other")
      }
      Annotation(
        annotatorType = VECTOR_SIMILARITY,
        begin = 0,
        end = 0,
        result = score.toString,
        metadata = Map(
          "sentence_a_idx" -> i.toString,
          "sentence_b_idx" -> j.toString,
          "sentence_a_text" -> annA.result,
          "sentence_b_text" -> annB.result,
          "similarity" -> score.toString,
          "similarityMethod" -> method),
        embeddings = Array.emptyFloatArray)
    }
  }

  private def validateSameDimension(
      a: Array[Double],
      b: Array[Double],
      indexA: Int,
      indexB: Int): Unit =
    require(
      a.length == b.length,
      "PairwiseVectorSimilarity requires equal embedding dimensions for each pair. " +
        s"Found ${a.length} and ${b.length} dimensions at sentence_a_idx=$indexA, " +
        s"sentence_b_idx=$indexB.")

  private def dotProduct(a: Array[Double], b: Array[Double]): Double = {
    var sum = 0.0
    var k = 0
    while (k < a.length) { sum += a(k) * b(k); k += 1 }
    sum
  }

  private def cosineSimilarity(a: Array[Double], b: Array[Double]): Double = {
    val dot = dotProduct(a, b)
    val normA = math.sqrt(dotProduct(a, a))
    val normB = math.sqrt(dotProduct(b, b))
    if (normA == 0.0 || normB == 0.0) 0.0 else dot / (normA * normB)
  }

  private def negativeEuclidean(a: Array[Double], b: Array[Double]): Double = {
    var sum = 0.0
    var k = 0
    while (k < a.length) { val d = a(k) - b(k); sum += d * d; k += 1 }
    -math.sqrt(sum)
  }

  /** Overrides the default `dfAnnotate` so the two input columns are kept separate rather than
    * being flattened into a single Seq. The first element of `rows` corresponds to col A (index 0
    * in `getInputCols`) and the second to col B (index 1).
    */
  override def dfAnnotate: UserDefinedFunction = udf { rows: Seq[Seq[Row]] =>
    val setA = rows.headOption.map(_.map(Annotation(_))).getOrElse(Seq.empty)
    val setB = rows.lift(1).map(_.map(Annotation(_))).getOrElse(Seq.empty)
    computeSimilarities(setA, setB)
  }

  /** Not supported — `LightPipeline` flattens both input columns into a single Seq, making it
    * impossible to distinguish col A from col B. Use `transform()` on a DataFrame instead.
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] =
    throw new UnsupportedOperationException(
      "PairwiseVectorSimilarity requires two named input columns. " +
        "LightPipeline is not supported — use transform() on a DataFrame instead.")
}

trait ReadablePairwiseVectorSimilarity extends ParamsAndFeaturesReadable[PairwiseVectorSimilarity]

object PairwiseVectorSimilarity extends ReadablePairwiseVectorSimilarity
