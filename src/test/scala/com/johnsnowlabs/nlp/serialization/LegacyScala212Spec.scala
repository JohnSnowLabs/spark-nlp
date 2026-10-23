/*
 * Copyright 2017-2026 John Snow Labs
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
package com.johnsnowlabs.nlp.serialization

import com.johnsnowlabs.collections.SearchTrie
import com.johnsnowlabs.nlp.annotators.{Tokenizer, TokenizerModel}
import com.johnsnowlabs.nlp.{DocumentAssembler, SparkAccessor}
import com.johnsnowlabs.tags.FastTest
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.scalatest.flatspec.AnyFlatSpec

import java.nio.file.Files
import scala.reflect.io.Directory

/** Reads data serialized by Scala 2.12 (Spark NLP 6.x). Fixtures in
  * src/test/resources/legacy-serialization/scala212 were written with Scala 2.12.15 from the
  * 6.2.2 sources:
  *   - tokenizer_custom_infix_rules.bin: `Array[RuleFactory]` of a Tokenizer whose infix patterns
  *     also split hyphenated words, as `fields/rules` stores it
  *   - wrapped_arrays.bin: `Array[AnyRef]` of WrappedArray ofRef, ofInt, ofLong, ofDouble,
  *     ofChar, ofBoolean
  *   - searchtrie_case_insensitive.bin: `Array[SearchTrie]` with a `caseSearch` lambda field
  */
class LegacyScala212Spec extends AnyFlatSpec {

  private val spark = SparkAccessor.spark
  import spark.implicits._

  private def resource(name: String): Array[Byte] = {
    val in = getClass.getClassLoader.getResourceAsStream(s"legacy-serialization/scala212/$name")
    try in.readAllBytes()
    finally in.close()
  }

  "A TokenizerModel stage saved by Scala 2.12" should "keep its uid, params and rules" taggedAs FastTest in {
    // metadata of stage 02_REGEX_TOKENIZER_68e3f8e08237 of clinical_deidentification_docwise_benchmark_medium_v2 6.2.2
    val metadata =
      """{"class":"com.johnsnowlabs.nlp.annotators.TokenizerModel","timestamp":1765730456704,"sparkVersion":"3.4.0","uid":"REGEX_TOKENIZER_68e3f8e08237","paramMap":{"targetPattern":"\\S+","maxLength":99999,"caseSensitiveExceptions":true,"minLength":0,"outputCol":"token","inputCols":["splitter"]},"defaultParamMap":{"targetPattern":"\\S+","maxLength":99999,"caseSensitiveExceptions":true,"lazyAnnotator":false,"minLength":0,"outputCol":"token","inputCols":["document"]}}"""
    val dir = Files.createTempDirectory("legacy-tokenizer-stage").toFile
    try {
      val stage = dir.getAbsolutePath + "/02_REGEX_TOKENIZER_68e3f8e08237"
      spark.sparkContext.parallelize(Seq(metadata), 1).saveAsTextFile(stage + "/metadata")
      spark.sparkContext
        .parallelize(Seq(resource("tokenizer_custom_infix_rules.bin")), 1)
        .map(bytes => (NullWritable.get(), new BytesWritable(bytes)))
        .saveAsSequenceFile(stage + "/fields/rules")

      val model = TokenizerModel.load(stage)
      assert(model.uid == "REGEX_TOKENIZER_68e3f8e08237")
      assert(model.getInputCols.sameElements(Array("splitter")))
      assert(model.getOutputCol == "token")

      // The saved rules split hyphenated words; the default rules (fallback load) would not
      val docs = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")
        .transform(Seq("This well-known fact, (state-of-the-art) works.").toDF("text"))
      def tokens(t: TokenizerModel) =
        t.setInputCols(Array("document"))
          .transform(docs)
          .selectExpr("token.result")
          .as[Seq[String]]
          .first()
      val contextChars = java.util.regex.Pattern.quote(".,;:!?*-()\"'")
      val expected = new Tokenizer()
        .setInputCols(Array("document"))
        .setOutputCol("token")
        .setInfixPatterns(
          Array("([^\\s\\-]+)(?:-)([^\\s\\-]+)", s"([^$contextChars](?:.*[^$contextChars])*)"))
        .fit(docs)
      assert(tokens(model).contains("well"))
      assert(tokens(model) == tokens(expected))
    } finally new Directory(dir).deleteRecursively()
  }

  "WrappedArrays serialized by Scala 2.12" should "be read as Scala 2.13 Seqs" taggedAs FastTest in {
    val values = LegacyObjectInputStream.deserializeArray[AnyRef](resource("wrapped_arrays.bin"))
    assert(values.toSeq == Seq(Seq("a", "b"), Seq(1, 2), Seq(3L), Seq(1.5), Seq('x'), Seq(true)))
  }

  "A SearchTrie serialized by Scala 2.12" should "keep its caseSearch lambda" taggedAs FastTest in {
    val tries =
      LegacyObjectInputStream.deserializeArray[SearchTrie](
        resource("searchtrie_case_insensitive.bin"))
    assert(tries(0).search(Seq("x", "A", "b")) == Seq((1, 1)))
  }
}
