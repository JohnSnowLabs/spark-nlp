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

package com.johnsnowlabs.nlp.examples

import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotators.seq2seq.AutoGGUFModel
import com.johnsnowlabs.nlp.annotators.ner.dl.LLMNerModel
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession

/** Example demonstrating LLM-based Named Entity Recognition with AutoGGUFModel and LLMNerModel.
  *
  * This follows the LangExtract pattern from Google Research:
  *   1. Provide few-shot examples in the prompt
  *   2. LLM extracts entities as JSON (without character positions)
  *   3. Post-processing parses JSON and creates entity annotations
  *
  * Future versions will add fuzzy matching to compute character indices.
  */
object LLMNerExample extends App {

  // Initialize Spark
  val spark: SparkSession = SparkSession
    .builder()
    .appName("LLM NER Example")
    .master("local[*]")
    .config("spark.driver.memory", "16G")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.kryoserializer.buffer.max", "2000M")
    .getOrCreate()

  import spark.implicits._

  println("=" * 80)
  println("LLM-based Medical NER Example")
  println("=" * 80)

  // Example 1: Basic Medical NER
  example1_BasicMedicalNER()

  // Example 2: With entity filtering
  example2_WithFiltering()

  // Example 3: Custom prompt for different domains
  example3_CustomDomain()

  spark.stop()

  /** Example 1: Basic Medical NER with few-shot prompting */
  def example1_BasicMedicalNER(): Unit = {
    println("\n--- Example 1: Basic Medical NER ---\n")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    // AutoGGUFModel with medical NER prompt
    val llm = AutoGGUFModel
      .pretrained()
      .setInputCols("document")
      .setOutputCol("llm_response")
      .setNPredict(500)
      .setTemperature(0.1f) // Low temperature for more deterministic output
      .setSystemPrompt(buildMedicalNERPrompt())

    // LLMNerModel to parse JSON and extract entities
    val llmNer = new LLMNerModel()
      .setInputCols("llm_response")
      .setOutputCol("entities")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, llm, llmNer))

    val data = Seq(
      "Patient prescribed 500mg amoxicillin PO TID for 7 days.",
      "Administer morphine 10mg IV q4h PRN for pain control.",
      "Continue aspirin 81mg daily and metformin 1000mg BID.").toDF("text")

    val result = pipeline.fit(data).transform(data)

    println("Results:")
    result.select("text", "entities.result", "entities.metadata").show(truncate = false)
  }

  /** Example 2: With entity type filtering and confidence thresholds */
  def example2_WithFiltering(): Unit = {
    println("\n--- Example 2: With Filtering ---\n")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val llm = AutoGGUFModel
      .pretrained()
      .setInputCols("document")
      .setOutputCol("llm_response")
      .setNPredict(500)
      .setTemperature(0.1f)
      .setSystemPrompt(buildMedicalNERPrompt())

    // Only extract MEDICATION and DOSAGE entities with confidence >= 0.7
    val llmNer = new LLMNerModel()
      .setInputCols("llm_response")
      .setOutputCol("entities")
      .setEntityTypes(Array("MEDICATION", "DOSAGE"))
      .setMinConfidence(0.7)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, llm, llmNer))

    val data = Seq("Patient prescribed 500mg amoxicillin PO TID for infection.").toDF("text")

    val result = pipeline.fit(data).transform(data)

    println("Filtered Results (MEDICATION & DOSAGE only, confidence >= 0.7):")
    result.select("entities.result", "entities.metadata").show(truncate = false)
  }

  /** Example 3: Custom domain (non-medical) */
  def example3_CustomDomain(): Unit = {
    println("\n--- Example 3: Custom Domain (Legal Entities) ---\n")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val llm = AutoGGUFModel
      .pretrained()
      .setInputCols("document")
      .setOutputCol("llm_response")
      .setNPredict(500)
      .setTemperature(0.1f)
      .setSystemPrompt(buildLegalNERPrompt())

    val llmNer = new LLMNerModel()
      .setInputCols("llm_response")
      .setOutputCol("entities")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, llm, llmNer))

    val data = Seq(
      "The plaintiff, John Doe, filed a complaint against Acme Corp on January 15, 2024.").toDF(
      "text")

    val result = pipeline.fit(data).transform(data)

    println("Legal Entity Results:")
    result.select("entities.result", "entities.metadata").show(truncate = false)
  }

  /** Build medical NER prompt with few-shot examples
    *
    * This follows the LangExtract pattern: - Emphasize extracting EXACT text from input - Provide
    * 1-3 high-quality examples - Request JSON output format
    */
  def buildMedicalNERPrompt(): String = {
    """You are a medical entity extraction assistant. Your task is to identify and extract medical entities from clinical text.

IMPORTANT: Extract the EXACT text as it appears in the input. Do not paraphrase or modify the text.

Entity types to extract:
- MEDICATION: Drug names (e.g., aspirin, amoxicillin, morphine)
- DOSAGE: Dose amounts (e.g., 500mg, 10mg, 81mg)
- ROUTE: Administration route (e.g., PO, IV, topical)
- FREQUENCY: Dosing frequency (e.g., BID, TID, q4h, daily)

Output format: JSON with an "extractions" array. Each extraction should have:
{
  "entity": "<entity_type>",
  "text": "<exact_text_from_input>",
  "confidence": <0.0_to_1.0>
}

Example 1:
Input: "Patient takes aspirin 81mg daily."
Output:
```json
{
  "extractions": [
    {"entity": "MEDICATION", "text": "aspirin", "confidence": 0.95},
    {"entity": "DOSAGE", "text": "81mg", "confidence": 0.98},
    {"entity": "FREQUENCY", "text": "daily", "confidence": 0.99}
  ]
}
```

Example 2:
Input: "Administer morphine 10mg IV q4h for pain."
Output:
```json
{
  "extractions": [
    {"entity": "MEDICATION", "text": "morphine", "confidence": 0.98},
    {"entity": "DOSAGE", "text": "10mg", "confidence": 0.97},
    {"entity": "ROUTE", "text": "IV", "confidence": 0.99},
    {"entity": "FREQUENCY", "text": "q4h", "confidence": 0.96}
  ]
}
```

Now extract entities from the following text. Remember to extract EXACT text as it appears:

"""
  }

  /** Build legal NER prompt for a different domain */
  def buildLegalNERPrompt(): String = {
    """You are a legal entity extraction assistant. Extract legal entities from text.

Entity types:
- PERSON: Names of individuals
- ORGANIZATION: Company names, institutions
- DATE: Specific dates
- COURT: Court names
- CASE_NUMBER: Legal case numbers

Output format: JSON with "extractions" array.

Example:
Input: "John Smith filed suit against ABC Corp in District Court."
Output:
```json
{
  "extractions": [
    {"entity": "PERSON", "text": "John Smith", "confidence": 0.95},
    {"entity": "ORGANIZATION", "text": "ABC Corp", "confidence": 0.98},
    {"entity": "COURT", "text": "District Court", "confidence": 0.92}
  ]
}
```

Extract entities from:

"""
  }
}

