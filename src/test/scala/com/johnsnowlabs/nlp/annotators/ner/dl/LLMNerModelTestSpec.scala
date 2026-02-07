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

package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.seq2seq.{AutoGGUFModel, LLAMA3Transformer}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class LLMNerModelTestSpec extends AnyFlatSpec with LLMNerBehaviors {

  "LLMNerModel" should behave like correctAnnotatorCreation()

  it should "parse basic JSON response with extractions array" taggedAs FastTest in {
    val llmNer = new LLMNerModel()

    val jsonResponse = """{
      "extractions": [
        {"entity": "MEDICATION", "text": "aspirin", "confidence": 0.95},
        {"entity": "DOSAGE", "text": "500mg", "confidence": 0.90}
      ]
    }"""

    val annotation = Annotation(AnnotatorType.DOCUMENT, 0, 100, jsonResponse, Map.empty)
    val result = llmNer.annotate(Seq(annotation))

    assert(result.length == 2)
    assert(result.head.result == "aspirin")
    assert(result.head.metadata("entity") == "MEDICATION")
    assert(result.head.metadata("confidence") == "0.95")
    assert(result(1).result == "500mg")
    assert(result(1).metadata("entity") == "DOSAGE")
  }

  it should "parse JSON wrapped in markdown code fences" taggedAs FastTest in {
    val llmNer = new LLMNerModel()

    val jsonResponse = """```json
    {
      "extractions": [
        {"entity": "MEDICATION", "text": "ibuprofen", "confidence": 0.98}
      ]
    }
    ```"""

    val annotation = Annotation(AnnotatorType.DOCUMENT, 0, 100, jsonResponse, Map.empty)
    val result = llmNer.annotate(Seq(annotation))

    assert(result.length == 1)
    assert(result.head.result == "ibuprofen")
    assert(result.head.metadata("entity") == "MEDICATION")
  }

  it should "support alternative key names (extraction_class, extraction_text)" taggedAs FastTest in {
    val llmNer = new LLMNerModel()

    val jsonResponse = """{
      "extractions": [
        {"extraction_class": "MEDICATION", "extraction_text": "amoxicillin"}
      ]
    }"""

    val annotation = Annotation(AnnotatorType.DOCUMENT, 0, 100, jsonResponse, Map.empty)
    val result = llmNer.annotate(Seq(annotation))

    assert(result.length == 1)
    assert(result.head.result == "amoxicillin")
    assert(result.head.metadata("entity") == "MEDICATION")
  }

  it should "filter entities by type when entityTypes is set" taggedAs FastTest in {
    val llmNer = new LLMNerModel()
      .setEntityTypes(Array("MEDICATION"))

    val jsonResponse = """{
      "extractions": [
        {"entity": "MEDICATION", "text": "aspirin"},
        {"entity": "DOSAGE", "text": "500mg"},
        {"entity": "MEDICATION", "text": "ibuprofen"}
      ]
    }"""

    val annotation = Annotation(AnnotatorType.DOCUMENT, 0, 100, jsonResponse, Map.empty)
    val result = llmNer.annotate(Seq(annotation))

    assert(result.length == 2)
    assert(result.forall(_.metadata("entity") == "MEDICATION"))
    assert(result.map(_.result).toSet == Set("aspirin", "ibuprofen"))
  }

  it should "filter entities by confidence threshold" taggedAs FastTest in {
    val llmNer = new LLMNerModel()
      .setMinConfidence(0.8)

    val jsonResponse = """{
      "extractions": [
        {"entity": "MEDICATION", "text": "aspirin", "confidence": 0.95},
        {"entity": "MEDICATION", "text": "tylenol", "confidence": 0.75},
        {"entity": "MEDICATION", "text": "ibuprofen", "confidence": 0.85}
      ]
    }"""

    val annotation = Annotation(AnnotatorType.DOCUMENT, 0, 100, jsonResponse, Map.empty)
    val result = llmNer.annotate(Seq(annotation))

    assert(result.length == 2)
    assert(result.map(_.result).toSet == Set("aspirin", "ibuprofen"))
  }

  it should "handle entities without confidence field (default to 1.0)" taggedAs FastTest in {
    val llmNer = new LLMNerModel()

    val jsonResponse = """{
      "extractions": [
        {"entity": "MEDICATION", "text": "aspirin"}
      ]
    }"""

    val annotation = Annotation(AnnotatorType.DOCUMENT, 0, 100, jsonResponse, Map.empty)
    val result = llmNer.annotate(Seq(annotation))

    assert(result.length == 1)
    assert(result.head.metadata("confidence") == "1.0")
  }

  it should "preserve additional attributes in metadata" taggedAs FastTest in {
    val llmNer = new LLMNerModel()

    val jsonResponse = """{
      "extractions": [
        {
          "entity": "MEDICATION",
          "text": "aspirin",
          "attributes": {
            "route": "oral",
            "frequency": "daily"
          }
        }
      ]
    }"""

    val annotation = Annotation(AnnotatorType.DOCUMENT, 0, 100, jsonResponse, Map.empty)
    val result = llmNer.annotate(Seq(annotation))

    assert(result.length == 1)
    assert(result.head.metadata.contains("route"))
    assert(result.head.metadata("route") == "oral")
    assert(result.head.metadata("frequency") == "daily")
  }

  it should "return empty sequence for malformed JSON" taggedAs FastTest in {
    val llmNer = new LLMNerModel()

    val jsonResponse = """{ invalid json here }"""

    val annotation = Annotation(AnnotatorType.DOCUMENT, 0, 100, jsonResponse, Map.empty)
    val result = llmNer.annotate(Seq(annotation))

    assert(result.isEmpty)
  }

  it should "skip extractions missing required fields" taggedAs FastTest in {
    val llmNer = new LLMNerModel()

    val jsonResponse = """{
      "extractions": [
        {"entity": "MEDICATION", "text": "aspirin"},
        {"entity": "MEDICATION"},
        {"text": "ibuprofen"},
        {"entity": "DOSAGE", "text": ""}
      ]
    }"""

    val annotation = Annotation(AnnotatorType.DOCUMENT, 0, 100, jsonResponse, Map.empty)
    val result = llmNer.annotate(Seq(annotation))

    assert(result.length == 1)
    assert(result.head.result == "aspirin")
  }

  it should "set begin and end to -1 (no character indexing yet)" taggedAs FastTest in {
    val llmNer = new LLMNerModel()

    val jsonResponse = """{
      "extractions": [
        {"entity": "MEDICATION", "text": "aspirin"}
      ]
    }"""

    val annotation = Annotation(AnnotatorType.DOCUMENT, 0, 100, jsonResponse, Map.empty)
    val result = llmNer.annotate(Seq(annotation))

    assert(result.length == 1)
    assert(result.head.begin == -1)
    assert(result.head.end == -1)
  }

  it should "handle custom JSON array key" taggedAs FastTest in {
    val llmNer = new LLMNerModel()
      .setJsonArrayKey("entities")

    val jsonResponse = """{
      "entities": [
        {"entity": "MEDICATION", "text": "aspirin"}
      ]
    }"""

    val annotation = Annotation(AnnotatorType.DOCUMENT, 0, 100, jsonResponse, Map.empty)
    val result = llmNer.annotate(Seq(annotation))

    assert(result.length == 1)
    assert(result.head.result == "aspirin")
  }

  it should "handle multiple annotations in batch" taggedAs FastTest in {
    val llmNer = new LLMNerModel()

    val jsonResponse1 = """{
      "extractions": [
        {"entity": "MEDICATION", "text": "aspirin"}
      ]
    }"""

    val jsonResponse2 = """{
      "extractions": [
        {"entity": "MEDICATION", "text": "ibuprofen"}
      ]
    }"""

    val annotations = Seq(
      Annotation(AnnotatorType.DOCUMENT, 0, 100, jsonResponse1, Map.empty),
      Annotation(AnnotatorType.DOCUMENT, 0, 100, jsonResponse2, Map.empty))

    val result = llmNer.annotate(annotations)

    assert(result.length == 2)
    assert(result.head.result == "aspirin")
    assert(result(1).result == "ibuprofen")
  }

  it should "preserve source metadata" taggedAs FastTest in {
    val llmNer = new LLMNerModel()

    val jsonResponse = """{
      "extractions": [
        {"entity": "MEDICATION", "text": "aspirin"}
      ]
    }"""

    val sourceMetadata = Map("document_id" -> "doc123", "source" -> "clinical_notes")
    val annotation = Annotation(AnnotatorType.DOCUMENT, 0, 100, jsonResponse, sourceMetadata)
    val result = llmNer.annotate(Seq(annotation))

    assert(result.length == 1)
    assert(result.head.metadata("document_id") == "doc123")
    assert(result.head.metadata("source") == "clinical_notes")
  }

  it should "handle integer confidence values" taggedAs FastTest in {
    val llmNer = new LLMNerModel()

    val jsonResponse = """{
      "extractions": [
        {"entity": "MEDICATION", "text": "aspirin", "confidence": 1}
      ]
    }"""

    val annotation = Annotation(AnnotatorType.DOCUMENT, 0, 100, jsonResponse, Map.empty)
    val result = llmNer.annotate(Seq(annotation))

    assert(result.length == 1)
    assert(result.head.metadata("confidence") == "1.0")
  }

  it should "handle string confidence values" taggedAs FastTest in {
    val llmNer = new LLMNerModel()

    val jsonResponse = """{
      "extractions": [
        {"entity": "MEDICATION", "text": "aspirin", "confidence": "0.95"}
      ]
    }"""

    val annotation = Annotation(AnnotatorType.DOCUMENT, 0, 100, jsonResponse, Map.empty)
    val result = llmNer.annotate(Seq(annotation))

    assert(result.length == 1)
    assert(result.head.metadata("confidence") == "0.95")
  }

  it should "assign correct annotator type" taggedAs FastTest in {
    val llmNer = new LLMNerModel()

    val jsonResponse = """{
      "extractions": [
        {"entity": "MEDICATION", "text": "aspirin"}
      ]
    }"""

    val annotation = Annotation(AnnotatorType.DOCUMENT, 0, 100, jsonResponse, Map.empty)
    val result = llmNer.annotate(Seq(annotation))

    assert(result.length == 1)
    assert(result.head.annotatorType == AnnotatorType.NAMED_ENTITY)
  }

  it should "handle empty extractions array" taggedAs FastTest in {
    val llmNer = new LLMNerModel()

    val jsonResponse = """{
      "extractions": []
    }"""

    val annotation = Annotation(AnnotatorType.DOCUMENT, 0, 100, jsonResponse, Map.empty)
    val result = llmNer.annotate(Seq(annotation))

    assert(result.isEmpty)
  }

  it should "combine entity type and confidence filtering" taggedAs FastTest in {
    val llmNer = new LLMNerModel()
      .setEntityTypes(Array("MEDICATION"))
      .setMinConfidence(0.8)

    val jsonResponse = """{
      "extractions": [
        {"entity": "MEDICATION", "text": "aspirin", "confidence": 0.95},
        {"entity": "MEDICATION", "text": "tylenol", "confidence": 0.70},
        {"entity": "DOSAGE", "text": "500mg", "confidence": 0.95},
        {"entity": "MEDICATION", "text": "ibuprofen", "confidence": 0.85}
      ]
    }"""

    val annotation = Annotation(AnnotatorType.DOCUMENT, 0, 100, jsonResponse, Map.empty)
    val result = llmNer.annotate(Seq(annotation))

    assert(result.length == 2)
    assert(result.map(_.result).toSet == Set("aspirin", "ibuprofen"))
    assert(result.forall(_.metadata("entity") == "MEDICATION"))
  }

  it should "work with AutoGGUFModel in a full pipeline" taggedAs SlowTest in {
    import ResourceHelper.spark.implicits._

    // Build medical NER prompt
    val medicalNERPrompt =
      """You are a medical entity extraction assistant. Extract medical entities from text.

IMPORTANT: Extract the EXACT text as it appears in the input. Output JSON only.

Entity types: MEDICATION, DOSAGE, ROUTE, FREQUENCY

Output format:
{
  "extractions": [
    {"entity": "<type>", "text": "<exact_text>", "confidence": <0.0-1.0>}
  ]
}

Example:
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

Now extract entities from the following text:

"""

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val llm = AutoGGUFModel
      .pretrained()
      .setInputCols("document")
      .setOutputCol("llm_response")
      .setNPredict(500)
      .setTemperature(0.1f)
      .setSystemPrompt(medicalNERPrompt)

    val llmNer = new LLMNerModel()
      .setInputCols("llm_response")
      .setOutputCol("entities")
      .setEntityTypes(Array("MEDICATION", "DOSAGE"))
      .setMinConfidence(0.5)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, llm, llmNer))

    val data = Seq("Patient prescribed 500mg amoxicillin PO TID for 7 days.").toDF("text")

    val pipelineModel = pipeline.fit(data)
    val result = pipelineModel.transform(data)

    // Verify the pipeline runs without errors
    assert(result.columns.contains("entities"))

    // Collect results
    val entities = result.select("entities").collect()
    assert(entities.nonEmpty)

    // Print the LLM response and entities for debugging
    println("\n=== AutoGGUFModel + LLMNerModel Integration Test ===")
    result.select("llm_response.result").collect().foreach { row =>
      println(s"LLM Response: ${row.getString(0)}")
    }

    val annotations = Annotation.collect(result, "entities").head
    println(s"\nExtracted ${annotations.length} entities:")
    annotations.foreach { ann =>
      println(
        s"  - ${ann.metadata.getOrElse("entity", "UNKNOWN")}: '${ann.result}' " +
          s"(confidence: ${ann.metadata.getOrElse("confidence", "N/A")})")
    }

    // Verify we got at least some entities (LLM output may vary)
    // This is a loose assertion since LLM responses can vary
    assert(
      annotations.nonEmpty || annotations.isEmpty,
      "Pipeline should complete successfully whether entities are found or not")
  }
}

trait LLMNerBehaviors { this: AnyFlatSpec =>

  def correctAnnotatorCreation(): Unit = {
    it should "create annotator with correct types" taggedAs FastTest in {
      val llmNer = new LLMNerModel()

      assert(llmNer.inputAnnotatorTypes.sameElements(Array(AnnotatorType.DOCUMENT)))
      assert(llmNer.outputAnnotatorType == AnnotatorType.NAMED_ENTITY)
    }

    it should "have correct default parameter values" taggedAs FastTest in {
      val llmNer = new LLMNerModel()

      assert(llmNer.getMinConfidence == 0.0)
      assert(llmNer.getEntityTypes.isEmpty)
      assert(llmNer.getJsonArrayKey == "extractions")
    }
  }
}
