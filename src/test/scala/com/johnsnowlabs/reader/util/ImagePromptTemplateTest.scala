package com.johnsnowlabs.reader.util

import org.scalatest.flatspec.AnyFlatSpec

class ImagePromptTemplateTest extends AnyFlatSpec {

  "ImagePromptTemplate.customTemplate" should "replace {prompt} with actual prompt" in {
    val template = "Instruction: {prompt}. Done."
    val prompt = "Analyze the scene"
    val result = ImagePromptTemplate.customTemplate(template, prompt)

    assert(result == "Instruction: Analyze the scene. Done.")
  }

}
