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
package com.johnsnowlabs.ml.onnx

import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}

class ElfMetadataTestSpec extends AnyFlatSpec {

  "ELF metadata validation" should "accept only the required DT_SONAME" taggedAs FastTest in {
    val path = syntheticElf("libcudnn.so.9")
    try {
      NativeLibraryPreloader.ElfInspector.runtime.validate(path, "libcudnn.so.9")

      val error = intercept[IllegalStateException] {
        NativeLibraryPreloader.ElfInspector.runtime.validate(path, "libcudnn.so.90")
      }
      assert(error.getMessage.contains("DT_SONAME"))
    } finally Files.deleteIfExists(path)
  }

  it should "reject non-ELF input" taggedAs FastTest in {
    val path = secureTempFile()
    Files.write(path, "not an ELF shared object".getBytes(StandardCharsets.UTF_8))
    try {
      val error = intercept[IllegalStateException] {
        NativeLibraryPreloader.ElfInspector.runtime.validate(path, "libcudnn.so.9")
      }
      assert(error.getMessage.contains("ELF"))
    } finally Files.deleteIfExists(path)
  }

  it should "reject a SONAME string table without a bounded DT_STRSZ" taggedAs FastTest in {
    val path = syntheticElf("libcudnn.so.9", includeStringTableSize = false)
    try {
      val error = intercept[IllegalStateException] {
        NativeLibraryPreloader.ElfInspector.runtime.validate(path, "libcudnn.so.9")
      }
      assert(error.getMessage.contains("DT_STRSZ"))
    } finally Files.deleteIfExists(path)
  }

  it should "reject a SONAME extending beyond its file-backed PT_LOAD range" taggedAs FastTest in {
    val path = syntheticElf("libcudnn.so.9", truncateLoadBeforeSonameEnd = true)
    try {
      val error = intercept[IllegalStateException] {
        NativeLibraryPreloader.ElfInspector.runtime.validate(path, "libcudnn.so.9")
      }
      assert(error.getMessage.contains("PT_LOAD"))
    } finally Files.deleteIfExists(path)
  }

  it should "reject a PT_DYNAMIC mapping that disagrees with its PT_LOAD mapping" taggedAs FastTest in {
    val path = syntheticElf("libcudnn.so.9", inconsistentDynamicMapping = true)
    try {
      val error = intercept[IllegalStateException] {
        NativeLibraryPreloader.ElfInspector.runtime.validate(path, "libcudnn.so.9")
      }
      assert(error.getMessage.contains("PT_DYNAMIC"))
    } finally Files.deleteIfExists(path)
  }

  it should "reject ambiguously overlapping PT_LOAD mappings" taggedAs FastTest in {
    val path = syntheticElf("libcudnn.so.9", overlappingLoadMapping = true)
    try {
      val error = intercept[IllegalStateException] {
        ElfMetadataInspector.validate(path, "libcudnn.so.9")
      }
      assert(error.getMessage.contains("unique PT_LOAD mapping"))
    } finally Files.deleteIfExists(path)
  }

  it should "reject a PT_LOAD overlap beginning inside the dynamic string table" taggedAs FastTest in {
    val path = syntheticElf("libcudnn.so.9", partialStringTableOverlap = true)
    try {
      val error = intercept[IllegalStateException] {
        ElfMetadataInspector.validate(path, "libcudnn.so.9")
      }
      assert(error.getMessage.contains("unique PT_LOAD mapping"))
    } finally Files.deleteIfExists(path)
  }

  it should "reject overflowing virtual PT_LOAD ranges" taggedAs FastTest in {
    val path = syntheticElf("libcudnn.so.9", overflowingLoadRange = true)
    try {
      val error = intercept[IllegalStateException] {
        NativeLibraryPreloader.ElfInspector.runtime.validate(path, "libcudnn.so.9")
      }
      assert(error.getMessage.contains("overflow"))
    } finally Files.deleteIfExists(path)
  }

  private def syntheticElf(
      soname: String,
      includeStringTableSize: Boolean = true,
      truncateLoadBeforeSonameEnd: Boolean = false,
      inconsistentDynamicMapping: Boolean = false,
      overlappingLoadMapping: Boolean = false,
      partialStringTableOverlap: Boolean = false,
      overflowingLoadRange: Boolean = false): Path = {
    val dynamicOffset = 256
    val stringOffset = 336
    val stringBytes = ("\u0000" + soname + "\u0000").getBytes(StandardCharsets.UTF_8)
    val totalSize = stringOffset + stringBytes.length
    val baseAddress = 0x400000L
    val buffer = ByteBuffer.allocate(totalSize).order(ByteOrder.LITTLE_ENDIAN)

    buffer.put(0x7f.toByte).put('E'.toByte).put('L'.toByte).put('F'.toByte)
    buffer.put(2.toByte).put(1.toByte).put(1.toByte)
    while (buffer.position() < 16) buffer.put(0.toByte)
    buffer.putShort(3.toShort)
    buffer.putShort(expectedMachine)
    buffer.putInt(1)
    buffer.putLong(0L)
    buffer.putLong(64L)
    buffer.putLong(0L)
    buffer.putInt(0)
    buffer.putShort(64.toShort)
    buffer.putShort(56.toShort)
    val extraLoadCount =
      Seq(overlappingLoadMapping, partialStringTableOverlap, overflowingLoadRange).count(identity)
    buffer.putShort((2 + extraLoadCount).toShort)
    buffer.putShort(0.toShort).putShort(0.toShort).putShort(0.toShort)

    buffer.position(64)
    val loadFileSize = if (truncateLoadBeforeSonameEnd) stringOffset + 2 else totalSize
    putProgramHeader(buffer, 1, 0, baseAddress, loadFileSize, 0x1000L)
    val dynamicSize = if (includeStringTableSize) 64 else 48
    val dynamicAddress = baseAddress + dynamicOffset + (if (inconsistentDynamicMapping) 16 else 0)
    putProgramHeader(buffer, 2, dynamicOffset, dynamicAddress, dynamicSize, 8L)
    if (overlappingLoadMapping)
      putProgramHeader(buffer, 1, 16, baseAddress, totalSize - 16, 0x1000L)
    if (partialStringTableOverlap)
      putProgramHeader(
        buffer,
        1,
        0,
        baseAddress + stringOffset + 2L,
        stringBytes.length - 2,
        0x1000L)
    if (overflowingLoadRange)
      putProgramHeader(buffer, 1, 0, Long.MaxValue - 8L, totalSize, 0x1000L)

    buffer.position(dynamicOffset)
    buffer.putLong(5L).putLong(baseAddress + stringOffset)
    if (includeStringTableSize) buffer.putLong(10L).putLong(stringBytes.length.toLong)
    buffer.putLong(14L).putLong(1L)
    buffer.putLong(0L).putLong(0L)
    buffer.position(stringOffset)
    buffer.put(stringBytes)

    val path = secureTempFile()
    Files.write(path, buffer.array())
    path
  }

  private def putProgramHeader(
      buffer: ByteBuffer,
      segmentType: Int,
      fileOffset: Int,
      virtualAddress: Long,
      size: Int,
      alignment: Long): Unit = {
    buffer.putInt(segmentType)
    buffer.putInt(4)
    buffer.putLong(fileOffset.toLong)
    buffer.putLong(virtualAddress)
    buffer.putLong(virtualAddress)
    buffer.putLong(size.toLong)
    buffer.putLong(size.toLong)
    buffer.putLong(alignment)
  }

  private def expectedMachine: Short =
    System.getProperty("os.arch").toLowerCase match {
      case "amd64" | "x86_64" => 62.toShort
      case "aarch64" | "arm64" => 183.toShort
      case other => throw new IllegalStateException(s"Unsupported test architecture: $other")
    }

  private def secureTempFile(): Path = {
    val root = Paths.get(System.getProperty("user.home"), ".cache", "spark-nlp-tests")
    Files.createDirectories(root)
    Files.createTempFile(root, "synthetic-elf-", ".so")
  }
}
