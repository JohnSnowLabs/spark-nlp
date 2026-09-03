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

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets
import java.nio.file.{Path, StandardOpenOption}
import scala.collection.mutable.ArrayBuffer
import scala.util.control.NonFatal

/** Minimal, bounded ELF parser used to verify native dependency identity before System.load. */
private[onnx] object ElfMetadataInspector extends NativeLibraryPreloader.ElfInspector {

  private val ElfHeaderSize = 64
  private val ProgramHeaderSize = 56
  private val MaxProgramHeaders = 4096
  private val MaxDynamicEntries = 65536
  private val MaxSonameBytes = 4096

  private val PtLoad = 1
  private val PtDynamic = 2
  private val DtNull = 0L
  private val DtStrtab = 5L
  private val DtStrsz = 10L
  private val DtSoname = 14L

  private final case class Segment(
      segmentType: Int,
      fileOffset: Long,
      virtualAddress: Long,
      fileSize: Long,
      memorySize: Long)

  private final case class FileMapping(fileOffset: Long, remainingFileBytes: Long)

  override def validate(path: Path, expectedSoname: String): Unit = {
    val channel = FileChannel.open(path, StandardOpenOption.READ)
    try validateOpenFile(channel, expectedSoname)
    catch {
      case error: IllegalStateException => throw error
      case NonFatal(error) =>
        throw new IllegalStateException(
          s"ELF validation failed for CUDA library $expectedSoname",
          error)
    } finally channel.close()
  }

  private def validateOpenFile(channel: FileChannel, expectedSoname: String): Unit = {
    val size = channel.size()
    val header = read(channel, 0L, ElfHeaderSize, size)
    if (header.get(0) != 0x7f.toByte || header.get(1) != 'E'.toByte ||
      header.get(2) != 'L'.toByte || header.get(3) != 'F'.toByte)
      fail(expectedSoname, "file is not ELF")
    if (header.get(4) != 2.toByte)
      fail(expectedSoname, "only ELF64 shared objects are supported")

    val byteOrder = header.get(5) match {
      case 1 => ByteOrder.LITTLE_ENDIAN
      case 2 => ByteOrder.BIG_ENDIAN
      case _ => fail(expectedSoname, "ELF byte order is invalid")
    }
    header.order(byteOrder)
    if (unsignedShort(header.getShort(16)) != 3)
      fail(expectedSoname, "ELF object is not a shared object")

    val machine = unsignedShort(header.getShort(18))
    val requiredMachine = expectedMachine(expectedSoname)
    if (machine != requiredMachine)
      fail(
        expectedSoname,
        s"ELF machine $machine does not match runtime architecture machine $requiredMachine")

    val programHeaderOffset = header.getLong(32)
    val programHeaderEntrySize = unsignedShort(header.getShort(54))
    val programHeaderCount = unsignedShort(header.getShort(56))
    if (programHeaderEntrySize < ProgramHeaderSize || programHeaderCount <= 0 ||
      programHeaderCount > MaxProgramHeaders)
      fail(expectedSoname, "ELF program-header table is invalid or exceeds bounds")

    val segments = ArrayBuffer.empty[Segment]
    var index = 0
    while (index < programHeaderCount) {
      val offset = checkedAdd(
        programHeaderOffset,
        checkedMultiply(index.toLong, programHeaderEntrySize.toLong, expectedSoname),
        expectedSoname)
      val entry = read(channel, offset, ProgramHeaderSize, size).order(byteOrder)
      val segment = Segment(
        entry.getInt(0),
        entry.getLong(8),
        entry.getLong(16),
        entry.getLong(32),
        entry.getLong(40))
      if (segment.fileOffset < 0 || segment.virtualAddress < 0 || segment.fileSize < 0 ||
        segment.memorySize < 0 || segment.fileSize > segment.memorySize)
        fail(expectedSoname, "ELF program segment contains invalid signed ranges")
      if (segment.segmentType == PtLoad || segment.segmentType == PtDynamic) {
        val segmentEnd = checkedAdd(segment.fileOffset, segment.fileSize, expectedSoname)
        if (segmentEnd > size)
          fail(expectedSoname, "ELF program segment exceeds the file-backed range")
        checkedAdd(segment.virtualAddress, segment.fileSize, expectedSoname)
        checkedAdd(segment.virtualAddress, segment.memorySize, expectedSoname)
      }
      segments += segment
      index += 1
    }

    val loads = segments.filter(_.segmentType == PtLoad)
    if (loads.isEmpty) fail(expectedSoname, "ELF PT_LOAD segment is missing")
    val dynamicSegments = segments.filter(_.segmentType == PtDynamic)
    if (dynamicSegments.size != 1)
      fail(expectedSoname, "ELF must contain exactly one PT_DYNAMIC segment")
    val dynamic = dynamicSegments.headOption.getOrElse {
      fail(expectedSoname, "ELF PT_DYNAMIC segment is missing")
    }
    if (dynamic.fileSize <= 0 || dynamic.fileSize % 16L != 0 ||
      dynamic.fileSize / 16L > MaxDynamicEntries)
      fail(expectedSoname, "ELF dynamic table is invalid or exceeds bounds")

    val dynamicMapping = virtualAddressToFileMapping(
      dynamic.virtualAddress,
      dynamic.fileSize,
      loads.toSeq,
      expectedSoname)
    if (dynamicMapping.fileOffset != dynamic.fileOffset ||
      dynamic.fileSize > dynamicMapping.remainingFileBytes)
      fail(expectedSoname, "ELF PT_DYNAMIC mapping does not match a unique PT_LOAD range")

    var stringTableAddress: Option[Long] = None
    var stringTableSize: Option[Long] = None
    var sonameIndex: Option[Long] = None
    var dynamicIndex = 0L
    var complete = false
    while (!complete && dynamicIndex < dynamic.fileSize / 16L) {
      val entryOffset = checkedAdd(
        dynamic.fileOffset,
        checkedMultiply(dynamicIndex, 16L, expectedSoname),
        expectedSoname)
      val entry = read(channel, entryOffset, 16, size).order(byteOrder)
      val tag = entry.getLong(0)
      val value = entry.getLong(8)
      tag match {
        case DtNull => complete = true
        case DtStrtab => stringTableAddress = Some(value)
        case DtStrsz => stringTableSize = Some(value)
        case DtSoname => sonameIndex = Some(value)
        case _ =>
      }
      dynamicIndex += 1
    }
    if (!complete) fail(expectedSoname, "ELF dynamic table has no bounded terminator")

    val stringAddress = stringTableAddress.getOrElse {
      fail(expectedSoname, "ELF DT_STRTAB entry is missing")
    }
    val nameIndex = sonameIndex.getOrElse {
      fail(expectedSoname, "ELF DT_SONAME entry is missing")
    }
    val tableSize = stringTableSize.getOrElse {
      fail(expectedSoname, "ELF DT_STRSZ entry is missing")
    }
    if (tableSize <= 0)
      fail(expectedSoname, "ELF DT_STRSZ value is invalid")
    if (nameIndex < 0 || nameIndex >= tableSize)
      fail(expectedSoname, "ELF DT_SONAME index is outside DT_STRTAB")

    val stringTableMapping =
      virtualAddressToFileMapping(stringAddress, tableSize, loads.toSeq, expectedSoname)
    if (tableSize > stringTableMapping.remainingFileBytes)
      fail(expectedSoname, "ELF DT_STRTAB exceeds its file-backed PT_LOAD range")
    val sonameOffset = checkedAdd(stringTableMapping.fileOffset, nameIndex, expectedSoname)
    val availableByTable = tableSize - nameIndex
    val availableBySegment = stringTableMapping.remainingFileBytes - nameIndex
    val availableByFile = size - sonameOffset
    val readLength = math.min(
      MaxSonameBytes.toLong,
      math.min(availableByTable, math.min(availableBySegment, availableByFile)))
    if (readLength <= 0) fail(expectedSoname, "ELF DT_SONAME value is unavailable")

    val nameBytes = read(channel, sonameOffset, readLength.toInt, size)
    val terminator = (0 until nameBytes.limit()).find(nameBytes.get(_) == 0.toByte).getOrElse {
      fail(expectedSoname, "ELF DT_SONAME is not null-terminated within bounds")
    }
    val bytes = new Array[Byte](terminator)
    nameBytes.position(0)
    nameBytes.get(bytes)
    val actualSoname = new String(bytes, StandardCharsets.UTF_8)
    if (actualSoname != expectedSoname)
      fail(
        expectedSoname,
        s"ELF DT_SONAME mismatch: expected $expectedSoname but found $actualSoname")
  }

  private def virtualAddressToFileMapping(
      address: Long,
      requiredBytes: Long,
      loads: Seq[Segment],
      expectedSoname: String): FileMapping = {
    if (address < 0 || requiredBytes <= 0)
      fail(expectedSoname, "ELF virtual-address range is invalid")
    val rangeEnd = checkedAdd(address, requiredBytes, expectedSoname)
    val overlappingLoads = loads.filter { segment =>
      val segmentMemoryEnd =
        checkedAdd(segment.virtualAddress, segment.memorySize, expectedSoname)
      segment.virtualAddress < rangeEnd && segmentMemoryEnd > address
    }
    if (overlappingLoads.size != 1)
      fail(expectedSoname, "ELF virtual-address range does not have a unique PT_LOAD mapping")

    val segment = overlappingLoads.head
    val segmentFileEnd =
      checkedAdd(segment.virtualAddress, segment.fileSize, expectedSoname)
    if (address < segment.virtualAddress || rangeEnd > segmentFileEnd)
      fail(expectedSoname, "ELF virtual-address range exceeds its file-backed PT_LOAD mapping")
    val relativeAddress = address - segment.virtualAddress
    FileMapping(
      checkedAdd(segment.fileOffset, relativeAddress, expectedSoname),
      segment.fileSize - relativeAddress)
  }

  private def read(
      channel: FileChannel,
      offset: Long,
      length: Int,
      fileSize: Long): ByteBuffer = {
    if (offset < 0 || length < 0 || offset > fileSize || length.toLong > fileSize - offset)
      throw new IllegalStateException("ELF structure points outside the file")
    val buffer = ByteBuffer.allocate(length)
    var position = offset
    while (buffer.hasRemaining) {
      val count = channel.read(buffer, position)
      if (count <= 0) throw new IllegalStateException("ELF file ended unexpectedly")
      position += count
    }
    buffer.flip()
    buffer
  }

  private def expectedMachine(expectedSoname: String): Int =
    System.getProperty("os.arch").toLowerCase match {
      case "amd64" | "x86_64" => 62
      case "aarch64" | "arm64" => 183
      case other => fail(expectedSoname, s"unsupported runtime architecture: $other")
    }

  private def unsignedShort(value: Short): Int = value & 0xffff

  private def checkedMultiply(left: Long, right: Long, expectedSoname: String): Long =
    try Math.multiplyExact(left, right)
    catch { case _: ArithmeticException => fail(expectedSoname, "ELF offset overflow") }

  private def checkedAdd(left: Long, right: Long, expectedSoname: String): Long =
    try Math.addExact(left, right)
    catch { case _: ArithmeticException => fail(expectedSoname, "ELF offset overflow") }

  private def fail(expectedSoname: String, detail: String): Nothing =
    throw new IllegalStateException(
      s"ELF validation failed for CUDA library $expectedSoname: $detail")
}
