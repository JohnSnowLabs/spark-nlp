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

import java.io.{ByteArrayOutputStream, File}
import java.nio.charset.StandardCharsets
import java.nio.file.attribute.{BasicFileAttributes, PosixFileAttributes, PosixFilePermission}
import java.nio.file._
import java.util.concurrent.TimeUnit
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.io.Source
import scala.util.control.NonFatal

/** Resolves and loads the native CUDA dependencies required by ONNX Runtime.
  *
  * Resolution is deliberately dormant until the ONNX CUDA provider reports a recognized missing
  * native dependency. All paths are validated before the first `System.load` call.
  */
private[onnx] final class NativeLibraryPreloader(
    requiredLibraries: Seq[String],
    sourceProvider: NativeLibraryPreloader.SearchBudget => NativeLibraryPreloader.SearchSources,
    limits: NativeLibraryPreloader.SearchLimits,
    libraryLoader: NativeLibraryPreloader.LibraryLoader,
    securityPolicy: NativeLibraryPreloader.SecurityPolicy =
      NativeLibraryPreloader.SecurityPolicy.runtime,
    elfInspector: NativeLibraryPreloader.ElfInspector =
      NativeLibraryPreloader.ElfInspector.runtime) {

  import NativeLibraryPreloader._

  private val logger = LoggerFactory.getLogger(classOf[NativeLibraryPreloader])
  private var loaded = false
  private var terminalFailure: Option[(String, Int)] = None

  def preload(config: PreloadConfig): Unit = this.synchronized {
    if (config.mode == Off) return
    if (loaded) return
    terminalFailure.foreach { case (failedLibrary, loadedLibraries) =>
      throw new IllegalStateException(
        s"ONNX CUDA native preload is terminal in this JVM after failing at $failedLibrary " +
          s"with $loadedLibraries earlier libraries loaded")
    }

    val resolved = config.mode match {
      case Explicit => resolveExplicit(config.paths)
      case Search => resolveSearch(config.paths)
      case Off => Seq.empty
    }

    // Resolution and validation intentionally finish before native process state is changed.
    var loadedLibraries = 0
    resolved.zip(requiredLibraries).foreach { case (path, library) =>
      try libraryLoader.load(path)
      catch {
        case error: UnsatisfiedLinkError =>
          terminalFailure = Some(library -> loadedLibraries)
          throw new IllegalStateException(
            s"Failed to preload $library from canonical path $path: ${safeMessage(error)}",
            error)
        case NonFatal(error) =>
          terminalFailure = Some(library -> loadedLibraries)
          throw new IllegalStateException(
            s"Failed to preload $library from canonical path $path: ${safeMessage(error)}",
            error)
      }
      loadedLibraries += 1
      logger.info(s"Preloaded ONNX CUDA native dependency $library from $path")
    }
    loaded = true
    logger.info(
      "ONNX CUDA native dependency preload completed for {} libraries",
      Int.box(resolved.size))
  }

  private def resolveExplicit(configuredPaths: Seq[String]): Seq[Path] = {
    if (configuredPaths.size != requiredLibraries.size) {
      throw new IllegalStateException(
        s"Explicit ONNX CUDA preload requires ${requiredLibraries.size} ordered absolute files, " +
          s"but received ${configuredPaths.size}")
    }

    val characterBudget = new CharacterBudget(MaxConfiguredPathChars)
    configuredPaths.zip(requiredLibraries).map { case (configuredPath, library) =>
      characterBudget.record(configuredPath)
      val path = Paths.get(configuredPath)
      if (!path.isAbsolute)
        throw new IllegalStateException(s"CUDA library path must be absolute for $library")
      validateCandidate(path, library, allowMatchingGroupWritable = true)
    }
  }

  private def resolveSearch(operatorPathStrings: Seq[String]): Seq[Path] = {
    val budget = new SearchBudget(limits.maxVisitedEntries)
    val characterBudget = new CharacterBudget(MaxConfiguredPathChars)
    val resolved = mutable.Map.empty[String, Path]
    val operatorDirectories = operatorPathStrings.map { configuredPath =>
      budget.recordVisit()
      characterBudget.record(configuredPath)
      val path = Paths.get(configuredPath)
      if (!path.isAbsolute)
        throw new IllegalStateException(
          s"ONNX CUDA operator search directory must be absolute: ${path.getFileName}")
      canonicalDirectory(path, "operator search")
    }

    def resolveTier(
        tierName: String,
        directories: Seq[Path],
        candidates: Map[String, Seq[Path]]): Option[Seq[Path]] = {
      requiredLibraries.filterNot(resolved.contains).foreach { library =>
        resolveFromCandidates(
          library,
          candidates.getOrElse(library, Seq.empty),
          tierName,
          directories).foreach(path => resolved += library -> path)
      }
      if (resolved.size == requiredLibraries.size)
        Some(requiredLibraries.map(resolved))
      else None
    }

    resolveTier(
      "operator",
      operatorDirectories,
      collectDirectoryCandidates(operatorDirectories, budget)).getOrElse {
      val sources = sourceProvider(budget)
      val runtimeDirectories = canonicalDirectories(sources.runtimeDirectories, "runtime", budget)
      resolveTier(
        "runtime",
        runtimeDirectories,
        collectDirectoryCandidates(runtimeDirectories, budget)).getOrElse {
        val linkerDirectories = canonicalDirectories(sources.linkerDirectories, "linker", budget)
        resolveTier(
          "linker",
          linkerDirectories,
          collectDirectoryCandidates(linkerDirectories, budget)).getOrElse {
          val genericRoots = canonicalDirectories(sources.genericRoots, "generic root", budget)
          resolveTier(
            "generic roots",
            genericRoots,
            collectGenericCandidates(genericRoots, budget)).getOrElse {
            val missing = requiredLibraries.filterNot(resolved.contains).head
            throw new IllegalStateException(
              s"Required CUDA library $missing was not found in approved ONNX CUDA search sources")
          }
        }
      }
    }
  }

  private def resolveFromCandidates(
      library: String,
      candidates: Seq[Path],
      tierName: String,
      approvedRoots: Seq[Path]): Option[Path] = {
    val exact = candidates.filter(_.getFileName.toString == library)
    if (exact.nonEmpty) uniqueCandidate(library, exact, tierName, approvedRoots)
    else uniqueCandidate(library, candidates, tierName, approvedRoots)
  }

  private def uniqueCandidate(
      library: String,
      candidates: Seq[Path],
      tierName: String,
      approvedRoots: Seq[Path]): Option[Path] = {
    val validated = candidates
      .map(validateCandidate(_, library, approvedRoots))
      .distinct
      .sortBy(_.toString)
    validated match {
      case Seq() => None
      case Seq(path) => Some(path)
      case multiple =>
        throw new IllegalStateException(
          s"Ambiguous CUDA library $library in $tierName: ${multiple.size} canonical candidates")
    }
  }

  private def collectDirectoryCandidates(
      directories: Seq[Path],
      budget: SearchBudget): Map[String, Seq[Path]] = {
    val found = mutable.Map(requiredLibraries.map(_ -> Array.newBuilder[Path]): _*)
    directories.sortBy(_.toString).foreach { directory =>
      val stream = Files.newDirectoryStream(directory)
      try {
        val iterator = stream.iterator()
        while (iterator.hasNext) {
          val path = iterator.next()
          budget.recordVisit()
          if (Files.isRegularFile(path)) {
            val fileName = path.getFileName.toString
            requiredLibraries.foreach { library =>
              if (isCompatibleName(fileName, library)) found(library) += path
            }
          }
        }
      } catch {
        case error: IllegalStateException => throw error
        case NonFatal(error) =>
          throw new IllegalStateException(
            s"Failed to enumerate approved ONNX CUDA search directory ${directory.getFileName}",
            error)
      } finally stream.close()
    }
    found.map { case (library, builder) => library -> builder.result().toSeq }.toMap
  }

  private def collectGenericCandidates(
      genericRoots: Seq[Path],
      budget: SearchBudget): Map[String, Seq[Path]] = {
    val found = mutable.Map(requiredLibraries.map(_ -> Array.newBuilder[Path]): _*)

    genericRoots.sortBy(_.toString).foreach { root =>
      Files.walkFileTree(
        root,
        java.util.EnumSet.noneOf(classOf[FileVisitOption]),
        limits.maxDepth,
        new SimpleFileVisitor[Path]() {
          override def preVisitDirectory(
              directory: Path,
              attributes: BasicFileAttributes): FileVisitResult = {
            budget.recordVisit()
            if ((directory != root && Files.isSymbolicLink(directory)) || !Files.isReadable(
                directory))
              FileVisitResult.SKIP_SUBTREE
            else FileVisitResult.CONTINUE
          }

          override def visitFile(file: Path, attributes: BasicFileAttributes): FileVisitResult = {
            budget.recordVisit()
            val isLoadableFile = attributes.isRegularFile ||
              (attributes.isSymbolicLink && Files.isRegularFile(file))
            if (isLoadableFile) {
              val fileName = file.getFileName.toString
              requiredLibraries.foreach { library =>
                if (isCompatibleName(fileName, library))
                  found(library) += file
              }
            }
            FileVisitResult.CONTINUE
          }

          override def visitFileFailed(
              file: Path,
              error: java.io.IOException): FileVisitResult = {
            budget.recordVisit()
            FileVisitResult.CONTINUE
          }
        })
    }

    found.map { case (library, builder) => library -> builder.result().toSeq }.toMap
  }

  private def canonicalDirectories(
      paths: Seq[Path],
      sourceName: String,
      budget: SearchBudget): Seq[Path] =
    paths.flatMap { path =>
      budget.recordVisit()
      if (!path.isAbsolute)
        throw new IllegalStateException(
          s"ONNX CUDA $sourceName search directory must be absolute: ${path.getFileName}")
      try Some(canonicalDirectory(path, sourceName))
      catch {
        case _: NoSuchFileException => None
        case _: AccessDeniedException => None
        case NonFatal(_) => None
      }
    }.distinct

  private def canonicalDirectory(path: Path, sourceName: String): Path = {
    val canonical = path.toRealPath()
    if (!Files.isDirectory(canonical) || !Files.isReadable(canonical))
      throw new IllegalStateException(s"Unreadable ONNX CUDA $sourceName directory")
    canonical
  }

  private def validateCandidate(
      path: Path,
      library: String,
      approvedRoots: Seq[Path] = Seq.empty,
      allowMatchingGroupWritable: Boolean = false): Path = {
    val suppliedName = Option(path.getFileName).map(_.toString).getOrElse("")
    if (!isCompatibleName(suppliedName, library))
      throw new IllegalStateException(
        s"CUDA library order mismatch: expected $library but received $suppliedName")

    val canonical =
      try path.toRealPath()
      catch {
        case NonFatal(error) =>
          throw new IllegalStateException(s"Required CUDA library $library is unavailable", error)
      }

    val canonicalName = canonical.getFileName.toString
    if (!isCompatibleName(canonicalName, library))
      throw new IllegalStateException(
        s"Canonical CUDA target for $library has an unexpected filename: $canonicalName")
    if (approvedRoots.nonEmpty && !approvedRoots.exists(canonical.startsWith))
      throw new IllegalStateException(
        s"Canonical CUDA target for $library resolves outside its approved search root")
    if (!Files.isRegularFile(canonical) || !Files.isReadable(canonical))
      throw new IllegalStateException(
        s"Required CUDA library $library is not a readable regular file")
    validateTrustedPath(canonical, library, allowMatchingGroupWritable)
    elfInspector.validate(canonical, library)
    canonical
  }

  private def validateTrustedPath(
      path: Path,
      library: String,
      allowMatchingGroupWritable: Boolean): Unit = {
    try {
      var current: Path = path
      var isLibrary = true
      while (current != null) {
        val attributes = securityPolicy.readAttributes(current)
        val owner = Option(attributes.owner()).map(_.getName).getOrElse("")
        if (!securityPolicy.trustedOwners.contains(owner))
          throw new IllegalStateException(
            s"Refusing CUDA library $library because ${current.getFileName} has an untrusted owner")

        val group = Option(attributes.group()).map(_.getName).getOrElse("")
        val permissions = attributes.permissions()
        if (permissions.contains(PosixFilePermission.OTHERS_WRITE)) {
          if (isLibrary)
            throw new IllegalStateException(
              s"Refusing world-writable CUDA library file for $library")
          else
            throw new IllegalStateException(
              s"Refusing CUDA library $library because it has a world-writable directory (writable ancestor)")
        }
        if (permissions.contains(PosixFilePermission.GROUP_WRITE)) {
          val isTrustedExecutorPair =
            isLibrary && allowMatchingGroupWritable &&
              securityPolicy.trustedGroupWritableOwnerGroups.contains(owner -> group)
          if (isLibrary && !isTrustedExecutorPair)
            throw new IllegalStateException(
              s"Refusing group-writable CUDA library file for $library")
          else if (!isLibrary)
            throw new IllegalStateException(
              s"Refusing CUDA library $library because it has a group-writable directory (writable ancestor)")
        }

        isLibrary = false
        current = current.getParent
      }
    } catch {
      case error: IllegalStateException => throw error
      case error: UnsupportedOperationException =>
        throw new IllegalStateException(
          s"POSIX ownership and permissions are required to validate CUDA library $library",
          error)
      case error: java.io.IOException =>
        throw new IllegalStateException(
          s"POSIX ownership and permissions are required to validate CUDA library $library",
          error)
      case NonFatal(error) =>
        throw new IllegalStateException(
          s"POSIX ownership and permissions are required to validate CUDA library $library",
          error)
    }
  }

  private def isCompatibleName(candidate: String, library: String): Boolean =
    candidate == library || candidate.matches(
      java.util.regex.Pattern.quote(library) + "(?:\\.[0-9]+)+")

  private def safeMessage(error: Throwable): String =
    Option(error.getMessage).getOrElse(error.getClass.getSimpleName).replaceAll("[\\r\\n]+", " ")
}

private[onnx] object NativeLibraryPreloader {

  sealed trait Mode
  case object Off extends Mode
  case object Search extends Mode
  case object Explicit extends Mode

  final case class PreloadConfig(mode: Mode, paths: Seq[String])

  object PreloadConfig {
    def parse(modeValue: String, pathValue: String): PreloadConfig = {
      val mode =
        Option(modeValue).map(_.trim.toLowerCase).filter(_.nonEmpty).getOrElse("search") match {
          case "off" => Off
          case "search" => Search
          case "explicit" => Explicit
          case unsupported =>
            throw new IllegalArgumentException(
              s"Unsupported ONNX CUDA preload mode '$unsupported'; expected one of: off, search, explicit")
        }
      if (mode == Off) return PreloadConfig(Off, Seq.empty)

      val paths = Option(pathValue)
        .filter(_.nonEmpty)
        .map { value =>
          if (value.length > MaxConfiguredPathChars)
            throw new IllegalArgumentException(
              s"ONNX CUDA preload paths character limit $MaxConfiguredPathChars was exceeded")
          val trimmed = value.trim
          if (trimmed.isEmpty) Seq.empty
          else {
            val parsed = trimmed.split(
              java.util.regex.Pattern.quote(File.pathSeparator),
              SearchLimits.default.maxVisitedEntries + 1)
            if (parsed.length > SearchLimits.default.maxVisitedEntries)
              throw new IllegalArgumentException(
                s"ONNX CUDA preload path count exceeds ${SearchLimits.default.maxVisitedEntries}")
            parsed.toSeq
          }
        }
        .getOrElse(Seq.empty)
      PreloadConfig(mode, paths)
    }
  }

  private[onnx] val MaxConfiguredPathChars = 1024 * 1024

  final case class SearchLimits(maxDepth: Int, maxVisitedEntries: Int)
  object SearchLimits {
    val default: SearchLimits = SearchLimits(maxDepth = 6, maxVisitedEntries = 20000)
  }

  final class SearchBudget(maxVisitedEntries: Int) {
    private var visitedEntries = 0

    def recordVisit(): Unit = {
      visitedEntries += 1
      if (visitedEntries > maxVisitedEntries)
        throw new IllegalStateException(
          s"ONNX CUDA bounded search entry limit $maxVisitedEntries was exceeded")
    }
  }

  final class ByteBudget(maxBytes: Long) {
    private var consumedBytes = 0L

    def recordByte(): Unit = {
      consumedBytes += 1L
      if (consumedBytes > maxBytes)
        throw new IllegalStateException(
          s"ONNX CUDA linker-config byte limit $maxBytes was exceeded")
    }
  }

  final class CharacterBudget(maxCharacters: Int) {
    private var consumedCharacters = 0L

    def record(value: String): Unit = {
      consumedCharacters += value.length.toLong
      if (consumedCharacters > maxCharacters.toLong)
        throw new IllegalStateException(
          s"ONNX CUDA configured-path character limit $maxCharacters was exceeded")
    }
  }

  final class SearchSources private (
      runtimeProvider: () => Seq[Path],
      linkerProvider: () => Seq[Path],
      genericProvider: () => Seq[Path]) {
    lazy val runtimeDirectories: Seq[Path] = runtimeProvider()
    lazy val linkerDirectories: Seq[Path] = linkerProvider()
    lazy val genericRoots: Seq[Path] = genericProvider()
  }

  object SearchSources {
    def apply(
        runtimeDirectories: Seq[Path],
        linkerDirectories: Seq[Path],
        genericRoots: Seq[Path]): SearchSources =
      new SearchSources(() => runtimeDirectories, () => linkerDirectories, () => genericRoots)

    def deferred(
        runtimeDirectories: => Seq[Path],
        linkerDirectories: => Seq[Path],
        genericRoots: => Seq[Path]): SearchSources =
      new SearchSources(() => runtimeDirectories, () => linkerDirectories, () => genericRoots)

    val empty: SearchSources = SearchSources(Seq.empty, Seq.empty, Seq.empty)

    def runtime(budget: SearchBudget, limits: SearchLimits): SearchSources = deferred(
      runtimeDirectories = runtimeDerivedDirectories(budget),
      linkerDirectories = linkerConfiguredDirectories(budget, limits),
      genericRoots = Seq(Paths.get("/opt"), Paths.get("/usr/local")))
  }

  trait LibraryLoader {
    def load(path: Path): Unit
  }

  trait ElfInspector {
    def validate(path: Path, expectedSoname: String): Unit
  }

  object ElfInspector {
    val accepting: ElfInspector = new ElfInspector {
      override def validate(path: Path, expectedSoname: String): Unit = ()
    }
    val runtime: ElfInspector = ElfMetadataInspector
  }

  final case class SecurityPolicy(
      trustedOwners: Set[String],
      readAttributes: Path => PosixFileAttributes,
      trustedGroupWritableOwnerGroups: Set[(String, String)] = Set.empty)

  object SecurityPolicy {
    lazy val runtime: SecurityPolicy = {
      val authenticatedIdentity = authenticatedProcessIdentity()
      SecurityPolicy(
        trustedOwners = Set(authenticatedIdentity._1, "root"),
        readAttributes = path => Files.readAttributes(path, classOf[PosixFileAttributes]),
        trustedGroupWritableOwnerGroups = Set(authenticatedIdentity))
    }
  }

  private[onnx] def authenticatedProcessOwner(): String = authenticatedProcessIdentity()._1

  private[onnx] def authenticatedProcessIdentity(): (String, String) = {
    val (owner, group) =
      try {
        val attributes =
          Files.readAttributes(Paths.get("/proc/self"), classOf[PosixFileAttributes])
        attributes.owner().getName -> attributes.group().getName
      } catch {
        case NonFatal(error) =>
          throw new IllegalStateException(
            "Unable to establish the authenticated POSIX executor identity",
            error)
      }
    val validatedOwner = Option(owner).map(_.trim).filter(_.nonEmpty).getOrElse {
      throw new IllegalStateException("Authenticated POSIX executor identity is empty")
    }
    val validatedGroup = Option(group).map(_.trim).filter(_.nonEmpty).getOrElse {
      throw new IllegalStateException("Authenticated POSIX executor group identity is empty")
    }
    validatedOwner -> validatedGroup
  }

  private val dependencyManifest = "/onnx/cuda-provider-dependencies-1.23.0.txt"

  lazy val packagedDependencies: Seq[String] = {
    val stream = Option(getClass.getResourceAsStream(dependencyManifest)).getOrElse {
      throw new IllegalStateException(
        s"Packaged ONNX CUDA dependency manifest is missing: $dependencyManifest")
    }
    val source = Source.fromInputStream(stream, "UTF-8")
    try {
      val libraries = source
        .getLines()
        .map(_.trim)
        .filter(line => line.nonEmpty && !line.startsWith("#"))
        .toVector
      if (libraries.isEmpty || libraries.distinct.size != libraries.size ||
        libraries.exists(!_.matches("lib[A-Za-z0-9]+\\.so(?:\\.[0-9]+)+"))) {
        throw new IllegalStateException(
          s"Packaged ONNX CUDA dependency manifest is invalid: $dependencyManifest")
      }
      libraries
    } finally source.close()
  }

  private lazy val executorPreloader = new NativeLibraryPreloader(
    packagedDependencies,
    budget => SearchSources.runtime(budget, SearchLimits.default),
    SearchLimits.default,
    new LibraryLoader {
      override def load(path: Path): Unit = System.load(path.toString)
    },
    SecurityPolicy.runtime,
    ElfInspector.runtime)

  def preload(config: PreloadConfig): Unit = executorPreloader.preload(config)

  private def runtimeDerivedDirectories(budget: SearchBudget): Seq[Path] = {
    val characterBudget = new CharacterBudget(MaxConfiguredPathChars)
    val directPathLists = parseRuntimePathLists(
      Seq(
        sys.env.get("LD_LIBRARY_PATH"),
        Option(System.getProperty("java.library.path"))).flatten,
      budget,
      characterBudget)

    val cudaRoots = Seq(sys.env.get("CUDA_HOME"), sys.env.get("CUDA_PATH")).flatten
      .flatMap { root =>
        characterBudget.record(root)
        budget.recordVisit()
        cudaRootDirectories(Paths.get(root), budget)
      }
    val condaDirectories = sys.env
      .get("CONDA_PREFIX")
      .map { root =>
        characterBudget.record(root)
        budget.recordVisit()
        Paths.get(root, "lib")
      }
      .toSeq

    (directPathLists ++ cudaRoots ++ condaDirectories).filter(_.isAbsolute).distinct
  }

  private[onnx] def parseRuntimePathLists(
      pathLists: Seq[String],
      limits: SearchLimits,
      maxCharacters: Int): Seq[Path] =
    parseRuntimePathLists(
      pathLists,
      new SearchBudget(limits.maxVisitedEntries),
      new CharacterBudget(maxCharacters))

  private def parseRuntimePathLists(
      pathLists: Seq[String],
      budget: SearchBudget,
      characterBudget: CharacterBudget): Seq[Path] =
    pathLists.flatMap { pathList =>
      characterBudget.record(pathList)
      val entries = pathList.split(
        java.util.regex.Pattern.quote(File.pathSeparator),
        SearchLimits.default.maxVisitedEntries + 1)
      if (entries.length > SearchLimits.default.maxVisitedEntries)
        throw new IllegalStateException(
          s"ONNX CUDA runtime path count exceeds ${SearchLimits.default.maxVisitedEntries}")
      entries.iterator
        .map(_.trim)
        .filter(_.nonEmpty)
        .map { entry =>
          budget.recordVisit()
          Paths.get(entry)
        }
        .toSeq
    }

  private val MaxLdConfigBytes = 1024 * 1024L

  private def linkerConfiguredDirectories(
      budget: SearchBudget,
      limits: SearchLimits): Seq[Path] = {
    val configured = parseLdConfig(
      Paths.get("/etc/ld.so.conf"),
      Set.empty,
      budget,
      new ByteBudget(MaxLdConfigBytes),
      depth = 0,
      limits.maxDepth)
    (configured ++ ldconfigDirectories(budget)).filter(_.isAbsolute).distinct
  }

  private[onnx] def linkerConfigDirectories(
      rootConfig: Path,
      maxBytes: Long,
      limits: SearchLimits): Seq[Path] =
    parseLdConfig(
      rootConfig,
      Set.empty,
      new SearchBudget(limits.maxVisitedEntries),
      new ByteBudget(maxBytes),
      depth = 0,
      limits.maxDepth)

  private def parseLdConfig(
      path: Path,
      visited: Set[Path],
      budget: SearchBudget,
      byteBudget: ByteBudget,
      depth: Int,
      maxDepth: Int): Seq[Path] = {
    if (depth > maxDepth)
      throw new IllegalStateException(
        s"ONNX CUDA linker-config depth limit $maxDepth was exceeded")
    if (!Files.isRegularFile(path) || !Files.isReadable(path)) return Seq.empty
    val canonical =
      try path.toRealPath()
      catch { case NonFatal(_) => return Seq.empty }
    if (visited.contains(canonical)) return Seq.empty

    readBoundedConfigLines(canonical, byteBudget).flatMap { rawLine =>
      budget.recordVisit()
      val line = rawLine.takeWhile(_ != '#').trim
      if (line.isEmpty) Seq.empty
      else if (line.startsWith("include ")) {
        expandGlob(line.stripPrefix("include ").trim, budget)
          .flatMap(parseLdConfig(_, visited + canonical, budget, byteBudget, depth + 1, maxDepth))
      } else {
        val directory = Paths.get(line)
        if (directory.isAbsolute) Seq(directory) else Seq.empty
      }
    }
  }

  private def readBoundedConfigLines(path: Path, byteBudget: ByteBudget): Seq[String] = {
    val input = Files.newInputStream(path)
    try {
      val lines = Vector.newBuilder[String]
      val currentLine = new ByteArrayOutputStream()

      def appendCurrentLine(): Unit = {
        val bytes = currentLine.toByteArray
        val length =
          if (bytes.nonEmpty && bytes.last == '\r'.toByte) bytes.length - 1
          else bytes.length
        lines += new String(bytes, 0, length, StandardCharsets.UTF_8)
        currentLine.reset()
      }

      var nextByte = input.read()
      while (nextByte >= 0) {
        byteBudget.recordByte()
        if (nextByte == '\n'.toInt) appendCurrentLine()
        else currentLine.write(nextByte)
        nextByte = input.read()
      }
      if (currentLine.size() > 0) appendCurrentLine()
      lines.result()
    } finally input.close()
  }

  private def expandGlob(pattern: String, budget: SearchBudget): Seq[Path] = {
    val path = Paths.get(pattern)
    val parent = Option(path.getParent).getOrElse(Paths.get("."))
    if (!parent.isAbsolute || !Files.isDirectory(parent)) return Seq.empty
    val stream = Files.newDirectoryStream(parent, path.getFileName.toString)
    try
      stream
        .iterator()
        .asScala
        .map { candidate =>
          budget.recordVisit()
          candidate
        }
        .toVector
        .sortBy(_.toString)
    catch {
      case error: IllegalStateException => throw error
      case NonFatal(_) => Seq.empty
    } finally stream.close()
  }

  private def ldconfigDirectories(budget: SearchBudget): Seq[Path] = {
    val executable =
      firstExecutable(Seq(Paths.get("/sbin/ldconfig"), Paths.get("/usr/sbin/ldconfig")))
        .getOrElse(return Seq.empty)
    try
      runBoundedProcess(
        executable,
        Seq("-p"),
        timeoutMillis = 2000L,
        maxOutputBytes = 1024 * 1024)
        .flatMap { line =>
          budget.recordVisit()
          val separator = line.indexOf("=>")
          if (separator < 0) None
          else {
            val libraryPath = Paths.get(line.substring(separator + 2).trim)
            if (libraryPath.isAbsolute) Option(libraryPath.getParent) else None
          }
        }
    catch {
      case error: IllegalStateException => throw error
      case NonFatal(_) => Seq.empty
    }
  }

  private[onnx] def runBoundedProcess(
      executable: Path,
      arguments: Seq[String],
      timeoutMillis: Long,
      maxOutputBytes: Int): Seq[String] = {
    if (!executable.isAbsolute || !Files.isRegularFile(executable) ||
      !Files.isExecutable(executable))
      throw new IllegalStateException(
        "Bounded process executable must be an absolute executable file")
    if (timeoutMillis <= 0 || maxOutputBytes <= 0)
      throw new IllegalArgumentException("Bounded process limits must be positive")

    val process = new ProcessBuilder((executable.toString +: arguments): _*)
      .redirectErrorStream(true)
      .start()
    runBoundedProcess(process, timeoutMillis, maxOutputBytes)
  }

  private[onnx] def runBoundedProcess(
      process: Process,
      timeoutMillis: Long,
      maxOutputBytes: Int): Seq[String] = {
    if (timeoutMillis <= 0 || maxOutputBytes <= 0)
      throw new IllegalArgumentException("Bounded process limits must be positive")

    val output = new java.util.concurrent.atomic.AtomicReference[Seq[String]]()
    val readFailure = new java.util.concurrent.atomic.AtomicReference[Throwable]()
    val readerThread = new Thread(
      new Runnable {
        override def run(): Unit = {
          val input = process.getInputStream
          try {
            val lines = Vector.newBuilder[String]
            var consumedBytes = 0L
            val currentLine = new ByteArrayOutputStream()

            def appendCurrentLine(): Unit = {
              val bytes = currentLine.toByteArray
              val length =
                if (bytes.nonEmpty && bytes.last == '\r'.toByte) bytes.length - 1
                else bytes.length
              lines += new String(bytes, 0, length, StandardCharsets.UTF_8)
              currentLine.reset()
            }

            var nextByte = input.read()
            while (nextByte >= 0) {
              consumedBytes += 1L
              if (consumedBytes > maxOutputBytes)
                throw new IllegalStateException(
                  s"Bounded process output exceeded $maxOutputBytes bytes")
              if (nextByte == '\n'.toInt) appendCurrentLine()
              else currentLine.write(nextByte)
              nextByte = input.read()
            }
            if (currentLine.size() > 0) appendCurrentLine()
            output.set(lines.result())
          } catch {
            case error: Throwable =>
              readFailure.set(error)
              process.destroyForcibly()
          } finally input.close()
        }
      },
      "onnx-cuda-ldconfig-reader")
    readerThread.setDaemon(true)
    readerThread.start()

    var failure: Throwable = null
    var result: Seq[String] = Seq.empty

    def recordFailure(error: Throwable): Unit =
      if (failure == null) failure = error else failure.addSuppressed(error)

    try {
      if (!process.waitFor(timeoutMillis, TimeUnit.MILLISECONDS)) {
        process.destroyForcibly()
        if (!process.waitFor(timeoutMillis, TimeUnit.MILLISECONDS))
          throw new IllegalStateException(
            s"Bounded process did not terminate after forced destruction within $timeoutMillis milliseconds")
        throw new IllegalStateException(
          s"Bounded process exceeded timeout of $timeoutMillis milliseconds")
      }
      readerThread.join(timeoutMillis)
      if (readerThread.isAlive) {
        process.destroyForcibly()
        if (!process.waitFor(timeoutMillis, TimeUnit.MILLISECONDS))
          throw new IllegalStateException(
            s"Bounded process did not terminate after forced destruction within $timeoutMillis milliseconds")
        throw new IllegalStateException("Bounded process output reader did not terminate")
      }
      Option(readFailure.get()).foreach(error => throw error)
      if (process.exitValue() != 0)
        throw new IllegalStateException(
          s"Bounded process exited with status ${process.exitValue()}")
      result = Option(output.get()).getOrElse(Seq.empty)
    } catch {
      case error: Throwable => recordFailure(error)
    } finally {
      if (process.isAlive) {
        try process.destroyForcibly()
        catch { case error: Throwable => recordFailure(error) }
        try {
          if (!process.waitFor(timeoutMillis, TimeUnit.MILLISECONDS))
            recordFailure(new IllegalStateException(
              s"Bounded process did not terminate after forced destruction within $timeoutMillis milliseconds"))
        } catch { case error: Throwable => recordFailure(error) }
      }
      Seq(process.getInputStream, process.getOutputStream, process.getErrorStream).foreach {
        stream =>
          try stream.close()
          catch { case error: Throwable => recordFailure(error) }
      }
      try readerThread.join(timeoutMillis)
      catch { case error: Throwable => recordFailure(error) }
      if (readerThread.isAlive) {
        readerThread.interrupt()
        try readerThread.join(timeoutMillis)
        catch { case error: Throwable => recordFailure(error) }
        if (readerThread.isAlive)
          recordFailure(
            new IllegalStateException("Bounded process output reader did not terminate"))
      }
    }

    if (failure != null) throw failure
    result
  }

  private[onnx] def cudaRootDirectories(root: Path): Seq[Path] = {
    val budget = new SearchBudget(SearchLimits.default.maxVisitedEntries)
    cudaRootDirectories(root, budget)
  }

  private def cudaRootDirectories(root: Path, budget: SearchBudget): Seq[Path] = {
    if (!root.isAbsolute) return Seq.empty
    val targetRoot = root.resolve("targets")
    val targetLibraries =
      if (!Files.isDirectory(targetRoot) || !Files.isReadable(targetRoot)) Seq.empty
      else {
        val stream = Files.newDirectoryStream(targetRoot)
        try
          stream
            .iterator()
            .asScala
            .map { path =>
              budget.recordVisit()
              path
            }
            .toVector
            .sortBy(_.toString)
            .map(_.resolve("lib"))
        catch {
          case error: IllegalStateException => throw error
          case NonFatal(_) => Seq.empty
        } finally stream.close()
      }
    root.resolve("lib64") +: targetLibraries
  }

  private[onnx] def firstExecutable(candidates: Seq[Path]): Option[Path] =
    candidates.find(path =>
      path.isAbsolute && Files.isRegularFile(path) && Files.isExecutable(path))
}
