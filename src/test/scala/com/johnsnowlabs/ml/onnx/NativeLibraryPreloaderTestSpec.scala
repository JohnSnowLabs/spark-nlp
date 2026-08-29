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

import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtException.OrtErrorCode
import com.johnsnowlabs.tags.FastTest
import org.scalatest.BeforeAndAfterEach
import org.scalatest.flatspec.AnyFlatSpec

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, File, InputStream, OutputStream}
import java.nio.file.{Files, Path, Paths}
import java.nio.file.attribute.{PosixFileAttributes, PosixFilePermissions}
import java.util.concurrent.TimeUnit
import scala.collection.mutable.ArrayBuffer

class NativeLibraryPreloaderTestSpec extends AnyFlatSpec with BeforeAndAfterEach {

  private val requiredLibraries = Seq(
    "libcudart.so.12",
    "libcublasLt.so.12",
    "libcublas.so.12",
    "libcurand.so.10",
    "libcufft.so.11",
    "libcudnn.so.9")

  private val temporaryDirectories = ArrayBuffer.empty[Path]

  override protected def afterEach(): Unit = {
    temporaryDirectories.reverseIterator.foreach(deleteRecursively)
    temporaryDirectories.clear()
    super.afterEach()
  }

  "CUDA preload configuration" should "default to search with empty paths" taggedAs FastTest in {
    val config = NativeLibraryPreloader.PreloadConfig.parse(null, null)

    assert(config.mode == NativeLibraryPreloader.Search)
    assert(config.paths.isEmpty)
  }

  it should "parse every supported mode and reject unknown modes" taggedAs FastTest in {
    assert(
      NativeLibraryPreloader.PreloadConfig.parse("off", "").mode == NativeLibraryPreloader.Off)
    assert(
      NativeLibraryPreloader.PreloadConfig.parse(" explicit ", "/tmp/a").mode ==
        NativeLibraryPreloader.Explicit)

    val error = intercept[IllegalArgumentException] {
      NativeLibraryPreloader.PreloadConfig.parse("automatic", "")
    }
    assert(error.getMessage.contains("off, search, explicit"))
  }

  it should "ignore oversized paths in off mode and reject them in enabled modes" taggedAs FastTest in {
    val oversized = "x" * (NativeLibraryPreloader.MaxConfiguredPathChars + 1)

    assert(
      NativeLibraryPreloader.PreloadConfig.parse("off", oversized) ==
        NativeLibraryPreloader.PreloadConfig(NativeLibraryPreloader.Off, Seq.empty))
    val error = intercept[IllegalArgumentException] {
      NativeLibraryPreloader.PreloadConfig.parse("search", oversized)
    }
    assert(error.getMessage.contains("character limit"))
  }

  it should "aggregate runtime path-list entry and character budgets" taggedAs FastTest in {
    val entryError = intercept[IllegalStateException] {
      NativeLibraryPreloader.parseRuntimePathLists(
        Seq("/one:/two", "/three"),
        NativeLibraryPreloader.SearchLimits(maxDepth = 1, maxVisitedEntries = 2),
        maxCharacters = 100)
    }
    assert(entryError.getMessage.contains("entry limit 2"))

    val characterError = intercept[IllegalStateException] {
      NativeLibraryPreloader.parseRuntimePathLists(
        Seq("/one:/two", "/three"),
        NativeLibraryPreloader.SearchLimits(maxDepth = 1, maxVisitedEntries = 10),
        maxCharacters = 14)
    }
    assert(characterError.getMessage.contains("character limit 14"))
  }

  "packaged CUDA dependency manifest" should "match the ONNX Runtime 1.23 GPU provider order" taggedAs FastTest in {
    assert(NativeLibraryPreloader.packagedDependencies == requiredLibraries)
  }

  "runtime CUDA source discovery" should "discover target libraries without platform-specific leaf names" taggedAs FastTest in {
    val cudaRoot = tempDirectory()
    val targetLib = Files.createDirectories(
      cudaRoot.resolve("targets").resolve("custom-platform").resolve("lib"))

    val directories = NativeLibraryPreloader.cudaRootDirectories(cudaRoot)

    assert(directories.contains(cudaRoot.resolve("lib64")))
    assert(directories.contains(targetLib))
    assert(!directories.exists(_.toString.contains("x86_64-linux")))
  }

  it should "select ldconfig only from fixed executable candidates" taggedAs FastTest in {
    val directory = tempDirectory()
    val notExecutable = createSecureFile(directory.resolve("ldconfig-disabled"))
    val executable = createSecureFile(directory.resolve("ldconfig-enabled"))
    Files.setPosixFilePermissions(executable, PosixFilePermissions.fromString("rwxr-xr-x"))

    assert(
      NativeLibraryPreloader.firstExecutable(Seq(notExecutable, executable)).contains(executable))
  }

  it should "drain bounded process output while the child is running" taggedAs FastTest in {
    val script = tempDirectory().resolve("emit-many-lines")
    Files.write(
      script,
      "#!/bin/sh\ni=0\nwhile [ $i -lt 5000 ]; do echo 'libx.so => /safe/libx.so'; i=$((i+1)); done\n"
        .getBytes("UTF-8"))
    Files.setPosixFilePermissions(script, PosixFilePermissions.fromString("rwx------"))

    val lines = NativeLibraryPreloader.runBoundedProcess(
      script,
      Seq.empty,
      timeoutMillis = 5000L,
      maxOutputBytes = 256 * 1024)

    assert(lines.size == 5000)
  }

  it should "stop an unterminated process line at the output byte limit" taggedAs FastTest in {
    val script = tempDirectory().resolve("emit-unterminated-line")
    Files.write(
      script,
      "#!/bin/sh\nhead -c 4096 /dev/zero | tr '\\000' x\nsleep 10\n".getBytes("UTF-8"))
    Files.setPosixFilePermissions(script, PosixFilePermissions.fromString("rwx------"))

    val error = intercept[IllegalStateException] {
      NativeLibraryPreloader.runBoundedProcess(
        script,
        Seq.empty,
        timeoutMillis = 5000L,
        maxOutputBytes = 1024)
    }

    assert(error.getMessage.contains("output exceeded 1024 bytes"))
  }

  it should "fail when forced process termination cannot be confirmed" taggedAs FastTest in {
    val process = new StubbornProcess

    val error = intercept[IllegalStateException] {
      NativeLibraryPreloader.runBoundedProcess(
        process,
        timeoutMillis = 10L,
        maxOutputBytes = 1024)
    }

    assert(error.getMessage.contains("did not terminate"))
    assert(process.destroyAttempted)
  }

  it should "enforce one aggregate byte limit across included linker configuration files" taggedAs FastTest in {
    val directory = tempDirectory()
    val includes = Files.createDirectories(directory.resolve("conf.d"))
    val firstBytes = (("#" * 80) + "\n/usr/lib\n").getBytes("UTF-8")
    val secondBytes = (("#" * 80) + "\n/usr/local/lib\n").getBytes("UTF-8")
    Files.write(includes.resolve("first.conf"), firstBytes)
    Files.write(includes.resolve("second.conf"), secondBytes)
    val rootLine = s"include ${includes.toAbsolutePath}/*.conf\n".getBytes("UTF-8")
    val rootConfig = directory.resolve("ld.so.conf")
    Files.write(rootConfig, rootLine)
    val aggregateLimit = rootLine.length + firstBytes.length + secondBytes.length - 1

    val error = intercept[IllegalStateException] {
      NativeLibraryPreloader.linkerConfigDirectories(
        rootConfig,
        maxBytes = aggregateLimit,
        NativeLibraryPreloader.SearchLimits(maxDepth = 3, maxVisitedEntries = 100))
    }

    assert(error.getMessage.contains(s"linker-config byte limit $aggregateLimit"))
  }

  "explicit CUDA preload" should "validate the complete ordered set before loading" taggedAs FastTest in {
    val directory = tempDirectory()
    val paths = createLibrarySet(directory).updated(
      requiredLibraries.size - 1,
      directory.resolve("missing-libcudnn.so.9"))
    val loaded = ArrayBuffer.empty[Path]
    val preloader = testPreloader(loaded)

    val error = intercept[IllegalStateException] {
      preloader.preload(
        NativeLibraryPreloader
          .PreloadConfig(NativeLibraryPreloader.Explicit, paths.map(_.toString)))
    }

    assert(error.getMessage.contains("libcudnn.so.9"))
    assert(loaded.isEmpty)
  }

  it should "canonicalize files and load them in manifest order only once" taggedAs FastTest in {
    val directory = tempDirectory()
    val paths = createLibrarySet(directory)
    val loaded = ArrayBuffer.empty[Path]
    val preloader = testPreloader(loaded)
    val config =
      NativeLibraryPreloader.PreloadConfig(NativeLibraryPreloader.Explicit, paths.map(_.toString))

    preloader.preload(config)
    preloader.preload(config)

    assert(loaded == paths.map(_.toRealPath()))
  }

  it should "avoid all search-source discovery in explicit mode" taggedAs FastTest in {
    val directory = tempDirectory()
    val paths = createLibrarySet(directory)
    var sourceReads = 0
    val preloader = new NativeLibraryPreloader(
      requiredLibraries,
      _ => {
        sourceReads += 1
        NativeLibraryPreloader.SearchSources.empty
      },
      NativeLibraryPreloader.SearchLimits.default,
      recordingLoader(ArrayBuffer.empty),
      NativeLibraryPreloader.SecurityPolicy.runtime,
      NativeLibraryPreloader.ElfInspector.accepting)

    preloader.preload(
      NativeLibraryPreloader
        .PreloadConfig(NativeLibraryPreloader.Explicit, paths.map(_.toString)))

    assert(sourceReads == 0)
  }

  it should "reject relative paths and incorrect library order" taggedAs FastTest in {
    val directory = tempDirectory()
    val paths = createLibrarySet(directory)
    val preloader = testPreloader(ArrayBuffer.empty)

    val relativeError = intercept[IllegalStateException] {
      preloader.preload(
        NativeLibraryPreloader.PreloadConfig(
          NativeLibraryPreloader.Explicit,
          Seq("libcudart.so.12") ++ paths.drop(1).map(_.toString)))
    }
    assert(relativeError.getMessage.contains("must be absolute"))

    val orderError = intercept[IllegalStateException] {
      preloader.preload(
        NativeLibraryPreloader.PreloadConfig(
          NativeLibraryPreloader.Explicit,
          Seq(paths(1), paths.head) ++ paths.drop(2) map (_.toString)))
    }
    assert(orderError.getMessage.contains("expected libcudart.so.12"))
  }

  it should "reject non-numeric compatible filename suffixes before loading" taggedAs FastTest in {
    val directory = tempDirectory()
    val paths = createLibrarySet(directory)
    Files.delete(paths.last)
    val invalid = createSecureFile(directory.resolve("libcudnn.so.9.attacker"))
    val loaded = ArrayBuffer.empty[Path]

    val error = intercept[IllegalStateException] {
      testPreloader(loaded).preload(
        NativeLibraryPreloader.PreloadConfig(
          NativeLibraryPreloader.Explicit,
          paths.dropRight(1).map(_.toString) :+ invalid.toString))
    }

    assert(error.getMessage.contains("expected libcudnn.so.9"))
    assert(loaded.isEmpty)
  }

  it should "reject libraries stored in a world-writable directory" taggedAs FastTest in {
    val directory = tempDirectory()
    val paths = createLibrarySet(directory)
    Files.setPosixFilePermissions(directory, PosixFilePermissions.fromString("rwxrwxrwx"))
    val preloader = testPreloader(ArrayBuffer.empty)

    try {
      val error = intercept[IllegalStateException] {
        preloader.preload(
          NativeLibraryPreloader
            .PreloadConfig(NativeLibraryPreloader.Explicit, paths.map(_.toString)))
      }
      assert(error.getMessage.contains("world-writable directory"))
    } finally {
      Files.setPosixFilePermissions(directory, PosixFilePermissions.fromString("rwx------"))
    }
  }

  it should "reject group-writable library files" taggedAs FastTest in {
    val directory = tempDirectory()
    val paths = createLibrarySet(directory)
    Files.setPosixFilePermissions(paths.head, PosixFilePermissions.fromString("rw-rw-r--"))

    val error = intercept[IllegalStateException] {
      testPreloader(ArrayBuffer.empty).preload(
        NativeLibraryPreloader
          .PreloadConfig(NativeLibraryPreloader.Explicit, paths.map(_.toString)))
    }

    assert(error.getMessage.contains("group-writable"))
  }

  it should "reject a library whose owner is outside the trusted-owner policy" taggedAs FastTest in {
    val directory = tempDirectory()
    val paths = createLibrarySet(directory)
    val untrustedPolicy = NativeLibraryPreloader.SecurityPolicy(
      trustedOwners = Set("owner-that-must-not-exist"),
      readAttributes = path => Files.readAttributes(path, classOf[PosixFileAttributes]))

    val error = intercept[IllegalStateException] {
      testPreloader(ArrayBuffer.empty, securityPolicy = untrustedPolicy).preload(
        NativeLibraryPreloader
          .PreloadConfig(NativeLibraryPreloader.Explicit, paths.map(_.toString)))
    }

    assert(error.getMessage.contains("untrusted owner"))
  }

  it should "derive the trusted executor owner from the authenticated process identity" taggedAs FastTest in {
    val originalUserName = Option(System.getProperty("user.name"))
    val actualOwner = currentOwner(Paths.get("/proc/self"))

    try {
      System.setProperty("user.name", "attacker-selected-owner")
      assert(NativeLibraryPreloader.authenticatedProcessOwner() == actualOwner)
      assert(NativeLibraryPreloader.authenticatedProcessOwner() != "attacker-selected-owner")
    } finally
      originalUserName.fold(System.clearProperty("user.name"))(System.setProperty("user.name", _))
  }

  it should "reject replacement through a writable ancestor directory" taggedAs FastTest in {
    val writableAncestor = tempDirectory()
    val libraryDirectory =
      Files.createDirectories(writableAncestor.resolve("secure").resolve("lib"))
    val paths = createLibrarySet(libraryDirectory)
    Files.setPosixFilePermissions(writableAncestor, PosixFilePermissions.fromString("rwxrwxr-x"))

    try {
      val error = intercept[IllegalStateException] {
        testPreloader(ArrayBuffer.empty).preload(
          NativeLibraryPreloader
            .PreloadConfig(NativeLibraryPreloader.Explicit, paths.map(_.toString)))
      }
      assert(error.getMessage.contains("writable ancestor"))
    } finally {
      Files.setPosixFilePermissions(
        writableAncestor,
        PosixFilePermissions.fromString("rwx------"))
    }
  }

  it should "fail closed when POSIX ownership attributes are unavailable" taggedAs FastTest in {
    val directory = tempDirectory()
    val paths = createLibrarySet(directory)
    val unavailablePolicy = NativeLibraryPreloader.SecurityPolicy(
      trustedOwners = Set(currentOwner(directory)),
      readAttributes = _ => throw new UnsupportedOperationException("POSIX unavailable"))

    val error = intercept[IllegalStateException] {
      testPreloader(ArrayBuffer.empty, securityPolicy = unavailablePolicy).preload(
        NativeLibraryPreloader
          .PreloadConfig(NativeLibraryPreloader.Explicit, paths.map(_.toString)))
    }

    assert(error.getMessage.contains("POSIX ownership and permissions are required"))
  }

  "search CUDA preload" should "prefer operator directories and exact SONAME links" taggedAs FastTest in {
    val operatorDirectory = tempDirectory()
    val runtimeDirectory = tempDirectory()
    val operatorPaths = createLibrarySet(operatorDirectory)
    createLibrarySet(runtimeDirectory, versionSuffix = ".99")
    val loaded = ArrayBuffer.empty[Path]
    val sources = NativeLibraryPreloader.SearchSources(
      runtimeDirectories = Seq(runtimeDirectory),
      linkerDirectories = Seq.empty,
      genericRoots = Seq.empty)
    val preloader = testPreloader(loaded, sources)

    preloader.preload(
      NativeLibraryPreloader
        .PreloadConfig(NativeLibraryPreloader.Search, Seq(operatorDirectory.toString)))

    assert(loaded == operatorPaths.map(_.toRealPath()))
  }

  it should "reject an operator SONAME symlink escaping its approved directory" taggedAs FastTest in {
    val operatorDirectory = tempDirectory()
    val outsideDirectory = tempDirectory()
    val outsideTarget = createSecureFile(outsideDirectory.resolve("libcudart.so.12.1"))
    Files.createSymbolicLink(operatorDirectory.resolve("libcudart.so.12"), outsideTarget)
    requiredLibraries
      .drop(1)
      .foreach(library => createSecureFile(operatorDirectory.resolve(library)))

    val error = intercept[IllegalStateException] {
      testPreloader(ArrayBuffer.empty).preload(
        NativeLibraryPreloader
          .PreloadConfig(NativeLibraryPreloader.Search, Seq(operatorDirectory.toString)))
    }

    assert(error.getMessage.contains("outside its approved search root"))
  }

  it should "not traverse generic roots when operator paths resolve the complete set" taggedAs FastTest in {
    val operatorDirectory = tempDirectory()
    val genericRoot = tempDirectory()
    createLibrarySet(operatorDirectory)
    (1 to 4).foreach(index => Files.createDirectory(genericRoot.resolve(s"directory-$index")))
    val loaded = ArrayBuffer.empty[Path]
    val preloader = testPreloader(
      loaded,
      NativeLibraryPreloader.SearchSources(
        runtimeDirectories = Seq.empty,
        linkerDirectories = Seq.empty,
        genericRoots = Seq(genericRoot)),
      NativeLibraryPreloader.SearchLimits(
        maxDepth = 3,
        maxVisitedEntries = requiredLibraries.size + 1))

    preloader.preload(
      NativeLibraryPreloader
        .PreloadConfig(NativeLibraryPreloader.Search, Seq(operatorDirectory.toString)))

    assert(loaded.size == requiredLibraries.size)
  }

  it should "not enumerate lower-priority tiers when operator paths resolve the complete set" taggedAs FastTest in {
    val operatorDirectory = tempDirectory()
    val runtimeDirectory = tempDirectory()
    createLibrarySet(operatorDirectory)
    (1 to 10).foreach(index => createSecureFile(runtimeDirectory.resolve(s"unrelated-$index.so")))
    val loaded = ArrayBuffer.empty[Path]
    val preloader = testPreloader(
      loaded,
      NativeLibraryPreloader.SearchSources(
        runtimeDirectories = Seq(runtimeDirectory),
        linkerDirectories = Seq.empty,
        genericRoots = Seq.empty),
      NativeLibraryPreloader.SearchLimits(
        maxDepth = 3,
        maxVisitedEntries = requiredLibraries.size + 1))

    preloader.preload(
      NativeLibraryPreloader
        .PreloadConfig(NativeLibraryPreloader.Search, Seq(operatorDirectory.toString)))

    assert(loaded.size == requiredLibraries.size)
  }

  it should "defer linker and generic discovery when runtime paths resolve the complete set" taggedAs FastTest in {
    val runtimeDirectory = tempDirectory()
    createLibrarySet(runtimeDirectory)
    val loaded = ArrayBuffer.empty[Path]
    var linkerReads = 0
    var genericReads = 0
    val sources = NativeLibraryPreloader.SearchSources.deferred(
      runtimeDirectories = Seq(runtimeDirectory),
      linkerDirectories = {
        linkerReads += 1
        Seq(tempDirectory())
      },
      genericRoots = {
        genericReads += 1
        Seq(tempDirectory())
      })
    val preloader = testPreloader(loaded, sources)

    preloader.preload(
      NativeLibraryPreloader.PreloadConfig(NativeLibraryPreloader.Search, Seq.empty))

    assert(loaded.size == requiredLibraries.size)
    assert(linkerReads == 0)
    assert(genericReads == 0)
  }

  it should "charge configured search directories to the shared work budget" taggedAs FastTest in {
    val directories = Seq(tempDirectory(), tempDirectory(), tempDirectory())
    val preloader = testPreloader(
      ArrayBuffer.empty,
      NativeLibraryPreloader.SearchSources.empty,
      NativeLibraryPreloader.SearchLimits(maxDepth = 3, maxVisitedEntries = 2))

    val error = intercept[IllegalStateException] {
      preloader.preload(
        NativeLibraryPreloader
          .PreloadConfig(NativeLibraryPreloader.Search, directories.map(_.toString)))
    }

    assert(error.getMessage.contains("entry limit 2"))
  }

  it should "fail closed when a source tier has ambiguous compatible files" taggedAs FastTest in {
    val runtimeDirectory = tempDirectory()
    requiredLibraries.foreach { library =>
      if (library == "libcudart.so.12") {
        createSecureFile(runtimeDirectory.resolve(library + ".1"))
        createSecureFile(runtimeDirectory.resolve(library + ".2"))
      } else createSecureFile(runtimeDirectory.resolve(library))
    }
    val preloader = testPreloader(
      ArrayBuffer.empty,
      NativeLibraryPreloader.SearchSources(
        runtimeDirectories = Seq(runtimeDirectory),
        linkerDirectories = Seq.empty,
        genericRoots = Seq.empty))

    val error = intercept[IllegalStateException] {
      preloader.preload(
        NativeLibraryPreloader.PreloadConfig(NativeLibraryPreloader.Search, Seq.empty))
    }

    assert(error.getMessage.contains("Ambiguous CUDA library libcudart.so.12"))
  }

  it should "bound direct source-tier enumeration" taggedAs FastTest in {
    val runtimeDirectory = tempDirectory()
    createLibrarySet(runtimeDirectory)
    (1 to 10).foreach(index => createSecureFile(runtimeDirectory.resolve(s"unrelated-$index.so")))
    val preloader = testPreloader(
      ArrayBuffer.empty,
      NativeLibraryPreloader.SearchSources(
        runtimeDirectories = Seq(runtimeDirectory),
        linkerDirectories = Seq.empty,
        genericRoots = Seq.empty),
      NativeLibraryPreloader.SearchLimits(maxDepth = 3, maxVisitedEntries = 4))

    val error = intercept[IllegalStateException] {
      preloader.preload(
        NativeLibraryPreloader.PreloadConfig(NativeLibraryPreloader.Search, Seq.empty))
    }

    assert(error.getMessage.contains("entry limit"))
  }

  it should "reject relative runtime-derived search directories" taggedAs FastTest in {
    val preloader = testPreloader(
      ArrayBuffer.empty,
      NativeLibraryPreloader.SearchSources(
        runtimeDirectories = Seq(Paths.get("relative-cuda-directory")),
        linkerDirectories = Seq.empty,
        genericRoots = Seq.empty))

    val error = intercept[IllegalStateException] {
      preloader.preload(
        NativeLibraryPreloader.PreloadConfig(NativeLibraryPreloader.Search, Seq.empty))
    }

    assert(error.getMessage.contains("must be absolute"))
  }

  it should "prefer an exact SONAME file symlink during generic-root traversal" taggedAs FastTest in {
    val root = tempDirectory()
    val libraryDirectory = Files.createDirectory(root.resolve("cuda-libraries"))
    Files.setPosixFilePermissions(libraryDirectory, PosixFilePermissions.fromString("rwxr-xr-x"))
    val selectedTarget = createSecureFile(libraryDirectory.resolve("libcudart.so.12.1"))
    createSecureFile(libraryDirectory.resolve("libcudart.so.12.2"))
    Files.createSymbolicLink(libraryDirectory.resolve("libcudart.so.12"), selectedTarget)
    requiredLibraries
      .drop(1)
      .foreach(library => createSecureFile(libraryDirectory.resolve(library)))
    val loaded = ArrayBuffer.empty[Path]
    val preloader = testPreloader(
      loaded,
      NativeLibraryPreloader.SearchSources(
        runtimeDirectories = Seq.empty,
        linkerDirectories = Seq.empty,
        genericRoots = Seq(root)))

    preloader.preload(
      NativeLibraryPreloader.PreloadConfig(NativeLibraryPreloader.Search, Seq.empty))

    assert(loaded.head == selectedTarget.toRealPath())
  }

  it should "not follow directory symlinks during bounded generic-root traversal" taggedAs FastTest in {
    val root = tempDirectory()
    val outside = tempDirectory()
    createLibrarySet(outside)
    Files.createSymbolicLink(root.resolve("linked-cuda"), outside)
    val preloader = testPreloader(
      ArrayBuffer.empty,
      NativeLibraryPreloader.SearchSources(
        runtimeDirectories = Seq.empty,
        linkerDirectories = Seq.empty,
        genericRoots = Seq(root)),
      NativeLibraryPreloader.SearchLimits(maxDepth = 3, maxVisitedEntries = 100))

    val error = intercept[IllegalStateException] {
      preloader.preload(
        NativeLibraryPreloader.PreloadConfig(NativeLibraryPreloader.Search, Seq.empty))
    }

    assert(error.getMessage.contains("libcudart.so.12"))
  }

  it should "stop when the generic-root entry budget is exhausted" taggedAs FastTest in {
    val root = tempDirectory()
    (1 to 4).foreach(index => Files.createDirectory(root.resolve(s"directory-$index")))
    val preloader = testPreloader(
      ArrayBuffer.empty,
      NativeLibraryPreloader.SearchSources(
        runtimeDirectories = Seq.empty,
        linkerDirectories = Seq.empty,
        genericRoots = Seq(root)),
      NativeLibraryPreloader.SearchLimits(maxDepth = 3, maxVisitedEntries = 2))

    val error = intercept[IllegalStateException] {
      preloader.preload(
        NativeLibraryPreloader.PreloadConfig(NativeLibraryPreloader.Search, Seq.empty))
    }

    assert(error.getMessage.contains("entry limit"))
  }

  "CUDA provider failure classification" should "accept only missing native dependencies of the CUDA provider" taggedAs FastTest in {
    assert(CudaProviderRecovery.isRecoverable(recoverableOrtFailure()))
    assert(
      CudaProviderRecovery.isRecoverable(new OrtException(
        OrtErrorCode.ORT_EP_FAIL,
        "Failed to load library libonnxruntime_providers_cuda.so with error: libcudnn.so.9: cannot open shared object file")))

    assert(!CudaProviderRecovery.isRecoverable(new RuntimeException(
      "Failed to load library libonnxruntime_providers_cuda.so with error: libcudnn.so.9: cannot open shared object file")))
    assert(
      !CudaProviderRecovery.isRecoverable(new OrtException(
        OrtErrorCode.ORT_INVALID_ARGUMENT,
        "Failed to load library libonnxruntime_providers_cuda.so with error: libcudnn.so.9: cannot open shared object file")))
    assert(
      !CudaProviderRecovery.isRecoverable(
        new OrtException(OrtErrorCode.ORT_FAIL, "Failed to find CUDA shared provider")))
    assert(!CudaProviderRecovery.isRecoverable(new RuntimeException("CUDA out of memory")))
    assert(!CudaProviderRecovery.isRecoverable(new RuntimeException("Invalid GPU device ID 4")))
    assert(!CudaProviderRecovery.isRecoverable(new RuntimeException("Unsupported ONNX operator")))
    assert(!CudaProviderRecovery.isRecoverable(recoverableOrtFailure("libcudnn.so.90")))
    assert(!CudaProviderRecovery.isRecoverable(recoverableOrtFailure("libcudnn.so.9.attacker")))
    assert(
      !CudaProviderRecovery.isRecoverable(
        recoverableOrtFailure(),
        () => throw new IllegalStateException("manifest unavailable")))
  }

  "OnnxWrapper CUDA preload runtime configuration" should "use the documented defaults and system-property overrides" taggedAs FastTest in {
    val modeKey = "spark.jsl.settings.onnx.cuda.preload.mode"
    val pathsKey = "spark.jsl.settings.onnx.cuda.preload.paths"
    val originalMode = Option(System.getProperty(modeKey))
    val originalPaths = Option(System.getProperty(pathsKey))

    try {
      System.clearProperty(modeKey)
      System.clearProperty(pathsKey)
      assert(
        OnnxWrapper.cudaPreloadConfig() == NativeLibraryPreloader
          .PreloadConfig(NativeLibraryPreloader.Search, Seq.empty))

      System.setProperty(modeKey, "explicit")
      System.setProperty(pathsKey, Seq("/cuda/a", "/cuda/b").mkString(File.pathSeparator))
      assert(
        OnnxWrapper.cudaPreloadConfig() == NativeLibraryPreloader
          .PreloadConfig(NativeLibraryPreloader.Explicit, Seq("/cuda/a", "/cuda/b")))
    } finally {
      originalMode.fold(System.clearProperty(modeKey))(System.setProperty(modeKey, _))
      originalPaths.fold(System.clearProperty(pathsKey))(System.setProperty(pathsKey, _))
    }
  }

  "CUDA provider recovery" should "leave the normal success path unchanged" taggedAs FastTest in {
    val created = ArrayBuffer.empty[TestSessionOptions]
    var configurationReads = 0
    var recoveries = 0

    val result = CudaProviderRecovery.configure(
      {
        configurationReads += 1
        NativeLibraryPreloader.Search
      },
      () => newOptions(created),
      (_: TestSessionOptions) => (),
      () => recoveries += 1)

    assert(result eq created.head)
    assert(created.size == 1)
    assert(configurationReads == 0)
    assert(recoveries == 0)
    assert(!created.head.closed)
  }

  it should "close failed options, preload, and retry once with fresh options" taggedAs FastTest in {
    val created = ArrayBuffer.empty[TestSessionOptions]
    var additions = 0
    var recoveries = 0

    val result = CudaProviderRecovery.configure(
      NativeLibraryPreloader.Search,
      () => newOptions(created),
      (_: TestSessionOptions) => {
        additions += 1
        if (additions == 1) throw recoverableOrtFailure("libcublasLt.so.12")
      },
      () => recoveries += 1)

    assert(created.size == 2)
    assert(created.head.closed)
    assert(result eq created(1))
    assert(!created(1).closed)
    assert(additions == 2)
    assert(recoveries == 1)
  }

  it should "preserve the original failure in off mode without preloading or retrying" taggedAs FastTest in {
    val created = ArrayBuffer.empty[TestSessionOptions]
    val original = recoverableOrtFailure("libcublasLt.so.12")
    var recoveries = 0

    val thrown = intercept[OrtException] {
      CudaProviderRecovery.configure(
        NativeLibraryPreloader.Off,
        () => newOptions(created),
        (_: TestSessionOptions) => throw original,
        () => recoveries += 1)
    }

    assert(thrown eq original)
    assert(created.size == 1)
    assert(created.head.closed)
    assert(recoveries == 0)
  }

  it should "not recover an unrelated CUDA failure" taggedAs FastTest in {
    val created = ArrayBuffer.empty[TestSessionOptions]
    val original = new RuntimeException("CUDA out of memory")
    var recoveries = 0

    val thrown = intercept[RuntimeException] {
      CudaProviderRecovery.configure(
        NativeLibraryPreloader.Search,
        () => newOptions(created),
        (_: TestSessionOptions) => throw original,
        () => recoveries += 1)
    }

    assert(thrown eq original)
    assert(created.size == 1)
    assert(created.head.closed)
    assert(recoveries == 0)
  }

  it should "preserve provider diagnostics when failed options closing is interrupted" taggedAs FastTest in {
    val created = ArrayBuffer.empty[TestSessionOptions]
    val original = new RuntimeException("CUDA out of memory")
    val closeFailure = new InterruptedException("close interrupted")

    val thrown = intercept[RuntimeException] {
      CudaProviderRecovery.configure(
        NativeLibraryPreloader.Search,
        () => newOptions(created, Some(closeFailure)),
        (_: TestSessionOptions) => throw original,
        () => ())
    }

    assert(thrown eq original)
    assert(thrown.getSuppressed.toSeq.contains(closeFailure))
    assert(created.head.closed)
  }

  it should "retain the original CUDA error when preloading fails" taggedAs FastTest in {
    val created = ArrayBuffer.empty[TestSessionOptions]
    val original = recoverableOrtFailure()
    val preloadFailure = new IllegalStateException("CUDA dependency resolution failed")

    val thrown = intercept[IllegalStateException] {
      CudaProviderRecovery.configure(
        NativeLibraryPreloader.Search,
        () => newOptions(created),
        (_: TestSessionOptions) => throw original,
        () => throw preloadFailure)
    }

    assert(thrown eq preloadFailure)
    assert(thrown.getSuppressed.contains(original))
    assert(created.size == 1)
    assert(created.head.closed)
  }

  it should "close retry options and fail instead of falling back when retry fails" taggedAs FastTest in {
    val created = ArrayBuffer.empty[TestSessionOptions]
    var additions = 0
    var recoveries = 0

    val retryFailure = intercept[RuntimeException] {
      CudaProviderRecovery.configure(
        NativeLibraryPreloader.Explicit,
        () => newOptions(created),
        (_: TestSessionOptions) => {
          additions += 1
          if (additions == 1) throw recoverableOrtFailure()
          else throw new RuntimeException("CUDA provider retry failed")
        },
        () => recoveries += 1)
    }

    assert(retryFailure.getMessage == "CUDA provider retry failed")
    assert(created.size == 2)
    assert(created.forall(_.closed))
    assert(additions == 2)
    assert(recoveries == 1)
  }

  it should "preserve the original failure when retry options cannot be created" taggedAs FastTest in {
    val created = ArrayBuffer.empty[TestSessionOptions]
    val original = recoverableOrtFailure()
    val creationFailure = new IllegalStateException("retry options construction failed")
    var creations = 0

    val thrown = intercept[IllegalStateException] {
      CudaProviderRecovery.configure(
        NativeLibraryPreloader.Search,
        () => {
          creations += 1
          if (creations == 1) newOptions(created) else throw creationFailure
        },
        (_: TestSessionOptions) => throw original,
        () => ())
    }

    assert(thrown eq creationFailure)
    assert(thrown.getSuppressed.contains(original))
    assert(created.size == 1)
    assert(created.head.closed)
  }

  private def testPreloader(
      loaded: ArrayBuffer[Path],
      sources: NativeLibraryPreloader.SearchSources = NativeLibraryPreloader.SearchSources.empty,
      limits: NativeLibraryPreloader.SearchLimits = NativeLibraryPreloader.SearchLimits.default,
      securityPolicy: NativeLibraryPreloader.SecurityPolicy =
        NativeLibraryPreloader.SecurityPolicy.runtime): NativeLibraryPreloader =
    new NativeLibraryPreloader(
      requiredLibraries,
      _ => sources,
      limits,
      recordingLoader(loaded),
      securityPolicy,
      NativeLibraryPreloader.ElfInspector.accepting)

  private def recordingLoader(loaded: ArrayBuffer[Path]): NativeLibraryPreloader.LibraryLoader =
    new NativeLibraryPreloader.LibraryLoader {
      override def load(path: Path): Unit = loaded += path
    }

  private def currentOwner(path: Path): String =
    Files.readAttributes(path, classOf[PosixFileAttributes]).owner().getName

  private def tempDirectory(): Path = {
    val testRoot = Paths.get(System.getProperty("user.home"), ".cache", "spark-nlp-tests")
    Files.createDirectories(testRoot)
    Files.setPosixFilePermissions(testRoot, PosixFilePermissions.fromString("rwx------"))
    val directory = Files.createTempDirectory(testRoot, "onnx-cuda-preload-test-")
    temporaryDirectories += directory
    directory
  }

  private def createSecureFile(path: Path): Path = {
    val file = Files.createFile(path)
    Files.setPosixFilePermissions(file, PosixFilePermissions.fromString("rw-r--r--"))
    file
  }

  private def createLibrarySet(directory: Path, versionSuffix: String = ""): Seq[Path] =
    requiredLibraries.map(library => createSecureFile(directory.resolve(library + versionSuffix)))

  private def deleteRecursively(path: Path): Unit = {
    if (Files.exists(path)) {
      val stream = Files.walk(path)
      try stream.sorted(java.util.Comparator.reverseOrder()).forEach(Files.deleteIfExists(_))
      finally stream.close()
    }
  }

  private def newOptions(
      created: ArrayBuffer[TestSessionOptions],
      closeFailure: Option[Throwable] = None): TestSessionOptions = {
    val options = new TestSessionOptions(closeFailure)
    created += options
    options
  }

  private def recoverableOrtFailure(missingLibrary: String = "libcudnn.so.9"): OrtException =
    new OrtException(
      OrtErrorCode.ORT_FAIL,
      s"Failed to load library libonnxruntime_providers_cuda.so with error: $missingLibrary: cannot open shared object file")

  private final class StubbornProcess extends Process {
    private val input = new ByteArrayInputStream(Array.emptyByteArray)
    private val error = new ByteArrayInputStream(Array.emptyByteArray)
    private val output = new ByteArrayOutputStream()
    var destroyAttempted = false

    override def getOutputStream: OutputStream = output
    override def getInputStream: InputStream = input
    override def getErrorStream: InputStream = error
    override def waitFor(): Int = throw new InterruptedException("stubborn process")
    override def waitFor(timeout: Long, unit: TimeUnit): Boolean = false
    override def exitValue(): Int = throw new IllegalThreadStateException("still running")
    override def destroy(): Unit = destroyAttempted = true
    override def destroyForcibly(): Process = {
      destroyAttempted = true
      this
    }
    override def isAlive: Boolean = true
  }

  private final class TestSessionOptions(closeFailure: Option[Throwable] = None)
      extends AutoCloseable {
    var closed: Boolean = false
    override def close(): Unit = {
      closed = true
      closeFailure.foreach(throw _)
    }
  }
}
