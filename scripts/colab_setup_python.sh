#!/bin/bash
#
# Google Colab setup for Spark NLP + PySpark, including a downgrade of the
# notebook kernel's Python version.
#
# WHY THIS EXISTS
#   Colab's default runtime now ships Python 3.13. No PySpark 3.x release
#   supports it - pyspark 3.4.x and 3.5.x both declare support only through
#   Python 3.11 - and Spark NLP 6.x runs on Apache Spark 3.0.x-3.5.x only, so
#   "just use Spark 4" is not an option either. scripts/colab_setup.sh installs
#   PySpark against whatever Python Colab gives it, which no longer works.
#
#   This script does everything colab_setup.sh does (Java, optional GPU cuDNN,
#   pyspark + spark-nlp + findspark) and additionally replaces the interpreter
#   behind Colab's default "Python 3" kernel with an older one, so the cells
#   you run after a restart execute on a Python that PySpark supports.
#
# HOW IT WORKS
#   Miniconda for the target Python is installed over /usr/local, and the
#   packages Colab's frontend needs to talk to a kernel (jupyter, google-colab,
#   traitlets) are reinstalled there. Technique adapted from
#   https://github.com/j3soon/colab-python-version
#
#   Redirecting the *default* kernel then needs one extra step. Colab does not
#   honour a kernelspec's argv for it: whatever kernel.json says, the runtime
#   launches the hardcoded command
#
#       /usr/bin/python3 -m colab_kernel_launcher -f {connection_file}
#
#   so installing a kernelspec - user-level or system - changes nothing. The
#   only interception point is the launcher module itself, so this script
#   replaces colab_kernel_launcher.py in Colab's 3.13 dist-packages with a shim
#   that re-execs the same launcher under the new interpreter, carrying the
#   connection file through unchanged. The original is kept alongside it and is
#   re-run automatically if the new interpreter has gone missing.
#
#   Kernelspecs are still registered (a "python3" override plus a versioned
#   name) so notebooks that do pin metadata.kernelspec.name keep working, but
#   the shim is what actually moves the default kernel.
#
# USAGE (in a Colab cell)
#   !wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp/master/scripts/colab_setup_python.sh -O - | bash
#   ...then Runtime > Restart session, and run your next cell.
#
#   With options:
#   !wget -q .../colab_setup_python.sh -O - | bash -s -- -v 3.11 -p 3.4 -s 6.4.2
#
# OPTIONS
#   -v  target Python version    (default 3.11; 3.8-3.11 verified, see below)
#   -s  Spark NLP version        (default 6.4.2)
#   -p  PySpark version/series   (default 3.4.4)
#   -m  Miniconda build to pull  (default 26.7.1-1)
#   -g  GPU: upgrade libcudnn8
#   -f  force: register the kernel even if the pre-flight check fails
#   -h  help
#
# WHY 3.11 AND NOT 3.12
#   Two independent ceilings both land on 3.11:
#     1. pyspark 3.4.4 / 3.5.9 declare Python 3.7-3.11 and 3.8-3.11.
#     2. This kernel-swap technique is only known to work through 3.11. On
#        3.12 the kernel dies during init with "The 'kernel_class' trait of
#        IPKernelApp instance must be a type, but 'google.colab._kernel.Kernel'
#        could not be imported".
#   -v 3.12 is accepted, but the pre-flight check below will very likely stop
#   the run before it can strand you. Use -f to override at your own risk.
#
# IF THE RUNTIME WON'T RECONNECT AFTER RESTARTING
#   Runtime > Disconnect and delete runtime. That discards everything this
#   script did, including the launcher shim, and is the only reliable recovery
#   once the kernel will not start - by then no cell can run to undo anything.
#   (For reference, the shim restores with
#   "cp <dist-packages>/colab_kernel_launcher.sparknlp-original.py \
#       <dist-packages>/colab_kernel_launcher.py"
#   and the kernelspec with "rm -rf ~/.local/share/jupyter/kernels/python3".)
#
# IF A RUN FAILS PART WAY THROUGH
#   Delete the runtime before retrying. A failed run can leave /usr/local
#   half-rewritten by the Miniconda installer, and re-running on top of that
#   does not reliably recover.

set -e

SPARKNLP="6.4.2"
PYSPARK="3.4.4"
PYTHON_VERSION="3.11"
MINICONDA_BUILD="26.7.1-1"
TRAITLETS="5.5.0"
GPU="false"
FORCE="false"

usage() {
  awk '/^# USAGE/,/^#   -h  help/' "$0" | cut -c3-
}

while getopts s:p:v:m:gfh option; do
  case "${option}" in
  s) SPARKNLP=${OPTARG} ;;
  p) PYSPARK=${OPTARG} ;;
  v) PYTHON_VERSION=${OPTARG} ;;
  m) MINICONDA_BUILD=${OPTARG} ;;
  g) GPU="true" ;;
  f) FORCE="true" ;;
  h)
    usage
    exit 0
    ;;
  *)
    echo "Error: Invalid option -${OPTARG}" >&2
    usage >&2
    exit 1
    ;;
  esac
done

# Same PySpark series mapping as scripts/colab_setup.sh, plus 3.5.
if [[ "$PYSPARK" == "3.5"* ]]; then
  PYSPARK="3.5.9"
elif [[ "$PYSPARK" == "3.4"* ]]; then
  PYSPARK="3.4.4"
elif [[ "$PYSPARK" == "3.3"* ]]; then
  PYSPARK="3.3.4"
elif [[ "$PYSPARK" == "3.2"* ]]; then
  PYSPARK="3.2.4"
elif [[ "$PYSPARK" == "3.1"* ]]; then
  PYSPARK="3.1.3"
elif [[ "$PYSPARK" == "3.0"* ]]; then
  PYSPARK="3.0.3"
else
  PYSPARK="3.4.4"
fi

PY_NODOT="${PYTHON_VERSION/./}"
KERNEL_NAME="py${PY_NODOT}"
MINICONDA_INSTALLER="Miniconda3-py${PY_NODOT}_${MINICONDA_BUILD}-Linux-x86_64.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}"
CONDA_PY="/usr/local/bin/python"
KERNEL_DIR="${HOME}/.local/share/jupyter/kernels"

CURRENT_PY="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo unknown)"

echo "=================================================================="
echo " Spark NLP ${SPARKNLP} + PySpark ${PYSPARK} on Python ${PYTHON_VERSION}"
echo " Colab kernel Python is ${CURRENT_PY}  ->  switching to ${PYTHON_VERSION}"
echo "=================================================================="

case "$PYTHON_VERSION" in
3.8 | 3.9 | 3.10 | 3.11) ;;
3.12 | 3.13 | 3.14)
  echo
  echo "WARNING: Python ${PYTHON_VERSION} is not a verified target for this technique."
  echo "         Colab's kernel is known to fail initialization on 3.12+, and no"
  echo "         PySpark 3.x supports it. The pre-flight check will most likely"
  echo "         abort before the kernel is switched. 3.11 is the recommended target."
  echo
  ;;
*)
  echo "Error: unsupported target Python version '${PYTHON_VERSION}'." >&2
  exit 1
  ;;
esac

# set -e makes any failure below abort silently, which is hard to read when it
# happens inside a quiet install step. Say plainly what state the runtime is in.
trap 'status=$?;
  echo;
  echo "------------------------------------------------------------------";
  echo "SETUP FAILED (exit ${status}) - the default kernel was NOT switched.";
  echo "See the error above. /usr/local may be partly rewritten, so use";
  echo "Runtime > Disconnect and delete runtime for a clean image before";
  echo "retrying rather than re-running in this session.";
  echo "------------------------------------------------------------------";
  exit ${status}' ERR

# Fail before touching anything if the requested Miniconda build does not exist.
echo "[1/7] Checking ${MINICONDA_INSTALLER} is available..."
if ! wget -q --spider "$MINICONDA_URL"; then
  echo "Error: ${MINICONDA_URL} not found." >&2
  echo "       Pick an existing build with -m (see https://repo.anaconda.com/miniconda/)." >&2
  exit 1
fi

# ---------------------------------------------------------------- Java -----
# Spark 3.x supports Java 8, 11 and 17 only, and Colab's image ships Java 21.
# Rather than parsing `java -version` (whose output format varies), just make
# sure a JDK Spark accepts is on disk and point JAVA_HOME straight at it.
echo "[2/7] Java (Spark 3.x requires Java 8, 11 or 17)..."
JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
if [ ! -d "$JAVA_HOME" ]; then
  echo "       Installing OpenJDK 11..."
  apt-get update -qq >/dev/null 2>&1 || true
  apt-get install -y -qq openjdk-11-jdk >/dev/null 2>&1 || true
fi

if [ ! -d "$JAVA_HOME" ]; then
  echo "       OpenJDK 11 unavailable; looking for another supported JDK..."
  for candidate in /usr/lib/jvm/java-17-openjdk-amd64 /usr/lib/jvm/java-8-openjdk-amd64; do
    if [ -d "$candidate" ]; then
      JAVA_HOME="$candidate"
      break
    fi
  done
fi

if [ ! -d "$JAVA_HOME" ]; then
  echo "Error: no Spark-compatible JDK (8/11/17) found or installable." >&2
  echo "       Colab's default Java 21 is rejected by Spark 3.x." >&2
  exit 1
fi
export JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"
echo "       JAVA_HOME=${JAVA_HOME}"

if [[ "$GPU" == "true" ]]; then
  echo "       Upgrading libcudnn8 to 8.1.0 for GPU"
  apt install -qq --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2 -y &>/dev/null
fi

# ------------------------------------------------------------ Interpreter --
# Installs over /usr/local so /usr/local/bin/python becomes the target version.
# Colab's own dist-packages live under /usr/local/lib/python3.13 and are left in
# place, so the kernel running this cell keeps working until you restart.
echo "[3/7] Installing Python ${PYTHON_VERSION} (${MINICONDA_INSTALLER})..."
wget -q -O /tmp/miniconda.sh "$MINICONDA_URL"
bash /tmp/miniconda.sh -b -f -p /usr/local >/dev/null
rm -f /tmp/miniconda.sh

# The packages Colab's frontend needs in order to drive the kernel. The
# traitlets pin works around an incompatibility between google-colab 1.0.0 (the
# version the Colab runtime itself ships) and newer traitlets releases.
#
# --override-channels is required: conda 25+ refuses to use repo.anaconda.com's
# "defaults" channels until their Terms of Service are accepted, which cannot
# happen in a non-interactive run and aborts with CondaToSNonInteractiveError.
# Everything needed here lives on conda-forge, so drop defaults entirely. The
# tos accept calls are a best-effort fallback for setups that still reach for
# them, and are ignored on conda versions without the subcommand.
echo "[4/7] Installing Colab kernel packages (jupyter, google-colab, traitlets=${TRAITLETS})..."
for channel in main r; do
  /usr/local/bin/conda tos accept --override-channels \
    --channel "https://repo.anaconda.com/pkgs/${channel}" >/dev/null 2>&1 || true
done
/usr/local/bin/conda install -q -y --override-channels -c conda-forge \
  jupyter google-colab "traitlets=${TRAITLETS}" >/dev/null

# --------------------------------------------------------------- Payload ---
echo "[5/7] Installing PySpark ${PYSPARK}, Spark NLP ${SPARKNLP}, findspark..."
"$CONDA_PY" -m pip install --upgrade -q "pyspark==${PYSPARK}" "spark-nlp==${SPARKNLP}" findspark

# ------------------------------------------------------------- Pre-flight --
# Everything above is additive and undone by discarding the runtime. The
# kernelspec below is what actually redirects Colab, so verify the new
# interpreter can do what the kernel needs before writing it. This is the check
# that catches the 3.12 "kernel_class ... could not be imported" failure
# *before* it can leave you with a runtime that never reconnects.
echo "[6/7] Pre-flight check on the new interpreter..."

# Colab launches "/usr/bin/python3 -m colab_kernel_launcher", so the new
# interpreter has to be able to run that module. conda-forge's google-colab
# does not always ship it; when it is missing, copy the one Colab is using.
#
# Locate it with find_spec rather than importing it: the launcher's job is to
# start a kernel, so importing it here could block or spawn one.
FIND_LAUNCHER='import importlib.util as u; s = u.find_spec("colab_kernel_launcher"); print(s.origin if s and s.origin else "")'
SYS_LAUNCHER="$(/usr/bin/python3 -c "$FIND_LAUNCHER" 2>/dev/null || true)"
NEW_LAUNCHER="$("$CONDA_PY" -c "$FIND_LAUNCHER" 2>/dev/null || true)"
if [ -z "$NEW_LAUNCHER" ] && [ -n "$SYS_LAUNCHER" ] && [ -f "$SYS_LAUNCHER" ]; then
  SITE_PACKAGES="$("$CONDA_PY" -c 'import site; print(site.getsitepackages()[0])')"
  echo "       copying colab_kernel_launcher into the new interpreter"
  cp "$SYS_LAUNCHER" "${SITE_PACKAGES}/colab_kernel_launcher.py"
  NEW_LAUNCHER="$("$CONDA_PY" -c "$FIND_LAUNCHER" 2>/dev/null || true)"
fi

trap - ERR # the pre-flight block reports its own failures
PREFLIGHT_OK="true"
"$CONDA_PY" - <<'PYEOF' || PREFLIGHT_OK="false"
import sys

from google.colab._kernel import Kernel  # noqa: F401  the class Colab's kernel config names
import ipykernel  # noqa: F401
import pyspark
import sparknlp

print("       interpreter : %s" % sys.executable)
print("       python      : %d.%d.%d" % sys.version_info[:3])
print("       pyspark     : %s" % pyspark.__version__)
print("       spark-nlp   : %s" % sparknlp.version())
PYEOF

if [[ "$PREFLIGHT_OK" == "true" ]]; then
  if [ -z "$SYS_LAUNCHER" ] || [ ! -f "$SYS_LAUNCHER" ]; then
    echo "       could not locate Colab's colab_kernel_launcher module"
    PREFLIGHT_OK="false"
  elif [ -z "$NEW_LAUNCHER" ]; then
    echo "       colab_kernel_launcher is not importable under Python ${PYTHON_VERSION}"
    PREFLIGHT_OK="false"
  else
    echo "       launcher    : ${SYS_LAUNCHER}"
  fi
fi

if [[ "$PREFLIGHT_OK" != "true" ]]; then
  echo
  echo "------------------------------------------------------------------"
  echo "PRE-FLIGHT CHECK FAILED - the default kernel was NOT switched."
  echo
  echo "Python ${PYTHON_VERSION} cannot start a Colab kernel in this runtime."
  echo "Nothing that would stop the runtime from reconnecting has been written,"
  echo "so this session is still usable."
  echo
  echo "Re-run with -v 3.11 (the recommended target), or pass -f to register the"
  echo "kernel anyway and accept that the runtime may fail to reconnect."
  echo "------------------------------------------------------------------"
  [[ "$FORCE" == "true" ]] || exit 1
  echo "-f given: continuing anyway."
fi

# ------------------------------------------------------------ Kernelspec ---
# These do NOT move Colab's default kernel - the runtime ignores kernel.json's
# argv and execs a hardcoded command instead (the shim below is what redirects
# it). They are registered so that notebooks pinning metadata.kernelspec.name,
# and any tooling that reads the specs, see the right interpreter.
echo "[7/7] Registering kernelspecs and redirecting Colab's kernel launcher..."
"$CONDA_PY" -m ipykernel install --user \
  --name "$KERNEL_NAME" --display-name "Python ${PYTHON_VERSION} (Spark NLP)" >/dev/null
"$CONDA_PY" -m ipykernel install --user \
  --name python3 --display-name "Python ${PYTHON_VERSION} (Spark NLP)" >/dev/null

# Kernel env vars have to travel in the kernelspec: anything exported by this
# script dies with it, and the kernel is relaunched by Colab, not by us.
"$CONDA_PY" - "$KERNEL_DIR" "$JAVA_HOME" "$CONDA_PY" "$KERNEL_NAME" python3 <<'PYEOF'
import json
import os
import sys

kernel_dir, java_home, python_bin = sys.argv[1:4]
for name in sys.argv[4:]:
    path = os.path.join(kernel_dir, name, "kernel.json")
    with open(path) as f:
        spec = json.load(f)
    spec.setdefault("env", {}).update({
        "JAVA_HOME": java_home,
        "PYSPARK_PYTHON": python_bin,
        "PYSPARK_DRIVER_PYTHON": python_bin,
    })
    with open(path, "w") as f:
        json.dump(spec, f, indent=1)
    print("       kernelspec  : %s" % path)
PYEOF

# --------------------------------------------------------- Launcher shim ---
# The actual redirect. Colab execs "/usr/bin/python3 -m colab_kernel_launcher",
# so replacing that module makes the kernel Colab talks to start under the new
# interpreter. The shim re-execs the same launcher, passing the connection file
# straight through, and exports the Spark variables the kernel needs (the
# kernelspec "env" block is not honoured either, for the same reason argv is
# not). If the new interpreter has disappeared, it falls back to running the
# original launcher so the runtime still comes up.
LAUNCHER_BACKUP="${SYS_LAUNCHER%.py}.sparknlp-original.py"
if [ ! -f "$LAUNCHER_BACKUP" ]; then
  cp "$SYS_LAUNCHER" "$LAUNCHER_BACKUP"
fi

cat > "$SYS_LAUNCHER" <<EOF
"""Redirects Colab's default kernel onto Python ${PYTHON_VERSION}.

Installed by scripts/colab_setup_python.sh from the Spark NLP repository.
The launcher this replaced is kept next to it as
${LAUNCHER_BACKUP##*/} and is used if the interpreter below is gone.
"""

import os
import runpy
import sys

_PYTHON = "${CONDA_PY}"
_ORIGINAL = "${LAUNCHER_BACKUP}"

os.environ.setdefault("JAVA_HOME", "${JAVA_HOME}")
os.environ.setdefault("PYSPARK_PYTHON", _PYTHON)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", _PYTHON)

if os.access(_PYTHON, os.X_OK):
    try:
        # execve, not execv: pass the environment explicitly so the Spark
        # variables above are guaranteed to reach the kernel process.
        os.execve(_PYTHON, [_PYTHON, "-m", "colab_kernel_launcher"] + sys.argv[1:], os.environ)
    except Exception:
        pass  # anything at all goes wrong -> run the original launcher instead

runpy.run_path(_ORIGINAL, run_name="__main__")
EOF
echo "       launcher    : ${SYS_LAUNCHER} -> ${CONDA_PY}"
echo "       original    : ${LAUNCHER_BACKUP}"

cat <<EOF

==================================================================
 Done. NOW RESTART THE RUNTIME:  Runtime > Restart session
==================================================================
 The default "Python 3" kernel comes back on Python ${PYTHON_VERSION} with
 PySpark ${PYSPARK} and Spark NLP ${SPARKNLP} already installed. Colab's
 preinstalled 3.13 packages are NOT available to it - pip install
 whatever else you need after the restart.

 Paste this into the first cell after restarting to confirm:

   import sys, pyspark, sparknlp
   print(sys.version)
   print("pyspark", pyspark.__version__, "| spark-nlp", sparknlp.version())
   spark = sparknlp.start()
   print(spark.version)

 If the runtime never reconnects: Runtime > Disconnect and delete runtime.
==================================================================
EOF
