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
#   Miniconda for the target Python is installed over /usr/local, the packages
#   Colab's frontend needs to talk to a kernel (jupyter, google-colab,
#   traitlets) are reinstalled there, and a Jupyter kernelspec named "python3"
#   is written to ~/.local/share/jupyter/kernels. That user-level kernelspec
#   shadows Colab's system one, so restarting the runtime relaunches the
#   default kernel on the new interpreter - no notebook metadata edits needed.
#   Technique adapted from https://github.com/j3soon/colab-python-version
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
#   script did. (The kernelspec alone can be undone with
#   "rm -rf ~/.local/share/jupyter/kernels/python3", but only from a kernel
#   that still starts.)

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

# Fail before touching anything if the requested Miniconda build does not exist.
echo "[1/6] Checking ${MINICONDA_INSTALLER} is available..."
if ! wget -q --spider "$MINICONDA_URL"; then
  echo "Error: ${MINICONDA_URL} not found." >&2
  echo "       Pick an existing build with -m (see https://repo.anaconda.com/miniconda/)." >&2
  exit 1
fi

# ---------------------------------------------------------------- Java -----
echo "[2/6] Java..."
install_openjdk11() {
  apt-get update -qq >/dev/null 2>&1 || true
  apt-get install -y -qq openjdk-11-jdk >/dev/null 2>&1
}

if ! type -p java >/dev/null 2>&1; then
  echo "       Java not found. Installing OpenJDK 11..."
  install_openjdk11
else
  JAVA_MAJOR="$(java -version 2>&1 | head -1 | sed -E 's/[^"]*"([0-9]+)\.?([0-9]*).*/\1/')"
  # "1.8.0_xxx" style version strings report major 1; Spark cares about the 8.
  if [ "$JAVA_MAJOR" = "1" ]; then
    JAVA_MAJOR=8
  fi
  # Spark 3.x supports Java 8/11/17 only; anything newer needs a downgrade too.
  if [ "${JAVA_MAJOR:-0}" -gt 17 ] 2>/dev/null; then
    echo "       Java ${JAVA_MAJOR} is too new for Spark 3.x. Installing OpenJDK 11..."
    install_openjdk11
  fi
fi

if [ -d /usr/lib/jvm/java-11-openjdk-amd64 ]; then
  JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
else
  JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(type -p java)")")")"
fi
export JAVA_HOME
export PATH="$PATH:$JAVA_HOME/bin"
echo "       JAVA_HOME=${JAVA_HOME}"

if [[ "$GPU" == "true" ]]; then
  echo "       Upgrading libcudnn8 to 8.1.0 for GPU"
  apt install -qq --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2 -y &>/dev/null
fi

# ------------------------------------------------------------ Interpreter --
# Installs over /usr/local so /usr/local/bin/python becomes the target version.
# Colab's own dist-packages live under /usr/local/lib/python3.13 and are left in
# place, so the kernel running this cell keeps working until you restart.
echo "[3/6] Installing Python ${PYTHON_VERSION} (${MINICONDA_INSTALLER})..."
wget -q -O /tmp/miniconda.sh "$MINICONDA_URL"
bash /tmp/miniconda.sh -b -f -p /usr/local >/dev/null
rm -f /tmp/miniconda.sh

# The packages Colab's frontend needs in order to drive the kernel. The
# traitlets pin works around an incompatibility between google-colab 1.0.0 (the
# version the Colab runtime itself ships) and newer traitlets releases.
echo "[4/6] Installing Colab kernel packages (jupyter, google-colab, traitlets=${TRAITLETS})..."
/usr/local/bin/conda install -q -y -c conda-forge jupyter google-colab "traitlets=${TRAITLETS}" >/dev/null

# --------------------------------------------------------------- Payload ---
echo "[5/6] Installing PySpark ${PYSPARK}, Spark NLP ${SPARKNLP}, findspark..."
"$CONDA_PY" -m pip install --upgrade -q "pyspark==${PYSPARK}" "spark-nlp==${SPARKNLP}" findspark

# ------------------------------------------------------------- Pre-flight --
# Everything above is additive and undone by discarding the runtime. The
# kernelspec below is what actually redirects Colab, so verify the new
# interpreter can do what the kernel needs before writing it. This is the check
# that catches the 3.12 "kernel_class ... could not be imported" failure
# *before* it can leave you with a runtime that never reconnects.
echo "[6/6] Pre-flight check on the new interpreter..."
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
# --user writes to ~/.local/share/jupyter/kernels, which Jupyter searches before
# the system directories, so a spec named "python3" shadows Colab's own without
# deleting it. The versioned name is registered too, for notebooks that pin
# metadata.kernelspec.name explicitly.
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
