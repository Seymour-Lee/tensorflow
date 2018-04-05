#!/usr/bin/env bash

set -e
set -o pipefail

bazel build --config=opt --incompatible_load_argument_is_label=false //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo pip uninstall tensorflow -y
sudo pip install /tmp/tensorflow_pkg/tensorflow-1.4.1-cp27-cp27m-macosx_10_13_intel.whl