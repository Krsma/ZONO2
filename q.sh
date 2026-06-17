source "$HOME/.cargo/env"
  cargo --version
  rustc --version

  CONDA_NO_PLUGINS=true external/miniconda3/bin/conda create --solver classic -y -n pzr-rtlola
  python=3.11
  CONDA_NO_PLUGINS=true external/miniconda3/bin/conda run -n pzr-rtlola python -m pip install --upgrade
  pip
  CONDA_NO_PLUGINS=true external/miniconda3/bin/conda run -n pzr-rtlola python -m pip install -e ".
  [dev,learning]" maturin numpy

  git submodule add git@projects.cispa.saarland:group-finkbeiner/tools/RTLola/rlolapythonbinding.git
  vendor/rlola-python-binding
  git -C vendor/rlola-python-binding checkout 72622a3

  CARGO_NET_GIT_FETCH_WITH_CLI=true CONDA_NO_PLUGINS=true \
    external/miniconda3/bin/conda run -n pzr-rtlola python -m maturin develop --release \
    --manifest-path vendor/rlola-python-binding/Cargo.toml

  CONDA_NO_PLUGINS=true external/miniconda3/bin/conda run -n pzr-rtlola python -c \
    "from rlola_python_binding import RLolaMonitor, ZonotopeConfig; print('rtlola binding ok')"