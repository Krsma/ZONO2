# syntax=docker/dockerfile:1
FROM rust:1.96-slim-bookworm@sha256:e18a79fc84dfcfc3ab5ba72290398a644c135c97eaa881447fddc354ee4701a3 AS rust
FROM python:3.11-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89 AS native
COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup
ENV PATH="/usr/local/cargo/bin:${PATH}" RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo
RUN apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config libfontconfig1-dev libopenblas-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir maturin==1.14.1 numpy==2.4.6
COPY build/learning-native /native
WORKDIR /native
ENV RUSTC_BOOTSTRAP=kmeans CARGO_BUILD_JOBS=2 LD_PRELOAD=libopenblas.so.0
RUN maturin build --release --locked --offline --compatibility linux --out /wheels

FROM python:3.11-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libopenblas0-pthread libfontconfig1 && rm -rf /var/lib/apt/lists/*
ENV LD_PRELOAD=libopenblas.so.0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
COPY docker/learning/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock && pip install --no-cache-dir --no-deps torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
COPY --from=native /wheels /wheels
RUN pip install --no-cache-dir --no-deps /wheels/*.whl && rm -rf /wheels
RUN python -c "import pathlib,sysconfig; pathlib.Path(sysconfig.get_paths()['purelib'],'rlola_python_binding_pzr_provenance.py').write_text('BINDING_REVISION = \'01c92a2bfac58755e3b832bb0094816f3f36e1d1\'\nINTERPRETER_REVISION = \'2724b05ae6c62ed0df14f1401ed8db89472725a6\'\nBINDING_BUILD_PROFILE = \'release\'\n')"
WORKDIR /opt/pzr
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps --no-build-isolation . && python -m pip check
COPY notebooks /opt/pzr/notebooks
COPY tests /opt/pzr/tests
COPY tools/learning/check_notebooks.py /opt/pzr/check_notebooks.py
COPY tools/learning/container_entrypoint.py /opt/pzr/container_entrypoint.py
RUN useradd --create-home --uid 1000 researcher && mkdir -p /workspace && chown researcher:researcher /workspace
USER researcher
WORKDIR /workspace
EXPOSE 8888
ENTRYPOINT ["python", "/opt/pzr/container_entrypoint.py"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--ServerApp.root_dir=/workspace"]

FROM runtime AS course
COPY data/learning/clean148 /opt/pzr/data/clean148
ENV PZR_LEARNING_DATA=/opt/pzr/data
