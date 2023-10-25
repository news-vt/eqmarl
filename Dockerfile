# FROM tensorflow/tensorflow:2.11.0-jupyter

# # WORKDIR /workspaces/eqmarl

# # # Install all requirement files.
# # COPY pyproject.toml /workspaces/eqmarl/pyproject.toml
# # RUN pip3 install --upgrade pip && pip3 install -U .

# # RUN pip3 install --upgrade pip && pip3 install \
#     # cirq \
#     matplotlib \
#     numpy \
#     pandas \
#     qutip \
#     seaborn \
#     sympy \
#     tensorflow-quantum
#     # tensorflow==2.11.0



###############

FROM python:3.9

WORKDIR /workspaces/eqmarl
COPY requirements*.txt ./

RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
        -r requirements.txt \
        -r requirements-dev.txt
# RUN pip install \
#     # cirq \ # Included in tensorflow-quantum.
#     matplotlib \
#     numpy \
#     pandas \
#     qutip \
#     seaborn \
#     sympy \
#     gymnasium \
#     # tensorflow-quantum \ # Use tfq-nightly below.
#     tfq-nightly==0.7.3.dev20231004 \
#     tensorflow-cpu==2.11.0