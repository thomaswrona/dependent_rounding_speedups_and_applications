SETUP EXAMPLE FOR WSL

get dependencies for c++ code

g++ -pthread -std=c++17 -g ./cpp_src/round_util.cpp ./cpp_src/dependent.cpp ./cpp_src/standard.cpp ./cpp_src/stochastic.cpp -shared -fPIC -O3 -o ./dependent.so



get dependencies for python code

pip install .

or, for editing mode,

pip install -e .



STRUCTURE



absl-py                      2.1.0
anyio                        4.3.0
argon2-cffi                  23.1.0
argon2-cffi-bindings         21.2.0
arrow                        1.3.0
asttokens                    2.4.1
astunparse                   1.6.3
async-lru                    2.0.4
attrs                        23.2.0
Babel                        2.14.0
beautifulsoup4               4.12.3
bitarray                     2.8.3
bleach                       6.1.0
blinker                      1.4
cachetools                   5.3.2
certifi                      2024.2.2
cffi                         1.16.0
charset-normalizer           3.3.2
comm                         0.2.1
command-not-found            0.3
contourpy                    1.2.0
cryptography                 3.4.8
cycler                       0.12.1
dbus-python                  1.2.18
debtcollector                3.0.0
debugpy                      1.8.1
decorator                    5.1.1
defusedxml                   0.7.1
dependent-rounding           1.0             /home/twrona/dependent_rounding
distro                       1.7.0
distro-info                  1.1+ubuntu0.2
exceptiongroup               1.2.0
executing                    2.0.1
fastjsonschema               2.19.1
flatbuffers                  24.3.25
fonttools                    4.49.0
fqdn                         1.5.1
gast                         0.4.0
google-auth                  2.27.0
google-auth-oauthlib         0.4.6
google-pasta                 0.2.0
grpcio                       1.60.1
h11                          0.14.0
h5py                         3.10.0
httpcore                     1.0.4
httplib2                     0.20.2
httpx                        0.27.0
idna                         3.6
image-utils                  0.1.6
imageio                      2.34.0
importlib-metadata           4.6.4
ipykernel                    6.29.2
ipython                      8.22.1
ipywidgets                   8.1.2
iso8601                      2.1.0
isoduration                  20.11.0
jedi                         0.19.1
jeepney                      0.7.1
Jinja2                       3.1.3
joblib                       1.3.2
json5                        0.9.17
jsonpointer                  2.4
jsonschema                   4.21.1
jsonschema-specifications    2023.12.1
jupyter                      1.0.0
jupyter_client               8.6.0
jupyter-console              6.6.3
jupyter_core                 5.7.1
jupyter-events               0.9.0
jupyter-lsp                  2.2.2
jupyter_server               2.12.5
jupyter_server_terminals     0.5.2
jupyterlab                   4.1.2
jupyterlab_pygments          0.3.0
jupyterlab_server            2.25.3
jupyterlab_widgets           3.0.10
keras                        2.10.0
Keras-Preprocessing          1.1.2
keyring                      23.5.0
kiwisolver                   1.4.5
launchpadlib                 1.10.16
lazr.restfulclient           0.14.4
lazr.uri                     1.0.6
lazy_loader                  0.3
libclang                     16.0.6
Markdown                     3.5.2
markdown-it-py               3.0.0
MarkupSafe                   2.1.5
matplotlib                   3.8.3
matplotlib-inline            0.1.6
mdurl                        0.1.2
mistune                      3.0.2
ml-dtypes                    0.4.1
more-itertools               8.10.0
namex                        0.0.8
nbclient                     0.9.0
nbconvert                    7.16.1
nbformat                     5.9.2
nest-asyncio                 1.6.0
netaddr                      1.2.1
netifaces                    0.11.0
networkx                     3.2.1
notebook                     7.1.0
notebook_shim                0.2.4
numpy                        1.26.2
oauthlib                     3.2.0
opt-einsum                   3.3.0
optree                       0.13.0
oslo.i18n                    6.3.0
oslo.utils                   7.1.0
overrides                    7.7.0
packaging                    23.2
pandas                       2.2.3
pandocfilters                1.5.1
parso                        0.8.3
pbr                          6.0.0
pexpect                      4.9.0
pillow                       10.2.0
pip                          22.0.2
platformdirs                 4.2.0
progressbar                  2.5
prometheus_client            0.20.0
prompt-toolkit               3.0.43
protobuf                     3.19.6
psutil                       5.9.8
ptyprocess                   0.7.0
pure-eval                    0.2.2
pyasn1                       0.5.1
pyasn1-modules               0.3.0
pycparser                    2.21
Pygments                     2.17.2
PyGObject                    3.42.1
PyJWT                        2.3.0
pyparsing                    2.4.7
python-apt                   2.4.0+ubuntu2
python-cephlibs              0.94.5.post1
python-dateutil              2.8.2
python-json-logger           2.0.7
pytz                         2024.2
PyYAML                       5.4.1
pyzmq                        25.1.2
qtconsole                    5.5.1
QtPy                         2.4.1
referencing                  0.33.0
requests                     2.31.0
requests-oauthlib            1.3.1
rfc3339-validator            0.1.4
rfc3986-validator            0.1.1
rich                         13.9.2
rpds-py                      0.18.0
rsa                          4.9
scikit-build                 0.17.6
scikit-image                 0.22.0
scikit-learn                 1.4.0
scipy                        1.12.0
SecretStorage                3.3.1
Send2Trash                   1.8.2
setuptools                   59.6.0
six                          1.16.0
sniffio                      1.3.0
soupsieve                    2.5
stack-data                   0.6.3
systemd-python               234
tensorboard                  2.10.1
tensorboard-data-server      0.6.1
tensorboard-plugin-wit       1.8.1
tensorflow                   2.17.0
tensorflow-cpu               2.10.0
tensorflow-directml-plugin   0.4.0.dev230202
tensorflow-estimator         2.10.0
tensorflow-io-gcs-filesystem 0.36.0
termcolor                    2.4.0
terminado                    0.18.0
threadpoolctl                3.3.0
tifffile                     2024.2.12
tinycss2                     1.2.1
tomli                        2.0.1
tornado                      6.4
tqdm                         4.66.2
traitlets                    5.14.1
types-python-dateutil        2.8.19.20240106
typing_extensions            4.9.0
tzdata                       2024.1
ubuntu-advantage-tools       8001
ufw                          0.36.1
unattended-upgrades          0.1
uri-template                 1.3.0
urllib3                      2.2.0
wadllib                      1.3.6
wcwidth                      0.2.13
webcolors                    1.13
webencodings                 0.5.1
websocket-client             1.7.0
Werkzeug                     3.0.1
wheel                        0.37.1
widgetsnbextension           4.0.10
wrapt                        1.16.0
zipp                         1.0.0