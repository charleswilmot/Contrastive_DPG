# Contrastive Determinitic Policy Gradient

Custom DPG algorithm, where the critic's estimated return is maximized in parallel by N actors, which are subject to a contrastive loss.

The implementation is based on

- jax
- rlax
- haiku
- optax

Visualization is made with

- tensorboardX
- moviepy

The robotic simulation software:

- CoppeliaSim 4.3.0
- PyRep


# pip freeze

```
absl-py @ file:///tmp/build/80754af9/absl-py_1623867230185/work
aiohttp @ file:///tmp/build/80754af9/aiohttp_1614360992924/work
argon2-cffi==21.1.0
arxiv==1.4.2
astor==0.8.1
astunparse==1.6.3
async-timeout==3.0.1
attrs @ file:///tmp/build/80754af9/attrs_1620827162558/work
backcall==0.2.0
bcrypt @ file:///tmp/build/80754af9/bcrypt_1597936221426/work
bleach==4.1.0
blinker==1.4
box2d-py==2.3.8
brotlipy==0.7.0
cachetools @ file:///tmp/build/80754af9/cachetools_1619597386817/work
certifi==2021.5.30
cffi==1.14.2
chardet @ file:///tmp/build/80754af9/chardet_1605303185383/work
charset-normalizer @ file:///tmp/build/80754af9/charset-normalizer_1630003229654/work
chex==0.0.8
click @ file:///tmp/build/80754af9/click_1621604852318/work
cloudpickle @ file:///Users/ktietz/demo/mc3/conda-bld/cloudpickle_1629142150447/work
coverage @ file:///tmp/build/80754af9/coverage_1614613670853/work
crc32c==2.2.post0
cryptography @ file:///tmp/build/80754af9/cryptography_1616769286105/work
cycler==0.10.0
Cython @ file:///tmp/build/80754af9/cython_1626256955500/work
cytoolz==0.11.0
dask @ file:///tmp/build/80754af9/dask-core_1630523592567/work
debugpy==1.4.3
decorator==4.4.2
defusedxml==0.7.1
dill @ file:///tmp/build/80754af9/dill_1623919422540/work
distrax==0.0.2
dm-haiku==0.0.5
dm-tree==0.1.6
entrypoints==0.3
feedparser==6.0.8
flatbuffers==2.0
flax==0.3.5
fsspec @ file:///tmp/build/80754af9/fsspec_1626383727127/work
future==0.18.2
gast==0.3.3
google-auth @ file:///tmp/build/80754af9/google-auth_1626320605116/work
google-auth-oauthlib @ file:///tmp/build/80754af9/google-auth-oauthlib_1617120569401/work
google-pasta @ file:///Users/ktietz/demo/mc3/conda-bld/google-pasta_1630577991354/work
googleapis-common-protos @ file:///tmp/build/80754af9/googleapis-common-protos-feedstock_1617957649450/work
grpcio @ file:///tmp/build/80754af9/grpcio_1614884175859/work
h5py @ file:///tmp/build/80754af9/h5py_1593454122442/work
idna @ file:///tmp/build/80754af9/idna_1622654382723/work
imageio @ file:///tmp/build/80754af9/imageio_1617700267927/work
imageio-ffmpeg==0.4.5
importlib-metadata @ file:///tmp/build/80754af9/importlib-metadata_1617874469820/work
ipykernel==6.4.1
ipython==7.27.0
ipython-genutils==0.2.0
ipywidgets==7.6.5
jax==0.2.27
jaxlib==0.1.75
jedi==0.18.0
Jinja2==3.0.1
jmp==0.0.2
jsonschema==3.2.0
jupyter==1.0.0
jupyter-client==7.0.3
jupyter-console==6.4.0
jupyter-core==4.8.1
jupyterlab-pygments==0.1.2
jupyterlab-widgets==1.0.2
keras==2.7.0
Keras-Preprocessing @ file:///tmp/build/80754af9/keras-preprocessing_1612283640596/work
kiwisolver @ file:///tmp/build/80754af9/kiwisolver_1612282420641/work
libclang==13.0.0
locket==0.2.1
Markdown @ file:///tmp/build/80754af9/markdown_1614363528767/work
MarkupSafe==2.0.1
matplotlib @ file:///tmp/build/80754af9/matplotlib-base_1603378225747/work
matplotlib-inline==0.1.3
mistune==0.8.4
mkchromecast==0.3.9
mkl-fft==1.3.0
mkl-random @ file:///tmp/build/80754af9/mkl_random_1626186064646/work
mkl-service==2.4.0
moviepy==1.0.3
mpld3==0.5.5
mpmath==1.1.0
msgpack==1.0.2
mujoco-py==2.0.2.10
multidict @ file:///tmp/build/80754af9/multidict_1607367757617/work
nbclient==0.5.4
nbconvert==6.1.0
nbformat==5.1.3
nest-asyncio==1.5.1
networkx @ file:///tmp/build/80754af9/networkx_1627459939258/work
notebook==6.4.4
numpy @ file:///tmp/build/80754af9/numpy_and_numpy_base_1626271506491/work
oauthlib @ file:///tmp/build/80754af9/oauthlib_1623060228408/work
olefile @ file:///Users/ktietz/demo/mc3/conda-bld/olefile_1629805411829/work
opencv-python==4.2.0.34
opt-einsum @ file:///tmp/build/80754af9/opt_einsum_1621500238896/work
optax==0.0.9
packaging @ file:///tmp/build/80754af9/packaging_1625611678980/work
pandas==1.2.0
pandocfilters==1.5.0
paramiko @ file:///tmp/build/80754af9/paramiko_1598886428689/work
parso==0.8.2
partd @ file:///tmp/build/80754af9/partd_1618000087440/work
pexpect==4.8.0
pickleshare==0.7.5
Pillow @ file:///tmp/build/80754af9/pillow_1625655817137/work
proglog==0.1.9
prometheus-client==0.11.0
promise @ file:///tmp/build/80754af9/promise_1614011636525/work
prompt-toolkit==3.0.20
protobuf==3.17.2
psutil @ file:///tmp/build/80754af9/psutil_1612298023621/work
ptyprocess==0.7.0
pyasn1 @ file:///Users/ktietz/demo/mc3/conda-bld/pyasn1_1629708007385/work
pyasn1-modules==0.2.8
pycparser @ file:///tmp/build/80754af9/pycparser_1594388511720/work
pydub==0.24.1
Pygments==2.10.0
PyJWT @ file:///tmp/build/80754af9/pyjwt_1619651636675/work
PyNaCl @ file:///tmp/build/80754af9/pynacl_1595009131182/work
pyOpenSSL @ file:///tmp/build/80754af9/pyopenssl_1608057966937/work
pyparsing @ file:///home/linux1/recipes/ci/pyparsing_1610983426697/work
PyRep @ file:///home/cwilmot/Software/PyRep
pyrsistent==0.18.0
PySocks @ file:///tmp/build/80754af9/pysocks_1605305779399/work
python-dateutil @ file:///tmp/build/80754af9/python-dateutil_1626374649649/work
pytz==2020.5
PyWavelets @ file:///tmp/build/80754af9/pywavelets_1601658317819/work
PyYAML==5.4.1
pyzmq==22.3.0
qtconsole==5.1.1
QtPy==1.11.1
requests @ file:///tmp/build/80754af9/requests_1629994808627/work
requests-oauthlib==1.3.0
rlax==0.0.4
rsa @ file:///tmp/build/80754af9/rsa_1614366226499/work
scikit-image==0.17.2
scipy @ file:///tmp/build/80754af9/scipy_1630606796110/work
selenium==3.141.0
Send2Trash==1.8.0
sgmllib3k==1.0.0
sip==4.19.13
six @ file:///tmp/build/80754af9/six_1623709665295/work
sounddevice==0.4.2
sympy==1.6.2
tabulate==0.8.9
tensorboard==2.8.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.6.0
tensorboardX==2.4.1
tensorflow==2.7.0
tensorflow-datasets @ file:///tmp/build/80754af9/tensorflow-datasets_1615394951098/work
tensorflow-estimator==2.7.0
tensorflow-io-gcs-filesystem==0.23.1
tensorflow-metadata==0.14.0
tensorflow-probability==0.15.0
termcolor==1.1.0
terminado==0.12.1
testpath==0.5.0
tifffile==2020.10.1
tikzplotlib==0.9.6
toolz @ file:///home/linux1/recipes/ci/toolz_1610987900194/work
tornado @ file:///tmp/build/80754af9/tornado_1606942300299/work
tqdm @ file:///tmp/build/80754af9/tqdm_1629302309755/work
traitlets==5.1.0
typing-extensions @ file:///tmp/build/80754af9/typing_extensions_1624965014186/work
urllib3 @ file:///tmp/build/80754af9/urllib3_1625084269274/work
wcwidth==0.2.5
webencodings==0.5.1
Werkzeug @ file:///home/ktietz/src/ci/werkzeug_1611932622770/work
widgetsnbextension==3.5.1
wrapt==1.12.1
xlrd==1.2.0
yarl @ file:///tmp/build/80754af9/yarl_1606939922162/work
zipp @ file:///tmp/build/80754af9/zipp_1625570634446/work
```
