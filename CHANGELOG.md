# CHANGELOG


## v0.4.0 (2025-07-03)

### Chores

* chore: add dockerignore file ([`898b374`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/898b3740fbfc9e2dd77204645e916b1c514204b5))

### Continuous Integration

* ci: use named docker volume for submission ([`c847d91`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/c847d91226a1a373ee51492509fb7b3865df1931))

* ci: add cuda:1 to submission workflow ([`be57234`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/be5723404f126c3b8ff3185f97ec25e143426119))

* ci: remove cleanup of build folder ([`bb5b7a4`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/bb5b7a4d8c50e9c2bdf95ad75eb76d6d81803695))

* ci: use self hosted runner without docker ([`3f196f9`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/3f196f98847aaba10f089ab152ad1ae246c37958))

### Features

* feat: introduce ensembling class ([`3008984`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/30089840bfe87b6850100ce861e299acac621d27))

### Unknown

* Merge pull request #10 from Ahus-AIM/full-dataset

Full dataset ([`06bd575`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/06bd575fbcc70678b0ac63450c720143fa8e4b6e))


## v0.3.0 (2025-07-01)

### Bug Fixes

* fix: details for submission code ([`41108c4`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/41108c4f7747928baa2446ad6d93335921fb3f1b))

* fix: update to python 3.12 ([`8cea2d8`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/8cea2d82721240e9720dacf373998c2bdbda3f41))

* fix: do not run submission test on self hosted runner ([`2a71cde`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/2a71cde920569dd6158f3f612aa13e24e153d1ed))

* fix: compile model instead of train fn, make less verbose ([`b758b61`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/b758b6110594116e16121a214b76dd0ce1f0f761))

### Chores

* chore: ensure isort compatible with black ([`227ec04`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/227ec0401914d2ecb9a2c041d4a762a5c3b16f67))

* chore: add linting script ([`bc9b158`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/bc9b15887a33cab567393114d14a2f4942fe6d7e))

* chore: update requirements.txt with latest versions ([`48560e4`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/48560e45ca248fd91fe6a1a4262a1cd3621a29ab))

* chore: add gitignore ([`f607311`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/f607311ea3380ebff4950422efdd6a6543f40e7a))

### Code Style

* style: fix mypy complaints ([`6ce960e`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/6ce960e6372992007dccfd9a714d47b2061feda1))

* style: remove comments ([`dbcf1b9`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/dbcf1b9abed5a4ac230aad4fc9ad617bb584bd8d))

* style: lint team code ([`f65a0ab`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/f65a0ab07331b916aba41c3050cf69461e43cf6e))

* style: apply isort ([`efbdeca`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/efbdecae83ee3afab644df80ee31b9c6a47cf853))

### Features

* feat: bayesian hyperparameter search ([`46314ae`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/46314aea1536d42b58a0ec55dabd15f717c016ab))

* feat: add muon optimizer ([`bef2fe2`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/bef2fe2e3bf1079457cbe7eafc79b0141ace75b7))

* feat: add random cropping transform ([`a919410`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/a919410633c90f111de1848462fe270058faf140))

* feat: disable tqdm when using ray ([`4965ac8`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/4965ac8d064214867225bc878773b577e82f1f4f))

* feat: support for cross-validation ([`a7573ca`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/a7573caf7b28341ffda8db37f29c0e0c99de09d7))

* feat: update experiment plotting code ([`c87ce95`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/c87ce95bea06b1073259561688f6367490d967a2))

* feat: add data preparation script for SaMI-Trop ([`7627564`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/7627564b43d8f907a421ae92772d18cc60f70524))

### Refactoring

* refactor: separate yaml file for submission ([`dfbc4cc`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/dfbc4cc1df039dcb72e2210c90f110da5d7ecd11))

### Unknown

* Merge pull request #9 from Ahus-AIM/full-dataset

Full dataset ([`88a8d70`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/88a8d70e63e875e8a2e920c8c0be18296816c460))


## v0.2.2 (2025-02-20)

### Bug Fixes

* fix: return bool instead of str ([`3812992`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/38129929529a0e30a4076c0917dd35f2df82ec73))

* fix: do not crash on all zero input ([`da36fc4`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/da36fc41abf5d281f6cfe1805ee4d1b1ff6b0a44))

### Testing

* test: add edge case with all zeros ([`cd58826`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/cd588260c90baf078c3366049857fc84bb027e94))

### Unknown

* Merge pull request #8 from Ahus-AIM/padding_detection

Padding detection ([`07267d3`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/07267d39af18752aaa77a1d002de3fc338730b1d))


## v0.2.1 (2025-02-19)

### Bug Fixes

* fix: return text for binary output ([`02fa2ee`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/02fa2ee11ef5affe8d4d23e3cc7a7972e8dbc971))

* fix: do not crash on all zero signal ([`de6275c`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/de6275c006b6b923505e83e760d41b79d04614f9))

### Unknown

* Merge pull request #7 from Ahus-AIM/padding_detection

Padding detection ([`7d918dd`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/7d918ddb59d271dd69518d758c4c6302ec12346e))


## v0.2.0 (2025-02-18)

### Continuous Integration

* ci: set up submission workflow ([`3a1b5c4`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/3a1b5c4c8fb6b6301aa6af3e965af0407c1d000b))

### Features

* feat: implement resampling ([`cac5b83`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/cac5b83d97e965c322a306c8a725e16e221bbce5))

### Unknown

* Merge pull request #6 from Ahus-AIM/resampling

feat: implement resampling ([`d732501`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/d7325013423521d8b8623ca160f654b9f9fe1faa))

* Merge pull request #5 from Ahus-AIM/dockerize

Dockerize ([`62bd8c7`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/62bd8c7832cd379bdaaff2d7112f94eaf7169b10))


## v0.1.1 (2025-02-13)

### Bug Fixes

* fix: search for files recursively in wfdb dataset ([`b54a3e3`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/b54a3e339d883367dda40a23da5d84d8f3e2a8ac))

### Documentation

* docs:  add authors ([`b3460f3`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/b3460f384997c3a256344c06893c71e7610d94e5))

### Unknown

* Merge pull request #4 from Ahus-AIM/recursive_dataloading

fix: search for files recursively in wfdb dataset ([`bf5ccd5`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/bf5ccd559d1e6f898ec9b25c1052865a9ee46c51))

* Merge pull request #3 from Ahus-AIM/authors

docs:  add authors ([`289cb71`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/289cb7151199ab7c44687941a3b587c96e6deadc))


## v0.1.0 (2025-02-05)

### Features

* feat: training and first submission attempt ([`75f3855`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/75f38559b0be95239835415daddf1e49570aaedb))

### Unknown

* Merge pull request #2 from Ahus-AIM/training_code

feat: training and first submission attempt ([`ad17f07`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/ad17f0777c8c7986c1a0237c719ffd4087da62ee))


## v0.0.1 (2025-01-15)

### Bug Fixes

* fix: mypy ignore team code for now ([`3d31435`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/3d314353837bc10ad98b223cdac63733f585e83a))

### Chores

* chore: add script to prepare code15 data ([`97b29a2`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/97b29a2e129035902cdde2f27ef4e3188288a911))

* chore: set up semantic versioning ([`61ae594`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/61ae5945883a05c32e265015c6d0dba0f9ed997c))

* chore: update .flake8 to ignore challenge code ([`e987ee5`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/e987ee505f1b276e71c0643f32be04fe32d89f5d))

* chore: update .flake to ignore challenge code ([`58a3f70`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/58a3f704bcbfe6053e494e91a2b4a0f14aaffc50))

### Code Style

* style: black formatting ([`0838694`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/08386943224c41bb5ba517981c72b02f3bb7718d))

* style: remove unused imports and lint ([`88a9b41`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/88a9b4127836edce40eb4a5dc787df0ca80cc443))

### Continuous Integration

* ci: typecheck python with mypy ([`7119fe2`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/7119fe2e567d2d28720778b6ce275ab0647e451b))

* ci: lint python and commit ([`93445fd`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/93445fd64306bdfb53b421e38597cd8cba8392e5))

### Unknown

* Merge pull request #1 from Ahus-AIM/workflow

Workflow ([`97d3e4e`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/97d3e4e8ea57ec63ee73b6a6ffb2bf418438088a))

* Fix broken link/data ([`832afb3`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/832afb3c202a3a20532e230106e574d7bd709c83))

* Initial commit for 2025 Challenge ([`e966bca`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/e966bcac7cbe730c72c7199e3ae8620228b116dd))
