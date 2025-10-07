# CHANGELOG


## v0.14.0 (2025-10-07)

### Documentation

* docs: add preprint and poster ([`597100a`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/597100a8ce6edd48e02a418e51055c595599e329))

### Features

* feat: add paper and poster with test scores ([`71e64e1`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/71e64e1e3f85e12b1dc1d8da09c968e52d04ddc4))

### Unknown

* Merge pull request #24 from Ahus-AIM/new-docs

feat: add paper and poster with test scores ([`ea9828f`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/ea9828f89ab66a597c497f0689e8913cd0d2511e))


## v0.13.2 (2025-09-14)

### Bug Fixes

* fix: replace weights for hackathon submission 2 ([`52d949f`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/52d949f9b940429c879e19d590f3879914a9a27d))

### Unknown

* Merge pull request #23 from Ahus-AIM/hackathon-2

fix: replace weights for hackathon submission 2 ([`cfb1569`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/cfb1569d57cc4cbc4a7ea7e7f40bdf4df41fb74c))


## v0.13.1 (2025-09-14)

### Bug Fixes

* fix: decrease learning rate for ensemble ([`785d7f2`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/785d7f20c12a787f4529ac22ea21599370a0fd72))

* fix: replace weights for hackathon 1 ([`1184676`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/118467691c95086552b45d9537c59a257f31ede9))

### Unknown

* Merge pull request #22 from Ahus-AIM/hackathon-1

Reduce LR for ensemble training, initialize training with weights obtained by pre-training on diagnostic codes from MIMIC-IV. ([`cc839f2`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/cc839f2d4af518f55c19afd51305abd0aa960c64))


## v0.13.0 (2025-08-20)

### Bug Fixes

* fix: update configs for submission 10 ([`ee91f6f`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/ee91f6f641686b3ee3e3dde5fa68f2b8e6f0df21))

### Features

* feat: update weights for submission 10 ([`b152ad8`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/b152ad89a48e9dd651e54d697b48f9d626f73c95))

* feat: add constant to logits pre-sigmoid ([`0d92e4c`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/0d92e4ce095ab6802ee994183850d9975440575c))

* feat: add random down-then-upsampling transform ([`922a686`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/922a686aaddb21704e60b83cb5a41da4307fb3cc))

### Unknown

* Merge pull request #21 from Ahus-AIM/submission-10

Submission 10 ([`1e646ce`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/1e646cee9a264ad6a3b335e800d27a7dd80892f3))


## v0.12.0 (2025-08-15)

### Bug Fixes

* fix: set lr=1e-5 for ensemble training ([`f19583e`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/f19583e54c19386be3c5b4cd1ddc595f736fdaa4))

### Features

* feat: use 20 chunks per model in inference ([`3a7e12a`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/3a7e12a399b8a1784eee62060f780a26bdff1819))

* feat: update train config ([`981f48a`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/981f48a6993d323a73540c49529370a5559f346c))

* feat: update weights for submission 9 ([`3b41119`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/3b4111928cf0c050729dbfb1c9c1ba5caf55ae7a))

* feat: add transform for ahus dataset conversion ([`b131f95`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/b131f9535657fca7bd7219187cc0ae225de880f1))

### Unknown

* Merge pull request #20 from Ahus-AIM/submission-9

Submission 9 ([`cb4dbec`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/cb4dbecfac7c911e48c6f3d4d7332a1223746c19))


## v0.11.0 (2025-08-11)

### Bug Fixes

* fix: update weights for submission 8 ([`b1e94cf`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/b1e94cf141c429c88ebb0fa51b8fcaf558c59645))

### Features

* feat: add normalization again ([`c6e76f6`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/c6e76f61447476847a7df61114bc2e60dde7e0e9))

* feat: update configs for submission 8 ([`756881f`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/756881f18ba335e9cc5fbc84cbf50d35ffd20b82))

### Unknown

* Merge pull request #19 from Ahus-AIM/submission-8

Submission 8 ([`6e74830`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/6e74830703ae57ba1328c768fae7ebaf8736de91))


## v0.10.0 (2025-08-04)

### Continuous Integration

* ci: do not fetch full history ([`fc2d307`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/fc2d307cef62dcf715505ae39d75aa07a6b1c666))

### Features

* feat: update code for submission 7, no z-normalization ([`253200e`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/253200e03bcd8dd9276650b760276ba827f06b92))

* feat: replace weights for submission 7 ([`3e8ea3c`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/3e8ea3c2d13e004976301984d19890d942776043))

### Unknown

* Merge pull request #18 from Ahus-AIM/submission-7

Submission 7 ([`c5f619b`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/c5f619bc7c83532e21207d793810846af2da1b61))

* Merge pull request #17 from Ahus-AIM/submission-6-fix

ci: do not fetch full history ([`17c0991`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/17c0991b50164d2b19a1d266c3aab33e637719ea))


## v0.9.0 (2025-08-03)

### Bug Fixes

* fix: use grid search instead of bayesian search ([`923c262`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/923c26232d2fbae2c1fe361c413f21362a353f82))

* fix: update weights for submission 6 ([`d759973`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/d7599733b2a3cf958d42afe6af0df25edb1d35f4))

* fix: update config files for submission 6 ([`63741f9`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/63741f9b0f63764df52a4500cb7ce4bac8b0d298))

### Documentation

* docs: update readme ([`8a44538`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/8a44538e31f7eb694634d1450f09bcdb913118a7))

### Features

* feat: add reweighting for patients with multiple ECGs ([`0ed843a`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/0ed843a77b3bbbd2367d6df20db8ea4bdc77b7e3))

* feat: use 2 second crops for submission 6 ([`271c471`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/271c4717dac50009c8670901812177689c1aa364))

### Refactoring

* refactor: simplify calculation for challenge metric ([`88470dd`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/88470ddb093e8fdbf1b7823b8e8385bec8414184))

### Unknown

* Merge pull request #16 from Ahus-AIM/submission-6

Submission 6 ([`c3e4ff1`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/c3e4ff1cbdbe00b4f1dde357801e564f96fd457b))


## v0.8.0 (2025-07-22)

### Bug Fixes

* fix: update weights for submission 5 ([`60e718e`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/60e718e6aa1e9db9c113b72be8b01a62477bd4dd))

* fix: only update one ensemble member per batch ([`d607834`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/d60783466007099828552dddb6c8964f0d0a5ff5))

* fix: add fallback in case transforms are imported from notebooks ([`7a18677`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/7a18677866382bae9ccb05127490d8e65f9f94b0))

* fix: z-normalize signals in inference ([`d583e71`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/d583e7159d73c585137b9e7928ac9b97722a1371))

* fix: update train configs for submission 5 ([`d82a688`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/d82a688adef864c7384384268939375bd79f4e2f))

### Features

* feat: add support for demographics in dataloader ([`426b4ae`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/426b4aeaa33e75a6b128269dde66a23e8c869c30))

### Unknown

* Merge pull request #15 from Ahus-AIM/submission-5

Submission 5 ([`9b553cb`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/9b553cbadc9c25e60347751263dfb718740b04c3))


## v0.7.0 (2025-07-19)

### Bug Fixes

* fix: update weights for submission 4 ([`74a29d1`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/74a29d18b8fd7354321eb30e00871f9b1dd15a1f))

* fix: add try catch in plotting script ([`1ce2f9b`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/1ce2f9bd32aea758a7fdc82b74c45a7123fb6496))

* fix: update search space for bloodtest pretraining ([`e9080a4`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/e9080a4705bfe29f28ef2f731d14c69e80c8fbbe))

* fix: reduce lr for finetuning ([`57161cc`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/57161ccd67c5005d409a5e7de043a09f28bf1ef3))

### Documentation

* docs: update licence ([`27e91ad`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/27e91ad7d7998c269be21500aa11058f243a5da6))

* docs: update readme ([`cb22bd8`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/cb22bd8a25c5bc9d9ce5eb0174644dd333f9c510))

### Features

* feat: add magnitude scaling transform ([`8f986e0`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/8f986e0c13345589e5e7f0874ac22901819df001))

* feat: add tabulate for ray run summary ([`f2c8f2d`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/f2c8f2d366e4d1c2b6da8d94439f651c5094e870))

### Unknown

* Merge pull request #14 from Ahus-AIM/submission-4

Submission 4 ([`f2936b4`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/f2936b48b4ad80f683eeff34d1a637b4cdfcccd9))


## v0.6.0 (2025-07-17)

### Bug Fixes

* fix: update weights for submission 3 ([`a135a21`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/a135a21cbbad9f0cfd7141e0f331d6dd51697272))

### Features

* feat: ensure training script works for both pretraining and finetuning ([`344cb70`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/344cb700cb113063fd607e036b70a0fa7fb125ef))

* feat: update the post ray run plotting script ([`7f54220`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/7f54220d717616161b7487577143283e3eeaedc6))

* feat: add script for fetching best weights after pretraining run ([`b88deac`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/b88deac61a0af88709f7b78114001420b5622849))

* feat: add weight smoothing for bloodtest pretraining ([`d0e8b3f`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/d0e8b3fd25e343be59b4017d17e74e0046b044c0))

* feat: update inference code for submission 3 ([`cad7d26`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/cad7d2697ec95ca03f2a1829aa08c417e2ba76e2))

### Unknown

* Merge pull request #13 from Ahus-AIM/submission-3

Submission 3 ([`3278f5f`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/3278f5f87ca969b8999d36453d26f5da8c013fbd))


## v0.5.0 (2025-07-14)

### Features

* feat: enable config argparsing and save weights if not using ray ([`1fc8211`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/1fc8211bcdea5ad1aecc676e76c8c38eff4bb7ba))

* feat: enable bloodtest pretraining ([`089b6fa`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/089b6fad51e430d6e2238c6172224b38066c9cb6))

* feat: update model weights to blood-pretrained ([`aee9703`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/aee9703cef6c210fa96f8d97740f6c5a64facd47))

### Refactoring

* refactor: remove legacy dataset ([`1929c44`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/1929c44471be473fbaf92f81387cdb04ff11ecea))

### Unknown

* Merge pull request #12 from Ahus-AIM/bloodtest-pretraining

Bloodtest pretraining ([`63feaa8`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/63feaa889a26cb77ccdde1f6ab3707e34a99c374))


## v0.4.1 (2025-07-06)

### Bug Fixes

* fix: create directories if they do not already exist ([`ab509ce`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/ab509ced4a45db776b92822195044b3e27ec5aa8))

* fix: bug where inference fails if signal has exactly 300 hz sample rate ([`9de2e79`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/9de2e797ef1928f95001e004304331947c3c2e79))

### Testing

* test: add test case with exactly 2x highpass frequency ([`5a03ce5`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/5a03ce511c183fdbb106d5a33671655cefb192ca))

### Unknown

* Merge pull request #11 from Ahus-AIM/bugfix-official-submission-one

Bugfix after first failed submission ([`1de337a`](https://github.com/Ahus-AIM/physionet-challenge-2025/commit/1de337acffa4255eba0cdc5b70fdc6e2644fd929))


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
