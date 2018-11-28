# url-generation
RNN Generation of phishy/non-phishy URL:s.

# Installation
```bash
git clone git@github.com:andhus/url-generation.git
cd url-generation
pip install .
```

# Usage
If you just want to sample some URL:s, go directly to the [sampling](#sampling) 
section.

## Training
```bash
python scripts/train.py --help
```
```
usage: train.py [-h] [--job-name JOB_NAME] [-o TARGET_DIR]
                [--batch-size BATCH_SIZE] [--epochs EPOCHS]
                [--sample-fraction SAMPLE_FRACTION] [--units UNITS]
                [--embedding-size EMBEDDING_SIZE]
                [--url-cap-length URL_CAP_LENGTH]

Train a Recurrent Neural Network to generate phishy and non-phishy URL:s.

optional arguments:
  -h, --help            show this help message and exit
  --job-name JOB_NAME   name of the training job
  -o TARGET_DIR, --target-dir TARGET_DIR
                        directory to write output to (model checkpoints etc.)
  --batch-size BATCH_SIZE
                        batch size to use in training
  --epochs EPOCHS       number of epochs to use in training
  --sample-fraction SAMPLE_FRACTION
                        use only a sub sample of data (useful for debugging)
  --units UNITS         state size of the RNN
  --embedding-size EMBEDDING_SIZE
                        size of character embeddings
  --url-cap-length URL_CAP_LENGTH
                        cap URL:s at this length during training
```
Leave all arguments empty for default settings (job results are found in 
`./results/DefaultJob/`), the training data is automatically downloaded and training 
takes about 1 hour on a modern laptop. 

For a faster training run e.g.:

```bash
python scripts/train.py --units=128 --url-cap-length=50  --sample-fraction=0.05
```
```
Running training with args: Namespace(batch_size=32, embedding_size=32, epochs=1, job_name='DefaultJob', sample_fraction=0.05, target_dir='/Users/andershuss/Develop/andhus/url-generation/results', units=128, url_cap_length=50)
Loading and preprocessing data...
Data shape: (420464, 52)
Using only 21023 samples for training
Creating model...
Epoch 1/1
2018-11-28 08:45:02.826076: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
21023/21023 [==============================] - 22s 1ms/step - loss: 3.2860
Done training!

--- Phishy URL samples ---
Hov.com.com/witoriott
ridt11r.com/00/227759/402-
tonkiochoaga.com/bovered.h.gi/moj-gasinant//hrl
bimevuf.c90303/AshvolFvtsapi/fhan
nileemebsut.com/ipoog/tiombrbcs_bepg/Ere
i-akeafc.orm/mraices/Henalra
adleider.com/
ewsis-pormersMogvardorfiletars.com/cighett/
ickimbes.som-cqcon/jalis/baksbn
ocogoxdoncanon.dame.com/s-Miinaj

--- Non-phishy URL samples ---
anpossd.com/Cam-942isuad/H26Lleyyq
raembhelskapos/20.ror/dlere-grnile/toli_cotme/iny-idtarr
vitici.ns.comg/sNPiystysixig/Srass-g=m9
koziaemidaiconw/colip/Hunecetoy-arok/are-8Ia
dvnorniwawm.coc/sutoypesemyg/enbe-dominrg/l=inie_Dablarix
gendatoted_ybsehpenpestirta-hrm.hrm/lhripld/Fersle/Nhosid.en
sfoarir.atso.com/-p-lon-ibis/Singlrk
ldeonitai.rapirnitat.cog//itdeds/bon-foin/4v/67
umuvgrirs.li.coma
zerlontc.cof/vegayj=madntetd-
```

## Sampling
```bash
python scripts/sample.py --help
```
```
usage: sample.py [-h] [--job-path JOB_PATH]

Sample phishy/non-phishy URL:s from a trained model.

optional arguments:
  -h, --help           show this help message and exit
  --job-path JOB_PATH  training job result directory to load the model from
```
Leave all arguments empty for sampling from the default model (i.e. the output from 
`scripts/train.py` with default arguments). If training has not been done, you will 
be prompted to download the default model.

```bash
python scripts/sample.py
```
```
Enter "<phishyness> <#samples> [<start of url>]" or just type "/" for random sample (type ":q" or press CTRL-C to exit)
1 5
oakhmike.chavesagdo9.gargetasholoky.com/pet/6w74O83t@p/Sansta/index.php
darkintrainicedsports.com/?playhillage=NWYZP1&linkidIId=4540,301,
clashbcart.czae.tt/wj1
soc1477937.com.au/wp-includes/pont13599/top3938f4
67.2b9.24.229/download/
0 5
blogs.2atarome.com/dellet_one.htm
presamonsengidalt.com/
anysandrapiduntv.com/forum.phymies.real.Chunkiduisman.php?board=catholocal-medispler&index=206518/23
en.wikipedia.org/wiki/Morris_Bakari
bizinessercydeld.org/en/all.html?id=f4189-fcad-delau+male-saint-luca-nai-match-2005
0 5 en.wik         
en.wikipedia.org/wiki/Manorviewer
en.wikipedia.org/wiki/Devallati
en.wikipedia.org/wiki/Droeme
en.wikipedia.org/wiki/Haunerfield_Vs67_C._Aggoli
en.wikipedia.org/wiki/Flyng
1 5 en.wik
en.wikivaliandual.com/wp-content/themes/twentyfope/ona374ybv\njenwotvsaictderaff.de/bin/login.php
en.wikimax-plotos.com/index.php
en.wikiwall-de-plannten.zoo/7B41Pgt
en.wikimayapfdhlotvstdlm.com/medalins/sch/index.php
en.wikinni.go/unyggnnuxo_eqixhxxqd\ntao.hont8443-joon-19768214.php

Enter "<phishyness> <#samples> [<start of url>]" or just type "/" for random sample (type ":q" or press CTRL-C to exit)
/
spokeo.cnation.co.za/admin/secine/xortum/cdrp/Region.aspx?wile=tosecup [phishyness: 0.73]
/
foreveed.about.com/hollyager/stilal_de0bakey/40 [phishyness: 0.14]
:q
```

# Model
The generation is done using an auto-regressive Recurrent Neural Network.

# Dataset
Currently used: https://www.kaggle.com/teseract/urldataset.
