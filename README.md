# scMultiomeGRN

## Install enviromment
### Install [MAESTRO](https://github.com/liulab-dfci/MAESTRO)
```bash
docker pull winterdongqing/maestro:1.2.1
```
### Create a conda environment
```bash
$ conda create -n scMultiomeGRN python bedops meme=5.4.1 perl-bioperl -c bioconda -c conda-forge
$ conda activate scMultiomeGRN
$ conda install -c bioconda arboreto
$ # cpan Parallel::ForkManager
$ # cpan Bio::DB::Fasta
$ pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
$ pip install torch_geometric
$ pip install lightning h5py pandas scikit-learn scipy tqdm gensim
```
## Running
see `demo/demo.ipynb`
