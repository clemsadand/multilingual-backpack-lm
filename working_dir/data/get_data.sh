#@title Get the data

## Europarl

wget https://www.statmt.org/europarl/v7/fr-en.tgz
tar -xvzf fr-en.tgz

#rename the files
mv europarl-v7.fr-en.en data/europarl/en.txt
mv europarl-v7.fr-en.fr data/europarl/fr.txt

## MultiUN

wget https://object.pouta.csc.fi/OPUS-MultiUN/v1/moses/en-fr.txt.zip
unzip en-fr.txt.zip

%mv MultiUN.en-fr.en data/multiun/en.txt
%mv MultiUN.en-fr.fr data/multiun/fr.txt

#remove original files
rm -rf fr-en.tgz en-fr.txt.zip MultiUN.en-fr.ids README

# *********************************************************
## Reduce the size of MultiUN

%cd data/multiun
cat en.txt | head -n 8030892 > en.txt.tmp && mv en.txt.tmp en.txt
cat fr.txt | head -n 8030892 > fr.txt.tmp && mv fr.txt.tmp fr.txt
%cd /content/working_dir/

# *********************************************************

# Split the files in shards
# https://www.ibm.com/docs/el/aix/7.2?topic=s-split-command
# %cd data/multiun
split --numeric-suffixes=1 --suffix-length=1 -l 2007723 data/multiun/fr.txt data/multiun/fr.txt.
rm data/multiun/fr.txt
#
split --numeric-suffixes=1 --suffix-length=1 -l 2007723 data/multiun/en.txt data/multiun/en.txt.
rm data/multiun/en.txt
# %cd /content/working_dir/
