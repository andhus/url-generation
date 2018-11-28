#!/usr/bin/env bash

USAGE="$0 [-t <path to target directory>]"

# default target directory
BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. >/dev/null && pwd )"
TARGET_DIR=${BASEDIR}/datasets

while getopts ':t:' opt
do
    case $opt in
         t) TARGET_DIR=$OPTARG;;
        \?) echo "ERROR: Invalid option: $USAGE" exit 1;;
    esac
done

DATA_DIR=${TARGET_DIR}/KaggleURLData
mkdir -p ${DATA_DIR}
pushd ${DATA_DIR}
curl -L -o urldata.csv.zip "https://drive.google.com/uc?authuser=0&id=1BLXPiZD_2m58ow7Kj35Oe8JqfP6T0kPG&export=download"
unzip urldata.csv.zip
rm urldata.csv.zip
popd
