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

DATA_DIR=${TARGET_DIR}/PhishTank
mkdir -p ${DATA_DIR}
pushd ${DATA_DIR}
wget http://data.phishtank.com/data/online-valid.csv.gz
gunzip online-valid.csv.gz
popd
