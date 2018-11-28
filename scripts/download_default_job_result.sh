#!/usr/bin/env bash

USAGE="$0 [-t <path to target directory>]"

# default target directory
BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. >/dev/null && pwd )"
TARGET_DIR=${BASEDIR}/results

while getopts ':t:' opt
do
    case $opt in
         t) TARGET_DIR=$OPTARG;;
        \?) echo "ERROR: Invalid option: $USAGE" exit 1;;
    esac
done

mkdir -p ${TARGET_DIR}
pushd ${TARGET_DIR}
curl -L -o DefaultJob.zip "https://drive.google.com/uc?authuser=0&id=15ChrygAc_CeYBcXcLwvCCtcPCp52h0MD&export=download"
unzip DefaultJob.zip
rm DefaultJob.zip
popd
