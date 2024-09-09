#!/usr/bin/env bash

source activate MAESTRO
root_dir="/mnt"

h5_file=$1
outprefix=$2
distance=$3
species=$4
owner=$5
node_file=$6
overwrite=$7

cd ${root_dir}

if [ ! -d MAESTRO/${sample_type} ]; then
    mkdir -p MAESTRO/${sample_type};
fi;

output_h5_file=${root_dir}/MAESTRO/${outprefix}_gene_score.h5
echo "scatac-genescore output: ${output_h5_file}"

if [ "$overwrite" = "1" ]
then
  echo "MAESTRO scatac-genescore --format h5 --peakcount $h5_file --genedistance $distance --species $species --model Enhanced --outprefix $outprefix"
  MAESTRO scatac-genescore --format h5 --peakcount $h5_file --genedistance $distance --species $species --model Enhanced --outprefix $outprefix
elif [ ! -f $output_h5_file ]
then
    echo "MAESTRO scatac-genescore --format h5 --peakcount $h5_file --genedistance $distance --species $species --model Enhanced --outprefix $outprefix"
    MAESTRO scatac-genescore --format h5 --peakcount $h5_file --genedistance $distance --species $species --model Enhanced --outprefix $outprefix
fi

#if [ -e ${dir}/output/${sample_type}/7_MAESTRO/ ]
#then
#  rm ${dir}/output/${sample_type}/7_MAESTRO/ -rf
#fi
#mv -f MAESTRO/ ${dir}/output/${sample_type}/7_MAESTRO/

echo "Rscript ${root_dir}/for_TF_gene_rp_score.R ${node_file} ${output_h5_file}"
Rscript ${root_dir}/for_TF_gene_rp_score.R ${node_file} ${output_h5_file}

chown -R ${owner} ${root_dir}/TF_rp_score.txt
chown -R ${owner} ${root_dir}/MAESTRO
