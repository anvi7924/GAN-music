#!/bin/bash

# NOTE: You must put ffmpeg (https://ffmpeg.org/) on your path!

# Required arguments
input_dir=$1
output_dir=$2

# Optional arguments
num_files=$3

if [[ -z "$num_files" ]]; then
  num_files=$(find $input_dir -name "*.mp3" | wc -l | awk '{print $1}')
fi
echo $num_files

pushd $input_dir

files=$(find . -name "*.mp3" | sed -e "s/^\.\///" | head -$num_files)

for mp3_file in $files; do
  wav_dir="$output_dir/$(dirname $mp3_file)"
  mkdir -p "$wav_dir"

  wav_file="$wav_dir/$(basename $mp3_file .mp3).wav"

  ffmpeg -i $mp3_file -ar 8000 -ac 1 $wav_file
done

popd