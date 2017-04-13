#!/bin/bash

# NOTE: You must put ffmpeg (https://ffmpeg.org/) on your path!

# EXAMPLES:
# Convert all wav files in /Users/pstover/workspace/school/csci5622/project/data/magnatagatune/extracted
# and store them with matching relative paths in /Users/pstover/workspace/school/csci5622/project/data/magnatagatune/wav/test
# bin/convert.sh /Users/pstover/workspace/school/csci5622/project/data/magnatagatune/extracted /Users/pstover/workspace/school/csci5622/project/data/magnatagatune/wav/test
#
# Convert the first 10 wav files in the directories above
# bin/convert.sh /Users/pstover/workspace/school/csci5622/project/data/magnatagatune/extracted/b /Users/pstover/workspace/school/csci5622/project/data/magnatagatune/wav/test 10

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