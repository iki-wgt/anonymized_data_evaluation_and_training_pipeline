#!/bin/bash

for file in /path/to/source_folder/*; do
    ln -s "$file" "/path/to/target_folder/$(basename "$file")"
done
