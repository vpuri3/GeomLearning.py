#!/bin/bash

# Directory containing the files
# DIR="./data/netfabb_ti64_hires_timeseries/DDDD/processed_merged"

# Loop through all files in the directory
for FILE in "$DIR"/*; do
  # Check if it is a file
  if [ -f "$FILE" ]; then
    # Extract the base filename without the path
    BASENAME=$(basename "$FILE")
    
    # Remove the first 10 characters
    NEWNAME="${BASENAME:10}"
    
    # Rename the file
    mv "$FILE" "$DIR/$NEWNAME"
    
    echo "Renamed: $BASENAME -> $NEWNAME"
  fi
done

