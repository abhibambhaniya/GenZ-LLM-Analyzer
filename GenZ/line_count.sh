#!/bin/bash

# Function to count lines in a single Python file
count_lines() {
  local file="$1"
  if [[ -f "$file" && "$file" == *.py ]]; then
    wc -l "$file" | awk '{ sum += $1 } END { print sum }'
  else
    echo 0  # Return 0 if not a file or not a .py file
  fi
}

# Export the function so subshells can use it
export -f count_lines

# Find all .py files and sum their line counts
total_lines=$(find . -name "*.py" -type f -print0 | xargs -0 -n 1 bash -c 'count_lines "$1"' _ | paste -sd+ - | bc)

# Print the result
if [[ -n "$total_lines" ]]; then
  echo "Total lines of Python code: $total_lines"
else
  echo "No Python files found."
fi