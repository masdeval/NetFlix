#!/bin/bash
timestamp=$(date +"%Y-%m-%d_%H-%M")
input_csv_file=$1
output_csv_file="header_"$timestamp"_"$input_csv_file

o=""
# Find the number of columns (commas) in the first row
n=$(($(head -n1 $input_csv_file | sed 's/[^,]//g' | wc -c)))    

for i in $(seq 1 $((n-1)));  # Get a list of numbers equal to column qty
do
        o=$o""$i",";
done
o=$o""$n;

#Write the numbers with commas to first line of new file.
echo $o > $output_csv_file              
#Append whole of other file to new file.
cat $input_csv_file >> $output_csv_file 
