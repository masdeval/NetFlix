#!/usr/bin/awk -f
BEGIN { width=0; }
{
    max = split($0, list, ",");
    # printf "%d:%s\n", NR, $0;
    if (width < max)
        width = max;
    for (n = 1; n <= max; ++n) {
        sub("^[     ]*","",list[n]);
        sub("[  ]*$","",list[n]);
        # printf "\t%d:%s\n", n, list[n];
        if ( columns[n] != "" ) {
            columns[n] = columns[n] ",";
        }
        columns[n] = columns[n] list[n];
    }
}
END {
    # printf "%d columns\n", width;
    for (n = 1; n <= width; ++n) {
        printf "%s\n", columns[n];
    }
}
