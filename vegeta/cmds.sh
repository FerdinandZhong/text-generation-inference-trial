vegeta attack --targets=$1_targets.txt -keepalive \
    -rate=50/100s -duration=$3s \
    | tee $4/$1_$2qps_$3duration_results.bin | vegeta report > $4/$1_$2qps_$3duration.txt;

cat $4/$1_$2qps_$3duration_results.bin \
| vegeta plot --title="$1 $2 rps for $3 seconds" > $4/$1_$2qps_$3duration.html