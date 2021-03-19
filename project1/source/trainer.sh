#!/bin/bash

let sta=$1
let end=$2
let prev_sta=$sta
let sta_after=$sta+2
while [ $end -ge $sta_after ]; do
    let s=$sta
    let e=$sta+2
    let e=$e-1
    for i in `seq $s $e`; do
        python3 retrieval.py --start=$i --end=$i --pc=$3 --tag=$4 &
    done
    let prev_sta=$sta
    let sta=$sta+2
    let sta_after=$sta+2
    wait
done
for i in `seq $sta $end`; do
    python3 retrieval.py --start=$i --end=$i --pc=$3 --tag=$4&
done
wait
