#!/usr/bin/env bash
cd /home/vansh.122433/cctv-search; D=data/raw/ucf_crime; mkdir -p $D
U=https://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip
LIMIT_GB=${LIMIT_GB:-150}
wget -c -nv --no-check-certificate -P $D $U &
P=$!
while kill -0 $P 2>/dev/null; do
  if [ $(du -sb $D | cut -f1) -gt $((LIMIT_GB*1024**3)) ]; then kill $P; echo "STOP: over ${LIMIT_GB} GB"; exit 1; fi
  sleep 30
done
wait $P || { echo "UCF download failed"; exit 1; }
unzip -l $D/UCF_Crimes.zip | tail -3
unzip -q -o $D/UCF_Crimes.zip '*Fighting*' '*RoadAccidents*' '*Robbery*' '*Assault*' '*Burglary*' '*Shoplifting*' '*Explosion*' '*Anomaly_Detection_splits*' '*Testing_Normal_Videos*' -d $D && rm -f $D/UCF_Crimes.zip
du -sh $D
