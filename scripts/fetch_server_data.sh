#!/usr/bin/env bash
cd /home/vansh.122433/cctv-search
source /home/soft/anaconda3/etc/profile.d/conda.sh; conda activate cctv
pip install -q awscli
R=data/raw; N=${N_VIDEOS:-400}
mkdir -p $R/mevid $R/meva/mevid_videos
[ -d $R/mevid/repo ] || git clone https://github.com/Kitware/MEVID.git $R/mevid/repo
aws s3 ls --no-sign-request s3://mevadata-public-01/mevid-annotations/
aws s3 sync --no-sign-request s3://mevadata-public-01/mevid-annotations/ $R/mevid/
U=$(find $R/mevid -name 'mevid-v1-video-URLS.txt' | head -1)
echo "video list: $U ($(wc -l < $U) urls), downloading first $N"
head -$N $U | wget -c -nv -i - -P $R/meva/mevid_videos
du -sh $R/*
