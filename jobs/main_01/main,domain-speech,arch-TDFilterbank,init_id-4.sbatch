#!/bin/bash
#SBATCH --job-name=main,domain-speech,arch-TDFilterbank,init_id-4
#SBATCH --output=jobs/main/slurm/slurm_%j__main,domain-speech,arch-TDFilterbank,init_id-4.log


#########################
#
# Job: main,domain-speech,arch-TDFilterbank,init_id-4
# Args: 
# {'arch': 'TDFilterbank',
#  'batch_size': 64,
#  'domain': 'speech',
#  'init_id': 4,
#  'job_id': 'main,domain-speech,arch-TDFilterbank,init_id-4',
#  'sav_dir': '/scratch/vl1019/waspaa2023_filterbanks_data'}
#
#########################




###### ( hop into the singularity o_O )
singularity exec  \
    --overlay /scratch/vl1019/wa23_overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash << EOF
echo "@: entered singularity container"

[[ -f /ext3/env.sh ]] && source /ext3/env.sh
[[ -f /ext3/env ]] && source /ext3/env








echo "Optimizing auditory filterbanks by gradient descent"

python main.py \
    --domain=speech \
    --arch=TDFilterbank \
    --init_id=4 \
    --batch_size=64 \
    --sav_dir=/scratch/vl1019/waspaa2023_filterbanks_data \
    --job_id=main,domain-speech,arch-TDFilterbank,init_id-4


echo "Many thanks to Bea Steers, author of SLURMJOBS."



###### (escape from the singularity @o@ )
echo "@: exiting singularity container"
exit 0;
EOF

