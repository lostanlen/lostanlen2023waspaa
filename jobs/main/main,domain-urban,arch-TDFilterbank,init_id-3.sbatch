#!/bin/bash
#SBATCH --job-name=main,domain-urban,arch-TDFilterbank,init_id-3
#SBATCH --output=jobs/main/slurm/slurm_%j__main,domain-urban,arch-TDFilterbank,init_id-3.log
#SBATCH --time=5:00:00


#########################
#
# Job: main,domain-urban,arch-TDFilterbank,init_id-3
# Args: 
# {'arch': 'TDFilterbank',
#  'batch_size': 64,
#  'domain': 'urban',
#  'init_id': 3,
#  'job_id': 'main,domain-urban,arch-TDFilterbank,init_id-3',
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
    --domain=urban \
    --arch=TDFilterbank \
    --init_id=3 \
    --batch_size=64 \
    --sav_dir=/scratch/vl1019/waspaa2023_filterbanks_data \
    --job_id=main,domain-urban,arch-TDFilterbank,init_id-3


echo "Many thanks to Bea Steers, author of SLURMJOBS."



###### (escape from the singularity @o@ )
echo "@: exiting singularity container"
exit 0;
EOF

