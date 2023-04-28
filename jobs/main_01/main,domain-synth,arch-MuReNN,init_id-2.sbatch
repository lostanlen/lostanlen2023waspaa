#!/bin/bash
#SBATCH --job-name=main,domain-synth,arch-MuReNN,init_id-2
#SBATCH --output=jobs/main/slurm/slurm_%j__main,domain-synth,arch-MuReNN,init_id-2.log
#SBATCH --time=0:45:00


#########################
#
# Job: main,domain-synth,arch-MuReNN,init_id-2
# Args: 
# {'arch': 'MuReNN',
#  'batch_size': 64,
#  'domain': 'synth',
#  'init_id': 2,
#  'job_id': 'main,domain-synth,arch-MuReNN,init_id-2',
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
    --domain=synth \
    --arch=MuReNN \
    --init_id=2 \
    --batch_size=64 \
    --sav_dir=/scratch/vl1019/waspaa2023_filterbanks_data \
    --job_id=main,domain-synth,arch-MuReNN,init_id-2


echo "Many thanks to Bea Steers, author of SLURMJOBS."



###### (escape from the singularity @o@ )
echo "@: exiting singularity container"
exit 0;
EOF

