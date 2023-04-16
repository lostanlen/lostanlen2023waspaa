
#########################
#
# Job Batch: main
# Params:
# [
#   ('domain', ['speech']),
#   ('arch', ['TDFilterbank', 'LEAF', 'MuReNN']),
#   ('init_id', range(0, 5)),
# ]
#
#########################



sbatch "jobs/main/main,domain-speech,arch-TDFilterbank,init_id-0.sbatch"
sbatch "jobs/main/main,domain-speech,arch-TDFilterbank,init_id-1.sbatch"
sbatch "jobs/main/main,domain-speech,arch-TDFilterbank,init_id-2.sbatch"
sbatch "jobs/main/main,domain-speech,arch-TDFilterbank,init_id-3.sbatch"
sbatch "jobs/main/main,domain-speech,arch-TDFilterbank,init_id-4.sbatch"
sbatch "jobs/main/main,domain-speech,arch-LEAF,init_id-0.sbatch"
sbatch "jobs/main/main,domain-speech,arch-LEAF,init_id-1.sbatch"
sbatch "jobs/main/main,domain-speech,arch-LEAF,init_id-2.sbatch"
sbatch "jobs/main/main,domain-speech,arch-LEAF,init_id-3.sbatch"
sbatch "jobs/main/main,domain-speech,arch-LEAF,init_id-4.sbatch"
sbatch "jobs/main/main,domain-speech,arch-MuReNN,init_id-0.sbatch"
sbatch "jobs/main/main,domain-speech,arch-MuReNN,init_id-1.sbatch"
sbatch "jobs/main/main,domain-speech,arch-MuReNN,init_id-2.sbatch"
sbatch "jobs/main/main,domain-speech,arch-MuReNN,init_id-3.sbatch"
sbatch "jobs/main/main,domain-speech,arch-MuReNN,init_id-4.sbatch"

