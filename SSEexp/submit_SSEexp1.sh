#PBS -A P08010000
#PBS -q main
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=128
#PBS -e /glade/campaign/cgd/tss/people/guoqiang/HBV_CAMELS/SSEexp1/joblogs
#PBS -o /glade/campaign/cgd/tss/people/guoqiang/HBV_CAMELS/SSEexp1/joblogs

module load conda
module load cdo
conda activate npl-2024a-tgq

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

ncpu=128

for iter in {2..15}
do
    python main_SSEworkflow.py config_SSEexp1.toml $iter $ncpu
done
