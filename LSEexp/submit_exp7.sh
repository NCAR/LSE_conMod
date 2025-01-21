#PBS -A P08010000
#PBS -q casper
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=36:mem=250GB
#PBS -e /glade/campaign/cgd/tss/people/guoqiang/HBV_CAMELS/joblogs
#PBS -o /glade/campaign/cgd/tss/people/guoqiang/HBV_CAMELS/joblogs

module load conda
module load cdo
conda activate npl-2024a-tgq

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

ncpu=36

for iter in {0..16}
do
    python main_LSEworkflow_HBVKM.py config_LSEexp7.toml $iter $ncpu
done


