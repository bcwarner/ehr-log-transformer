# 0: gpt2/132_3M-APP/2023-11-17
# 1: gpt2/132_3M-Attending/2023-11-12
# 2: gpt2/26_0M-APP/2023-11-11
# 3: gpt2/26_0M-Attending/2023-11-12
# 4: llama/113_0M-APP/2023-11-12
# 5: llama/113_0M-Attending/2023-11-13

bsub -q gpu-compute python gen.py --exp NextActionExperiment --model 0 --provider_type APP --count 0
bsub -q gpu-compute python gen.py --exp NextActionExperiment --model 1 --provider_type Attending --count 0
bsub -q gpu-compute python gen.py --exp NextActionExperiment --model 2 --provider_type APP --count 0
bsub -q gpu-compute python gen.py --exp NextActionExperiment --model 3 --provider_type Attending --count 0
bsub -q gpu-compute python gen.py --exp NextActionExperiment --model 4 --provider_type APP --count 0
bsub -q gpu-compute python gen.py --exp NextActionExperiment --model 5 --provider_type Attending --count 0


python gen.py --exp ScoringExperiment --model 0 --provider_type APP --count 5000
python gen.py --exp ScoringExperiment --model 1 --provider_type Attending --count 5000
python gen.py --exp ScoringExperiment --model 2 --provider_type APP --count 5000
python gen.py --exp ScoringExperiment --model 3 --provider_type Attending --count 5000
python gen.py --exp ScoringExperiment --model 4 --provider_type APP --count 5000
python gen.py --exp ScoringExperiment --model 5 --provider_type Attending --count 5000