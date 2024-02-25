
source ~/venvs/egg/bin/activate
python -m egg.zoo.emergent_captioner.finetuning.train mle coco/mle
python -m egg.zoo.emergent_captioner.finetuning.train mle mistral_4/mle
python -m egg.zoo.emergent_captioner.finetuning.train mle blip2mistral/mle
#echo "Training succesfull"
