
source ~/venvs/egg/bin/activate
python -m egg.zoo.emergent_captioner.finetuning.train cider coco
python -m egg.zoo.emergent_captioner.finetuning.train cider mistral_4
python -m egg.zoo.emergent_captioner.finetuning.train cider blip2mistral
#echo "Training succesfull"
