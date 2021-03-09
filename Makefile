hasname:
ifdef NAME
	@echo "NAME is $(NAME)"
else
	@echo "please define NAME"
	@exit 1
endif

default:
	echo $(NAME)

run: train

train: hasname
	CUDA_VISIBLE_DEVICES=1 python main.py --input_dir midi --input_phrase ./3000adamno_l.mid --fs 8 --name $(NAME)  --niter 3000

train_pickle: hasname
	CUDA_VISIBLE_DEVICES=0 python main.py --input_dir JSB-Chorales-dataset --input_phrase ./jsb-chorales-16th.pkl --fs 8 --name $(NAME) --niter 5

cleanModels:
	rm -vi TrainedModels/* -rf

cleanOutputs:
	rm -vi Output/* -rf

test:
	CUDA_VISIBLE_DEVICES=1 python random_sample.py --input_dir midi --input_phrase ./3000adamno_l.mid --mode random_samples --time 2021-3-5-23-13-1

test_arbitrary:
	CUDA_VISIBLE_DEVICES=1 python random_sample.py --input_dir midi --input_phrase ./3000adamno_l.mid --mode random_samples_arbitrary_sizes --scale_v 2 --time 2021-3-8-15-39-12
