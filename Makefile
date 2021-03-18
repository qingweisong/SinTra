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
	CUDA_VISIBLE_DEVICES=1 python main.py --input_dir JSB-Chorales-dataset --input_phrase ./jsb-chorales-16th.pkl --fs 8 --name $(NAME) --niter 100

cleanModels:
	rm -vi TrainedModels/* -rf

cleanOutputs:
	rm -vi Output/* -rf

test: hasname
	CUDA_VISIBLE_DEVICES=1 python random_sample.py --input_dir midi --input_phrase ./3000adamno_l.mid --mode random_samples --name $(NAME)

test_arbitrary: hasname
	CUDA_VISIBLE_DEVICES=0 python random_sample.py --input_dir midi --input_phrase ./3000adamno_l.mid --mode random_samples_arbitrary_sizes --scale_v 2 --name $(NAME)

test_pickle: hasname
	CUDA_VISIBLE_DEVICES=1 python random_sample.py --input_dir JSB-Chorales-dataset --input_phrase ./jsb-chorales-16th.pkl --mode random_samples --name $(NAME)

test_pickle_arbitrary: hasname	
	CUDA_VISIBLE_DEVICES=0 python random_sample.py --input_dir JSB-Chorales-dataset --input_phrase ./jsb-chorales-16th.pkl --mode random_samples_arbitrary_sizes --scale_v 2 --name $(NAME)
