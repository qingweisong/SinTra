
run:
	CUDA_VISIBLE_DEVICES=1 python main.py --input_dir midi --input_phrase ./3000adamno_l.mid --fs 8 --niter 1000

run_test:
	CUDA_VISIBLE_DEVICES=1 python random_sample.py --input_dir midi --input_phrase ./3000adamno_l.mid --mode random_samples

cleanModels:
	rm -vi TrainedModels/* -rf

cleanOutputs:
	rm -vi Output/* -rf

test:
	CUDA_VISIBLE_DEVICES=1 python random_sample.py --input_dir midi --input_phrase ./3000adamno_l.mid --mode random_samples --time 2021-2-23-22-8-13
