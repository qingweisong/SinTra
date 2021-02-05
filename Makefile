
run:
	CUDA_VISIBLE_DEVICES=8,9 python main.py --input_dir midi --input_phrase ./I_Wont_Let_You_Down.mid --fs 8 --niter 5000

test:
	CUDA_VISIBLE_DEVICES=8,9 python main.py --input_dir midi --input_phrase ./I_Wont_Let_You_Down.mid --niter 100 --fs 8

cleanModels:
	rm -vi TrainedModels/* -rf
