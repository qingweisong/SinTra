
run:
	CUDA_VISIBLE_DEVICES=9 python main.py --input_dir midi --input_phrase ./I_Wont_Let_You_Down.mid

test:
	CUDA_VISIBLE_DEVICES=9 python main.py --input_dir midi --input_phrase ./I_Wont_Let_You_Down.mid --niter 1
