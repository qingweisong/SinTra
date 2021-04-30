hasname:
ifdef NAME
	@echo "NAME is $(NAME)"
else
	@echo "please define NAME"
	@exit 1
endif
ifdef TYPE
	@echo "TYPE is $(TYPE)"
else
	@echo "please define TYPE"
	@exit 1
endif
ifdef CUDA
	@echo "CUDA is $(CUDA)"
else
	@echo "please define CUDA"
	@exit 1
endif
ifdef DIR
	@echo "DIR is $(DIR)"
else
	@echo "please define DIR"
	@exit 1
endif
ifdef FILE
	@echo "FILE is $(FILE)"
else
	@echo "please define FILE"
	@exit 1
endif
# c-rnn-gan-master/data/classical/sor 987study16.mid
default:
	echo $(NAME)

run: train

# multi stage
train: hasname
	CUDA_VISIBLE_DEVICES=$(CUDA) python main.py --input_dir $(DIR) --input_phrase $(FILE) --fs 8 --name $(NAME)_$(TYPE) --model_type $(TYPE) --niter $(N)

test: hasname
	CUDA_VISIBLE_DEVICES=$(CUDA) python random_sample_word.py --input_dir $(DIR) --input_phrase $(FILE) --fs 8 --mode random_samples --name $(NAME)_$(TYPE)

# single stage
train_single: hasname
	CUDA_VISIBLE_DEVICES=$(CUDA) python main.py --input_dir $(DIR) --input_phrase $(FILE) --fs 8 --name SS_$(NAME)_$(TYPE) --model_type $(TYPE) --niter $(N) --single

test_single: hasname
	CUDA_VISIBLE_DEVICES=$(CUDA) python random_sample_word.py --input_dir $(DIR) --input_phrase $(FILE) --fs 8 --mode random_samples --name SS_$(NAME)_$(TYPE) --single

# pickle
train_pickle: hasname
	CUDA_VISIBLE_DEVICES=$(CUDA) python main.py --input_dir JSB-Chorales-dataset --input_phrase ./jsb-chorales-16th.pkl --fs 8 --name $(NAME)_$(INDEX)_$(TYPE) --model_type $(TYPE) --niter $(N) --index $(INDEX)

test_pickle: hasname
	CUDA_VISIBLE_DEVICES=$(CUDA) python random_sample_word.py --input_dir JSB-Chorales-dataset --input_phrase ./jsb-chorales-16th.pkl --mode random_samples --name $(NAME)_$(INDEX)_$(TYPE) --niter $(N) --index $(INDEX)

train_pickle_single: hasname
	CUDA_VISIBLE_DEVICES=$(CUDA) python main.py --input_dir JSB-Chorales-dataset --input_phrase ./jsb-chorales-16th.pkl --fs 8 --name $(NAME)_$(INDEX)_$(TYPE) --model_type $(TYPE) --niter $(N) --index $(INDEX) --single

test_pickle_single: hasname
	CUDA_VISIBLE_DEVICES=$(CUDA) python random_sample_word.py --input_dir JSB-Chorales-dataset --input_phrase ./jsb-chorales-16th.pkl --mode random_samples --name $(NAME)_$(INDEX)_$(TYPE) --niter $(N) --index $(INDEX) --single

##### topP
train_pickle_topP: hasname
	CUDA_VISIBLE_DEVICES=$(CUDA) python main.py --input_dir JSB-Chorales-dataset --input_phrase ./jsb-chorales-16th.pkl --fs 8 --name $(NAME)_$(INDEX)_topP_$(TYPE) --model_type $(TYPE) --niter $(N) --index $(INDEX) --topP

test_pickle_topP: hasname
	CUDA_VISIBLE_DEVICES=$(CUDA) python random_sample_word.py --input_dir JSB-Chorales-dataset --input_phrase ./jsb-chorales-16th.pkl --mode random_samples --name $(NAME)_$(INDEX)_$(TYPE) --niter $(N) --index $(INDEX) --topP

train_pickle_single_topP: hasname
	CUDA_VISIBLE_DEVICES=$(CUDA) python main.py --input_dir JSB-Chorales-dataset --input_phrase ./jsb-chorales-16th.pkl --fs 8 --name $(NAME)_$(INDEX)_topP_$(TYPE) --model_type $(TYPE) --niter $(N) --index $(INDEX) --single --topP

test_pickle_single_topP: hasname
	CUDA_VISIBLE_DEVICES=$(CUDA) python random_sample_word.py --input_dir JSB-Chorales-dataset --input_phrase ./jsb-chorales-16th.pkl --mode random_samples --name $(NAME)_$(INDEX)_$(TYPE) --niter $(N) --index $(INDEX) --single --topP


# clean files
cleanModels:
	rm TrainedModels/jsb-chorales-16th_$(NAME)_$(TYPE) -rf

cleanOutputs:
	rm -vi Output/* -rf