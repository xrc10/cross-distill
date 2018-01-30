CUDA_ROOT=/usr0/home/ruochenx/local/cuda-7.5/;
CPATH="/usr0/home/ruochenx/local/cuda-7.5/:$CPATH"
export CPATH
export CUDA_ROOT;
export PATH=/usr0/home/ruochenx/local/cuda-7.5/bin:$PATH;
export LD_LIBRARY_PATH=/usr0/home/ruochenx/local/cuda-7.5/lib64:/usr0/home/ruochenx/local/cuda-7.5/extras/CUPTI/lib64:$LD_LIBRARY_PATH;
export KERAS_BACKEND=theano;

source env/bin/activate;

# batch runs
dataset=amazon_review;
tgt_langs=(de fr jp)
for i in 0 1 2
do
    tgt_lang="${tgt_langs["$i"]}";
    for dom in book dvd music
    do
        mkdir -p log/"$dataset"/"$tgt_lang"/"$dom"
        rm -rf experiments/"$dataset"/en-"$tgt_lang"/"$dom";
        THEANO_FLAGS=mode=FAST_RUN,device=gpu"$i",floatX=float32,optimizer=fast_compile,gpuarray.preallocate=1500 python train.py \
        -src_train_path data/amazon_review/en/"$dom"/train \
        -src_emb_path data/amazon_review/en/all.review.vec.txt \
        -tgt_test_path data/amazon_review/"$tgt_lang"/"$dom"/train \
        -tgt_emb_path data/amazon_review/"$tgt_lang"/all.review.vec.txt \
        -parl_data_path data/amazon_review/"$tgt_lang"/"$dom"/parl \
        -save_path experiments/"$dataset"/en-"$tgt_lang"/"$dom" \
        -dataset "$dataset" \
        &> log/"$dataset"/"$tgt_lang"/"$dom"/exp.log &
    done
done

# show the results
# cd experiments
# python gen_table.py
