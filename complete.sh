# use kl div to backprop

python complete.py /ssd/face/data_prepare/xdog_tmp/* --checkpointDir checkpoints_face_4_test --maskType right --batchSize 64 --lam 100 --lr 0.001 --nIter 1000 --outDir completions_face_4_3 --text_vector_dim 18 --text_path datasets/celeba/imAttrs.pkl