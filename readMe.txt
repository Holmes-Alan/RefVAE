Requirements:

Pytorch > 1.0
CUDA >= 10.0


The process of testing:

1. put LR images under input/LR
2. If SR without using reference, run the command:

   For 4x SR ---> python eval_4x.py --num_sample 10
   For 8x SR ---> python eval_8x.py --num_sample 10
   (--num_sample defines how many SR images you want)
   
3. If SR using external references, put refenrece images under input/Ref (any arbitrary references can be used! try it!)

   then run the command:

   For 4x SR ---> python eval_4x.py --use_ref
   For 8x SR ---> python eval_8x.py --use_ref

4. If SR using LR image itself

   then run the command:

   For 4x SR ---> python eval_4x.py --use_img_self
   For 8x SR ---> python eval_8x.py --use_img_self
