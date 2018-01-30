# training.....
for i in range(1000):
    losses = []
    for j in range(0, 10, 3):
        print(j)
    break

# slice index larger than len(list)
A = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
print(A[:20])

max_len = 10
lens = [3, 9, 10, 6, 10]
mask = [
        ([1] * (l - 1)) + ([0] * (max_len - l))
        for l in lens
    ]
print(mask)

#  filter
def is_odd(x):
    return x % 2 == 1
print(list(filter(is_odd, [1, 4, 6, 7, 9, 12, 17])))

# 使用codecs打开文件，如果文件不存在的话，将会自动创建
import codecs
codecs.open("test.txt", 'wb')

open("test.cl", 'wb')

# open以w打开文件，如果文件不存在，会自动创建
# 不过需要注意系统创建文件名字长度的情况，超出长度也会抛出异常
import os
save_dir = "data/lisatmp4/subramas/models/torch_vanilla_seq2seq"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
open(os.path.join(save_dir, "model_translation__src_fr__trg_en__minibatch_0.model"), 'wb')



