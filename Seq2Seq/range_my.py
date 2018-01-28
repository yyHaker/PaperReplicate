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
