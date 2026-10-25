
x = [12, 56, 89]
range_len = range(len(x))
leng = len(x)
print(range_len)
print(leng)
for i in range_len:
    if i <= 2:
        x.pop(i)
        print(x)

    else:
        print(x)
