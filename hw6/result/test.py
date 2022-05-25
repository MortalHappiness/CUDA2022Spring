for i in range(6, 17):
    x = 2 ** i
    with open(f"input_{x}.txt", "w") as fout:
        fout.write(f"{x}\n2\n0 1\n32\n")
