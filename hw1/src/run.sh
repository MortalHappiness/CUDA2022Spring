#/usr/bin/bash
[[ ! -d "../result" ]] && mkdir "../result"
for block_size in 4 8 10 16 20 32; do
    echo "Running block size $block_size"
    for ((i=1; i<=10; ++i)); do
        folder="../result/block_$block_size"
        [[ ! -d "$folder" ]] && mkdir "$folder"
        ./MatAdd 0 "$block_size" > "$folder/$i.txt"
    done
done