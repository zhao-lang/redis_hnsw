redis-cli HNSW.NEW test1 DIM 128 M 5

for i in {1..100}
do
data=$(printf "${i} %.0s" {1..128})
redis-cli HNSW.NODE.ADD test1 node${i-1} DATA 128 ${data}
done

# redis-cli bgsave

redis-cli HNSW.GET test1
redis-cli HNSW.NODE.GET test1 node1

data=$(printf "2 %.0s" {1..128})
redis-cli HNSW.SEARCH test1 QUERY 128 ${data}

for i in {1..100}
do
redis-cli HNSW.NODE.DEL test1 node${i-1}
done

redis-cli HNSW.DEL test1
