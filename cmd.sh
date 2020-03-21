redis-cli hnsw.new test1 128 5

for i in {1..100}
do
data=$(printf "${i} %.0s" {1..128})
redis-cli hnsw.node.add test1 node${i-1} ${data}
done

redis-cli bgsave

redis-cli hnsw.get test1
redis-cli hnsw.node.get test1 node1

data=$(printf "2 %.0s" {1..128})
redis-cli hnsw.search test1 5 ${data}

for i in {1..100}
do
redis-cli hnsw.node.del test1 node${i-1}
done

redis-cli hnsw.del test1
