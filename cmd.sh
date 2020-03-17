redis-cli hnsw.new test1 source 512

for i in {1..100}
do
data=$(printf "${i} %.0s" {1..512})
redis-cli hnsw.node.add test1 node${i-1} ${data}
done

# redis-cli hnsw.node.search test1
redis-cli hnsw.get test1