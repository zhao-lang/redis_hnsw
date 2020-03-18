redis-cli hnsw.new test1 4

for i in {1..100}
do
data=$(printf "${i} %.0s" {1..4})
redis-cli hnsw.node.add test1 node${i-1} ${data}
done

data=$(printf "50 %.0s" {1..4})
redis-cli hnsw.search test1 5 ${data}
