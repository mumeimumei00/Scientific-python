import networkx as nx
import matplotlib.pyplot as plt
UG = nx.Graph()
UG.add_nodes_from(['A','B','C','D','E','F','G','H','I','J','K','L'])
UG.add_edges_from([('A','B'),('A','E'),('E','I'),('E','J'),('I','J'),('C','D'),('C','H'),('C','G'),('D','H'),('G','H'),('G','K'),('K','H'),('H','L')])

visited_list = []

compo = nx.connected_components(UG)
for x in compo:
    print(x)

order = 1
pre_visited = {}
post_visited = {}
# visited = []
def dfs(graph,node,visited):
    global order
    pre_visited[node] = order
    order +=1 

    visited.append(node)
    
    for vertex in graph.neighbors(node):
        if vertex not in visited:
            dfs(graph,vertex,visited)
    post_visited[node] = order
    order += 1
    return visited
# since this it not going trhough alll isolated nodes, I will add this to continue 

# for vertex in UG.nodes():
#     if vertex not in visited:
        # print("DFS Starting from a new disconnect node")
dfs(UG,next(iter(UG.nodes)),[])
print(pre_visited)
print(post_visited)
nx.draw(UG)
plt.show()
