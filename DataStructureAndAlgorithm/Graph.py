import networkx as nx
import matplotlib.pyplot as plt
UG = nx.Graph()
for i in range(20):
    UG.add_edge(i,(i**2) % 20)
print(nx.is_tree(UG))
print(nx.is_forest(UG))

compo = nx.connected_components(UG)
for x in compo:
    print(x)

# Looking at a graph, it is obvious that it's neither a tree or a forest 
# It is not a tree because it is not connected
# It is not a Forest because there a loops 

nx.draw(UG)
plt.show()
