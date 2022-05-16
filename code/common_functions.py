import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
import jenkspy
import networkx as nx
from networkx.algorithms import bipartite, community
import colorcet as cc
import matplotlib.pyplot as plt

PROJECT_DIR = "../../data-communities/"
FIGURES_PATH = PROJECT_DIR + "figures/"
OUTPUTS_PATH = PROJECT_DIR + "outputs/"

# HANDLING GRAPHS

def load_graph(filename):
    """Loads a saved graph from a file.

    Parameters
    ----------
    filename: str
        Name of graph

    Returns
    ------
    G: networkx.classes.graph.Graph
        Graph
    left_nodes: set
        Container for left set of nodes in bipartite graph
    right_nodes: set
        Container for right set of nodes in bipartite graph
    """
    
    load_path = OUTPUTS_PATH + filename +'.gpickle'
    G = nx.read_gpickle(load_path)
    if nx.bipartite.is_bipartite(G) == True:
        print(nx.info(G))
        left_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
        print("Left nodes:",len(left_nodes))
        right_nodes = set(G) - left_nodes
        print("Right nodes:",len(right_nodes))
        return G, left_nodes, right_nodes
    else:
        print(nx.info(G))
        return G

def save_bipartite(G, filename):
    """Saves a biparite graph to a file.

    Parameters
    ----------
    G: networkx.classes.graph.Graph
        Graph
    filename: str
        Name to save graph

    Returns
    ------
    """
    
    save_path = OUTPUTS_PATH + filename +'.gpickle'
    nx.write_gpickle(G, save_path)     
    
def save_graph(G, filename):
    """Saves a graph.

    Parameters
    ----------
    G: networkx.classes.graph.Graph
        Graph
    filename: str
        Name to save graph

    Returns
    ------
    """
    
    save_path = OUTPUTS_PATH + filename +'.gpickle'
    nx.write_gpickle(G, save_path)

# MAKING GRAPHS

def make_bipartite(df, U, V, graphname=None):
    """Makes a bipartite graph based on two user-defined nodesets, and optionally names the graph.

    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        Dataframe with node information columns
    U: str
        Name of column to make left node set in bipartite 
    V: str
        Name of column to make right node set in bipartite

    Returns
    ------
    G: networkx.classes.graph.Graph
        Graph
    """
    
    G = nx.Graph()
    G.add_nodes_from(df[U], bipartite=0)
    G.add_nodes_from(df[V], bipartite=1)
    G.add_edges_from([(row[U], row[V]) for idx, row in df.iterrows()])
    nodes_U = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    nodes_V = set(G) - nodes_U
    
    color = bipartite.color(G)
    color_dict = {0:'b',1:'r'}
    color_list = [color_dict[i[1]] for i in G.nodes.data('bipartite')]
    pos = {node:[0, i] for i,node in enumerate(df[U])}
    pos.update({node:[1, i] for i,node in enumerate(df[V])})
    pos = {node:[0, i] for i,node in enumerate(df[U])}
    pos.update({node:[1, i] for i,node in enumerate(df[V])})
    
    print("Network is connected:", nx.is_connected(G))
    print(U,len(nodes_U))
    print(V,len(nodes_V))
    
    if len(df)<100:
        return G, pos, color_list
    else:
        G.name = graphname
        return G

def project_graph(G, partition, min_weight, identity, edgenames, nodenames, graphname=None):
    """Adds edge weights; projects nodes from a bipartite graph partition; saves node and edge lists.

    Parameters
    ----------
    G: networkx.classes.graph.Graph
        Graph
    partition: int
        Identity of node partition to select
    min_weight: int
        Minimum edge weight to remove from edge set
    identity: str
        Name of the node identifier type (options: "study", "data")
    edgenames: str
        Name of file to save edge list
    nodenames: str
        Name of file to save node list
    Returns
    ------
    networkx.classes.graph.Graph
        Graph
    """
    
    nodeset = [n for n, d in G.nodes(data=True) if d["bipartite"] == partition]
    print("Confirming partition type:", type(nodeset[0]))
    S = bipartite.projected_graph(G, nodeset, multigraph=True)
    
    citation_dict = dict(S.degree(S.nodes()))
    
    weight_dict = Counter(S.edges())
    edge_weights = [ (u, v, {'weight': value}) 
                    for ((u, v), value) in weight_dict.items()]
    W = nx.Graph()
    W.add_nodes_from(S.nodes)
    W.add_edges_from(edge_weights)
    
    degree_dict = dict(W.degree(W.nodes()))
    nx.set_node_attributes(W, degree_dict, 'degree')
    nx.set_node_attributes(W, citation_dict, 'citations')
    
    to_remove = [(a,b) for a,b,attrs in W.edges(data=True) if attrs['weight'] < min_weight]
    remove_size = len(to_remove)
    
    print(f"Removing {remove_size} edges of weight less than {min_weight}")
    W.remove_edges_from(to_remove)
    
    isolates_list = list(nx.isolates(W))
    isolates_size = len(list(nx.isolates(W)))
    print(f"Removing {isolates_size} isolated nodes")
    W.remove_nodes_from(isolates_list)

    edge_df = nx.to_pandas_edgelist(W)
    save_edge_path = OUTPUTS_PATH + edgenames +'.csv'
    edge_df.to_csv(save_edge_path,index=False)
    
    nodelist = list(W.nodes(data=True))
    
    if G.name == "S":
        if identity == "study":
            node_df = pd.DataFrame(nodelist, columns=['STUDY', 'degree'])
        else:
            node_df = pd.DataFrame(nodelist, columns=['REF_DATA', 'degree'])
        save_node_path = OUTPUTS_PATH + nodenames +'.csv'
        node_df.to_csv(save_node_path,index=False)
    
    elif G.name == "A":
        node_df = pd.DataFrame(nodelist, columns=['AUTHOR_ID', 'degree'])
        save_node_path = OUTPUTS_PATH + nodenames +'.csv'
        node_df.to_csv(save_node_path,index=False)
    
    else:
        print("Node names not found")
    
    W.name = graphname
    print(nx.info(W))
    return W    
    
def add_attributes(G, identity, df):
    """Adds a fixed set of attributes to nodes in a graph based on the name of the graph.
    
    Parameters
    ----------
    G: networkx.classes.graph.Graph
        Graph
    identity: str
        Name of node id type (options: "study", "data")
    df: pandas.core.frame.DataFrame
        Dataframe with attribute information columns

    Returns
    ------
    networkx.classes.graph.Graph
        Graph
    """
    
    degree_dict = dict(G.degree(G.nodes()))
    betweenness_dict = nx.betweenness_centrality(G)

    nx.set_node_attributes(G, degree_dict, 'degree')
    nx.set_node_attributes(G, betweenness_dict, 'between')

    if G.name == "S":
        if identity == "study":
            study_number_dict = dict(zip(df['STUDY'], df['STUDY']))
            study_name_dict = dict(zip(df['STUDY'], df['NAME']))
            study_series_dict = dict(zip(df['STUDY'], df['SERIES_TITLE']))
            study_owner_dict = dict(zip(df['STUDY'], df['OWNER']))
            study_funding_dict = dict(zip(df['STUDY'], df['FUNDINGAGENCY']))
            study_geo_dict = dict(zip(df['STUDY'], df['GEO']))
            study_terms_dict = dict(zip(df['STUDY'], df['TERMS']))
            study_release_dict = dict(zip(df['STUDY'], df['ORIGRELDATE']))
        
        else:
            study_number_dict = dict(zip(df['REF_DATA'], df['REF_DATA']))
            study_name_dict = dict(zip(df['REF_DATA'], df['NAME']))
            study_series_dict = dict(zip(df['REF_DATA'], df['SERIES_TITLE']))
            study_owner_dict = dict(zip(df['REF_DATA'], df['OWNER']))
            study_funding_dict = dict(zip(df['REF_DATA'], df['FUNDINGAGENCY']))
            study_geo_dict = dict(zip(df['REF_DATA'], df['GEO']))
            study_terms_dict = dict(zip(df['REF_DATA'], df['TERMS']))
            study_release_dict = dict(zip(df['REF_DATA'], df['ORIGRELDATE']))

        nx.set_node_attributes(G, study_number_dict, 'study_number')
        nx.set_node_attributes(G, study_name_dict, 'study_name')
        nx.set_node_attributes(G, study_series_dict, 'study_series')
        nx.set_node_attributes(G, study_owner_dict, 'study_owner')
        nx.set_node_attributes(G, study_funding_dict, 'study_funding')
        nx.set_node_attributes(G, study_geo_dict, 'study_geo')
        nx.set_node_attributes(G, study_terms_dict, 'study_terms')
        nx.set_node_attributes(G, study_release_dict, 'study_release')
        
    elif G.name == "A":
        author_number_dict = dict(zip(df['AUTHOR_ID'], df['AUTHOR_ID']))
        author_name_dict = dict(zip(df['AUTHOR_ID'], df['AUTHOR']))
        paper_title_dict = dict(zip(df['AUTHOR_ID'], df['TITLE']))
        paper_year_dict = dict(zip(df['AUTHOR_ID'], df['YEAR_PUB']))
        paper_type_dict = dict(zip(df['AUTHOR_ID'], df['RIS_TYPE']))
        paper_funder_dict = dict(zip(df['AUTHOR_ID'], df['FUNDER']))
        paper_authors_dict = dict(zip(df['AUTHOR_ID'], df['authors']))
        paper_concepts_dict = dict(zip(df['AUTHOR_ID'], df['concepts']))
        paper_cited_dict = dict(zip(df['AUTHOR_ID'], df['times_cited']))
        paper_category_dict = dict(zip(df['AUTHOR_ID'], df['category_for']))
        paper_funders_dict = dict(zip(df['AUTHOR_ID'], df['funders']))

        nx.set_node_attributes(G, author_number_dict, 'author_number')
        nx.set_node_attributes(G, author_name_dict, 'author_name')
        nx.set_node_attributes(G, paper_title_dict, 'paper_title')
        nx.set_node_attributes(G, paper_year_dict, 'paper_year')
        nx.set_node_attributes(G, paper_type_dict, 'paper_type')
        nx.set_node_attributes(G, paper_funder_dict, 'paper_funder')
        nx.set_node_attributes(G, paper_authors_dict, 'paper_authors')
        nx.set_node_attributes(G, paper_concepts_dict, 'paper_concepts')
        nx.set_node_attributes(G, paper_cited_dict, 'paper_citations')
        nx.set_node_attributes(G, paper_category_dict, 'paper_categories')
        nx.set_node_attributes(G, paper_funders_dict, 'paper_funders')
    else:
        print("Graph not found.")
    return G

def get_node_sets(G):
    """Gets node sets from a bipartite graph.
    
    Parameters
    ----------
    G: networkx.classes.graph.Graph
        Graph

    Returns
    ------
    left_nodes: set
        Container for left set of nodes in bipartite graph
    right_nodes: set
        Container for right set of nodes in bipartite graph
    """
    
    left_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    print("Left nodes:",len(left_nodes))
    right_nodes = set(G) - left_nodes
    print("Right nodes:",len(right_nodes))
    return left_nodes, right_nodes

# PLOTTING GRAPHS
    
def plot_bipartite(G, pos, color_list, filename):
    """Plots a bipartite graph.

    Parameters
    ----------
    G: networkx.classes.graph.Graph
        Graph
    pos: dict
        Node label positions 
    color_list: list
        Nodes and colors for plotting
    filename: str
        Name to save plot

    Returns
    ------
    """
    
    plt.figure(1, figsize=(20, 12), dpi=50)
    nx.draw(G, pos, with_labels=False, node_color=color_list)
    for p in pos:
        pos[p][1] += 0.5
    nx.draw_networkx_labels(G, pos)
    # save_path = FIGURES_PATH + filename +'.png'
    # plt.savefig(save_path)
    plt.show()
    print(nx.info(G))
    
def plot_graph(G, filename):
    """Plots a graph.

    Parameters
    ----------
    G: networkx.classes.graph.Graph
        Graph
    filename: str
        Name to save plot

    Returns
    ------
    """
    
    fig = plt.figure(1, figsize=(20, 20), dpi=50)
    pos = nx.spring_layout(G, k=0.1)
    plt.rcParams.update(plt.rcParamsDefault)
    
    all_edges = G.edges()
    weight = [all_edges[(u, v)]['weight'] for u,v in all_edges]
    
    nx.draw_networkx(
        G,
        pos=pos,
        node_size=0.1,
        edge_color="#444444",
        width=weight,
        alpha=0.05,
        with_labels=False)
    
    # save_path = FIGURES_PATH + filename +'.png'
    # plt.savefig(save_path)
    plt.show()    
    
def plot_hubs(G, hub_type, threshold, filename):
    """Plots hubs (high degree, high betweenness) in a graph with labels.

    Parameters
    ----------
    G: networkx.classes.graph.Graph
        Graph
    hub_type: str
        Definition for node importance (Options: degree, betweenness)
    threshold: float
        Threshold for selecting importance nodes
    filename: str
        Name to save plot

    Returns
    ------
    """
    
    if hub_type=="degree":
        top_degree = [x for x,y in G.nodes(data=True) if y['degree']>=threshold]
        subgraph = G.subgraph(top_degree)
        color='red'
        print("Top degree datasets:",top_degree)
    
    else:
        top_between = [x for x,y in G.nodes(data=True) if y['between']>=threshold]
        subgraph = G.subgraph(top_between)
        color='blue'
        print("Top betweenness datasets:",top_between)

    fig = plt.figure(1, figsize=(40, 40), dpi=50)
    pos = nx.spring_layout(G, k=0.2, seed=42)
    
    all_edges = G.edges()
    weight = [all_edges[(u, v)]['weight'] for u,v in all_edges]

    nx.draw_networkx_edges(G,
                           pos=pos,
                           width=weight,
                           edge_color="black",
                           alpha=0.05);
    
    nx.draw_networkx(G.subgraph(subgraph), 
                     pos=pos,  
                     node_size=1500,
                     width=0,
                     node_color=color, 
                     with_labels=True,
                     font_color="white")

    # save_path = FIGURES_PATH + filename +'.png'
    # plt.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=0)    
    plt.show()

def plot_owners(G, df, filename):
    """Plots a graph with color symbolizing ICPSR study owner.

    Parameters
    ----------
    G: networkx.classes.graph.Graph
        Graph
    df : pandas.core.frame.DataFrame
        Dataframe with node information columns
    filename: str
        Name to save plot

    Returns
    ------
    """
    
    df['OWNER'] = pd.Categorical(df['OWNER'])
    owner_groups = df['STUDY'].groupby(df['OWNER']).count().reset_index()
    owner_names = ['Archive of Data on Disability', 
                'American Educational Research Association', 
                'Archives of Scientific Psychology', 
                'Child and Family Data Archive',
                'Data Archive for Interdisciplinary Research on Learning', 
                'Data Sharing for Demographic Research', 
                'Health and Medical Care Archive', 
                'ICPSR General Archive', 
                'Measures of Effective Teaching Longitudinal Database', 
                'National Archive of Computerized Data on Aging', 
                'National Archive of Criminal Justice Data', 
                'National Archive of Data on Arts and Culture', 
                'National Addiction & HIV Data Archive', 
                'Resource Center for Minority Data', 
                'Centre for Data Archiving, Management, Analysis and Advocacy', 
                'ResearchDataGov', 
                'Civic Learning, Engagement, and Action Data Sharing', 
                'Open Data Flint', 
                'Patient-Centered Outcomes Data Repository', 
                'Patient-Centered Outcomes Research Institute Data Repository']
    
    owner_short = ['disability', 
                  'education', 
                  'psychology', 
                  'child and family',
                  'learning', 
                  'demographics', 
                  'health and medical', 
                  'general', 
                  'teaching methods', 
                  'aging', 
                  'criminal justice', 
                  'arts and culture', 
                  'addiction and HIV', 
                  'minority data', 
                  'management and advocacy', 
                  'federal statistics', 
                  'civic engagement', 
                  'open flint',
                  'patient data', 
                  'patient research']
    
    name_series = pd.Series(owner_names)
    owner_groups['OWNER_full'] = name_series
    owner_groups['OWNER_short'] = owner_short
    owner_groups['OWNER_cat'] = owner_groups['OWNER'].cat.codes
    fig = plt.figure(1, figsize=(20, 20), dpi=50)
    pos = nx.spring_layout(G, k=0.1)
    plt.rcParams.update(plt.rcParamsDefault)
    nc = nx.draw_networkx_nodes(G, 
                                pos=pos,
                                node_size=2,
                                node_color=df.OWNER.cat.codes,
                                cmap=plt.cm.tab20)
    ec = nx.draw_networkx_edges(G, 
                                pos=pos,
                                width=0,
                                edge_color="silver",
                                alpha=1)
    cbar = plt.colorbar(nc)
    cbar.set_ticks(list())
    owner_list = owner_groups.OWNER_short.to_list()
    for index, label in enumerate(owner_list):
        x = 0.5
        y = (2 * index + 1) / 2
        cbar.ax.text(x, y, label)
    plt.axis('off')
    # save_path = FIGURES_PATH + filename +'.png'
    # plt.savefig(save_path)
    plt.show()
    
# GRAPH CALCULATIONS

def graph_metrics(G, weight=None):
    """Calculates metrics (density, transitivity, assortativity) and components for a graph.

    Parameters
    ----------
    G: networkx.classes.graph.Graph
        Graph
    weight: str or None
        Name of the edge attribute that holds edge weight
    Returns
    ------
    """
    
    density = nx.density(G)
    print("Network density:", density)
    transitivity = nx.transitivity(G)
    print("Network transitivity:", transitivity)
    assortativity = nx.degree_assortativity_coefficient(G, weight=weight)
    print("Network assortativity:", assortativity)
    print("Network is connected:", nx.is_connected(G))
    print("Network components:", nx.number_connected_components(G))
    
def get_largest_subgraph(G): 
    """Finds the largest subgraph of a graph.

    Parameters
    ----------
    G: networkx.classes.graph.Graph
        Graph

    Returns
    ------
    subgraph: networkx.classes.graph.Graph
        The largest connected subgraph of G
    """
    
    components = nx.connected_components(G)
    largest_component = max(components, key=len)
    subgraph = G.subgraph(largest_component)
    nx.diameter(subgraph)
    density = nx.density(G)
    print("Largest subgraph density:", density)
    transitivity = nx.transitivity(G)
    print("Largest subgraph transitivity:", transitivity)
    return subgraph    

def find_heavy_edges(G, nb_class):
    """
    Uses natural breaks to find the heaviest edges in a graph.

    Parameters
    ----------
    G: networkx.classes.graph.Graph
        Graph
    nb_class: int
        Number of classes
    
    Returns
    ------
    """
    
    df_edge_wts = pd.DataFrame.from_dict(G.edges(), orient='index').reset_index()
    df_edge_wts = df_edge_wts.rename(columns={"index":"source",0:"target",1:"weight"}).sort_values(by="weight")
    data = df_edge_wts['weight']
    breaks = jenkspy.jenks_breaks(df_edge_wts['weight'], nb_class=nb_class)
    print("Breaks: ",breaks)
    
    plt.figure(figsize = (10, 8))
    hist = plt.hist(data, bins=20, align='left', color='g')
    for b in breaks:
        plt.vlines(b, ymin=0, ymax = max(hist[0]))
    
    top_break = int(breaks[nb_class-1])
    top_weights = ((u,v,d) for u,v,d in G.edges(data=True) if d['weight']>=top_break)
    sorted_weights = sorted(top_weights,key=lambda x: x[2]['weight'],reverse=True)
    print("Size: ",len(sorted_weights))
    if G.name == "S":
        for i in sorted_weights:
            source_num = i[0]
            target_num = i[1]
            edge_weight = i[2]
            for edge_key in edge_weight:
                edge_value = edge_weight[edge_key]
            
            print("EDGE WEIGHT:",edge_value,"\n",
                  "SOURCE (study):",G.nodes[source_num]['study_name'],"\n",
                  "SOURCE (series):",G.nodes[source_num]['study_series'],"\n",
                  "SOURCE (owner):",G.nodes[source_num]['study_owner'],"\n",
                  "SOURCE (number):",G.nodes[source_num]['study_number'],"\n",
                  "SOURCE (degree):",G.nodes[source_num]['degree'],"\n",
                  "SOURCE (betweenness):",G.nodes[source_num]['between'],"\n",
                  "SOURCE (citations):",G.nodes[source_num]['citations'],"\n",

                  "TARGET (study):",G.nodes[target_num]['study_name'],"\n", 
                  "TARGET (series):",G.nodes[target_num]['study_series'],"\n",
                  "TARGET (owner):",G.nodes[target_num]['study_owner'],"\n",
                  "TARGET (number):",G.nodes[target_num]['study_number'],"\n",
                  "TARGET (degree):",G.nodes[target_num]['degree'],"\n", 
                  "TARGET (betweenness):",G.nodes[target_num]['between'],"\n", 
                  "TARGET (citations):",G.nodes[target_num]['citations'],"\n")
    
    elif G.name == "A":
        for i in sorted_weights:
            source_num = i[0]
            target_num = i[1]
            edge_weight = i[2]
            for edge_key in edge_weight:
                edge_value = edge_weight[edge_key]
            
            print("EDGE WEIGHT:",edge_value,"\n",
                  "SOURCE (author):",G.nodes[source_num]['author_name'],"\n",
                  "SOURCE (fields):",G.nodes[source_num]['paper_categories'],"\n",
                  "SOURCE (number):",G.nodes[source_num]['author_number'],"\n",

                  "TARGET (author):",G.nodes[target_num]['author_name'],"\n", 
                  "TARGET (fields):",G.nodes[target_num]['paper_categories'],"\n",
                  "TARGET (number):",G.nodes[target_num]['author_number'],"\n")
        
    else:
        print("Edge weights not found")
        
def find_central_nodes(G, nb_class):
    """
    Uses natural breaks to find the top degree nodes in a graph.

    Parameters
    ----------
    G: networkx.classes.graph.Graph
        Graph
    nb_class: int
        Number of classes
    
    Returns
    ------
    """
    
    degree_dict = dict(G.degree(G.nodes()))
    df_stud_deg = pd.DataFrame.from_dict(degree_dict, orient='index').reset_index()
    df_stud_deg = df_stud_deg.rename(columns={"index":"study",0:"degree"}).sort_values(by="degree")
    breaks = jenkspy.jenks_breaks(df_stud_deg['degree'], nb_class=nb_class)
    print("Breaks: ",breaks)
    top_break = int(breaks[nb_class-1])
    top_degree = ((u,v) for u,v in G.nodes(data=True) if v['degree']>=top_break)
    sorted_degree = sorted(top_degree,key=lambda x: x[1]['degree'],reverse=True)
    print("Size: ",len(sorted_degree))
    for i in sorted_degree:
        data_num = i[0]
        data_deg = i[1]
        if G.name == "S":
            print("DEGREE:", G.nodes[data_num]['degree'],"\n",
                  "BETWEENNESS:", G.nodes[data_num]['between'],"\n",
                  "SERIES:",G.nodes[data_num]['study_series'],"\n",
                  "STUDY:",G.nodes[data_num]['study_name'],"\n",
                  "STUDY NUMBER:",G.nodes[data_num]['study_number'],"\n",
                  "OWNER:",G.nodes[data_num]['study_owner'],"\n",
                  "CITATIONS:", G.nodes[data_num]['citations'],"\n")
        elif G.name == "A":
            print("DEGREE:", G.nodes[data_num]['degree'],"\n",
                  "AUTHOR NAME:",G.nodes[data_num]['author_name'],"\n",
                  "AUTHOR FIELDS:",G.nodes[data_num]['paper_categories'],"\n",
                  "AUTHOR NUMBER:",G.nodes[data_num]['author_number'],"\n")

def find_between_nodes(G, nb_class):
    """
    Uses natural breaks to find the top betweenness nodes in a graph.

    Parameters
    ----------
    G: networkx.classes.graph.Graph
        Graph
    nb_class: int
        Number of classes
    
    Returns
    ------
    """
    
    betweenness_dict = nx.betweenness_centrality(G)
    df_stud_bet = pd.DataFrame.from_dict(betweenness_dict, orient='index').reset_index()
    df_stud_bet = df_stud_bet.rename(columns={"index":"study",0:"between"}).sort_values(by="between")
    breaks = jenkspy.jenks_breaks(df_stud_bet['between'], nb_class=nb_class)
    print("Breaks: ",breaks)
    top_break = int(breaks[nb_class-1])
    top_between = ((u,v) for u,v in G.nodes(data=True) if v['between']>=top_break)
    sorted_between = sorted(top_between,key=lambda x: x[1]['between'],reverse=True)
    print("Size: ",len(sorted_between))
    for i in sorted_between:
        data_num = i[0]
        data_bet = i[1]
        if G.name == "S":
            print("BETWEENNESS:", G.nodes[data_num]['between'],"\n",
                  "DEGREE:", G.nodes[data_num]['degree'],"\n",
                  "SERIES:",G.nodes[data_num]['study_series'],"\n",
                  "STUDY:",G.nodes[data_num]['study_name'],"\n",
                  "STUDY NUMBER:",G.nodes[data_num]['study_number'],"\n",
                  "OWNER:",G.nodes[data_num]['study_owner'],"\n",
                  "CITATIONS:", G.nodes[data_num]['citations'],"\n")
        elif G.name == "A":
            print("BETWEENNESS:", G.nodes[data_num]['between'],"\n",
                  "DEGREE:", G.nodes[data_num]['degree'],"\n",
                  "CITATIONS:", G.nodes[data_num]['citations'],"\n",
                  "AUTHOR NAME:",G.nodes[data_num]['author_name'],"\n",
                  "AUTHOR FIELDS:",G.nodes[data_num]['paper_categories'],"\n",
                  "AUTHOR NUMBER:",G.nodes[data_num]['author_number'],"\n")
            
def shortest_path(G, source_node, target_node):
    """
    Finds shortest path between two nodes in a graph.

    Parameters
    ----------
    G: networkx.classes.graph.Graph
        Graph
    source_node: int
        Identity of source node
    target_node: int
        Identity of target node
    Returns
    ------
    """
    
    path_check = nx.has_path(G, source_node, target_node)
    if path_check == True:
        if G.name =="S":
            print("SOURCE found:",G.nodes[source_node]['study_name'], "\n")
            print("TARGET found:",G.nodes[target_node]['study_name'], "\n")
            path = nx.shortest_path(G, source=source_node, target=target_node)
            print("Shortest SOURCE to TARGET path:", (len(path)-1), "\n")
            for i in path:
                print("\t Path link:", G.nodes[i]['study_name'], "| Path link:", G.nodes[i]['study_name'], "\n")
        else:
            print("SOURCE found:",G.nodes[source_node]['author_name'], "\n")
            print("TARGET found:",G.nodes[target_node]['author_name'], "\n")
            path = nx.shortest_path(G, source=source_node, target=target_node)
            print("Shortest SOURCE to TARGET path:", (len(path)-1), "\n")
            for i in path:
                print("\t Path link:", G.nodes[i]['author_name'], "| Path link:", G.nodes[i]['author_name'], "\n")  
    else:
        print("No path exists between SOURCE and TARGET")
        
# DETECTING COMMUNITIES (BIPARTITE)
    
def bipartite_comms(G, theta=None, lambd=None):
    """Applies community detection algorithm to a bipartite graph.

    Parameters
    ----------
    G: networkx.classes.graph.Graph
        Graph
    theta: float or None
        Label weights threshold. Default 0.3.
    lambd: int or None
        The max number of labels. Default 7.
    Returns
    ------
    comms: cdlib.classes.bipartite_node_clustering.BiNodeClustering
        BiNodeClustering network object with community information
    """
    
    print(nx.info(G))
    comms = bimlpa(G)
    print("Overlapping clustering: ", comms.overlap)
    print("Node coverage: ", comms.node_coverage)
    print("Left communities detected: ", len(comms.left_communities))
    print("Right communities detected: ", len(comms.right_communities))
    return comms

def save_author_comms(comms, df_author, authorfile, studyfile):
    """Saves community detection result for author and study bipartite graph.

    Parameters
    ----------
    comms: cdlib.classes.bipartite_node_clustering.BiNodeClustering
        Network object with community information
    df_author: pandas.core.frame.DataFrame
        Dataframe with author information columns
    authorfile: string
       Name of the file to save the community detection result for authors
    studyfile: string
       Name of the file to save the community detection result for studies
    Returns
    ------
    """
    
    comms_dict = comms.to_node_community_map()
    df = pd.DataFrame(comms_dict.items(), columns=['id', 'code'])
    df['community'] = df['code'].str[0]
    df = df.drop(columns=['code'])

    m = df['id'].str.contains('_', na=False)

    authors = df[m].reset_index(drop=True)
    authors['REF_AUTHOR'] = authors['id']
    authors = authors.drop(columns=['id'])
    
    studies = df[~m].reset_index(drop=True)
    studies['STUDY'] = studies['id']
    studies = studies.drop(columns=['id'])

    author_result = pd.merge(authors, df_author, on="REF_AUTHOR", how="left").drop_duplicates(subset=['REF_AUTHOR'])
    author_result = author_result[['community',
                               'REF_AUTHOR',
                               'AUTHOR',
                               'STUDY',
                               'NAME',
                               'research_org_names',
                               'concepts',
                               'category_for']]

    study_result = pd.merge(studies, df_author, on="STUDY", how="left").drop_duplicates(subset=['STUDY'])
    study_result = study_result[['community',
                             'STUDY',
                             'NAME',
                             'SERIES_TITLE',
                             'SERIES',
                             'AUTHOR',
                             'OWNER',
                             'FUNDINGAGENCY',
                             'GEO',
                             'TERMS',
                             'ORIGRELDATE']]
    
    save_path_author = OUTPUTS_PATH + authorfile +'.csv'
    save_path_study = OUTPUTS_PATH + studyfile +'.csv'
    author_result.to_csv(save_path_author,index=False)
    study_result.to_csv(save_path_study,index=False)

def save_paper_comms(comms, df_paper, paperfile, studyfile):
    """Saves community detection result for paper and study bipartite graph.

    Parameters
    ----------
    comms: cdlib.classes.bipartite_node_clustering.BiNodeClustering
        Network object with community information
    df_paper: pandas.core.frame.DataFrame
        Dataframe with paper information columns
    paperfile: string
       Name of the file to save the community detection result for papers
    studyfile: string
       Name of the file to save the community detection result for studies
    Returns
    ------
    """
    
    comms_dict = comms.to_node_community_map()
    df = pd.DataFrame(comms_dict.items(), columns=['id', 'code'])
    df['community'] = df['code'].str[0]
    df = df.drop(columns=['code'])

    m = df['id'].str.contains('_', na=False)

    papers = df[m].reset_index(drop=True)
    papers['REF_PAPER'] = papers['id']
    papers = papers.drop(columns=['id'])
    
    studies = df[~m].reset_index(drop=True)
    studies['STUDY'] = studies['id']
    studies = studies.drop(columns=['id'])

    paper_result = pd.merge(papers, df_paper, on="REF_PAPER", how="left").drop_duplicates(subset=['REF_PAPER'])
    paper_result = paper_result[['community',
                               'REF_PAPER',
                               'TITLE',
                               'STUDY',
                               'NAME',
                               'FUNDER',
                               'YEAR_PUB',
                               'RIS_TYPE',
                               'research_org_names',
                               'concepts',
                               'category_for']]

    study_result = pd.merge(studies, df_paper, on="STUDY", how="left").drop_duplicates(subset=['STUDY'])
    study_result = study_result[['community',
                             'STUDY',
                             'NAME',
                             'SERIES_TITLE',
                             'SERIES',
                             'OWNER',
                             'FUNDINGAGENCY',
                             'GEO',
                             'TERMS',
                             'ORIGRELDATE']]
    
    save_path_paper = OUTPUTS_PATH + paperfile +'.csv'
    save_path_study = OUTPUTS_PATH + studyfile +'.csv'
    paper_result.to_csv(save_path_paper,index=False)
    study_result.to_csv(save_path_study,index=False)    
    
# DETECTING COMMUNITIES

def detect_communities(G, algorithm):
    """Applies a community detection algorithm to a network.
    Parameters
    ----------
    G : networkx.classes.graph.Graph
        Graph
    algorithm: str
        Name of a community detection algorithm defined in CDLib or Networkx. 
            CDLib options:  girvan_newman (level 3), spectral (kmax 50)
    Returns
    ------
    communities: list
        A list of nodes belonging to each community.
    """
    
    if algorithm=="cdlib_girvan_newman":
        girvan_newman_comp = girvan_newman(G, level=3)
        communities = girvan_newman_comp.communities
        print("Communities detected: ", len(communities))
    elif algorithm=="cdlib_spectral":
        spectral_comp = spectral(G, kmax=50)
        communities = spectral_comp.communities
        print("Communities detected: ", len(communities))   
    else:
        print("Algorithm not found. No communities detected.")
    
    return communities

def make_communities_df(communities, df, filename):
    """Makes a dataframe of nodes and their communities.

    Parameters
    ----------
    communities: list
        A list of nodes belonging to each community
    df: pandas.core.frame.DataFrame
        Dataframe with node information columns
    filename: str
        Name to save plot

    Returns
    ------
    result: pandas.core.frame.DataFrame
        Dataframe of nodes, communities, and attribute information
        
    """
    
    df_communities = pd.DataFrame(communities)
    df_communities = df_communities.reset_index().rename(columns={'index':'community'})
    df_communities = pd.melt(df_communities, id_vars=['community'])
    df_communities = df_communities[df_communities.value.notna()].rename(columns={'value':'AUTHOR_ID'})
    df_communities = df_communities[['community','AUTHOR_ID']].sort_values(by=['community']).reset_index().drop(columns=['index'])

    result = pd.merge(df_communities, df, on="AUTHOR_ID", how="left")
    result = result[['community',
                 'degree', 
                 'AUTHOR', 
                 'AUTHOR_ID', 
                 'TITLE', 
                 'YEAR_PUB', 
                 'RIS_TYPE', 
                 'FUNDER', 
                 'concepts', 
                 'times_cited', 
                 'times_cited', 
                 'category_for', 
                 'funders']]
    
    result = result.dropna(subset=['concepts'])
    result['concepts'] = result['concepts'].map(lambda x: x.lstrip('[').rstrip(']'))
    comm_terms = result[result['concepts'].notna()]
    
    def label_community(group):
        """Function within function to make community labels.
        """
        
        comm = comm_terms[comm_terms.community == group]

        label = []

        for index, row in comm.iterrows():
            term = row.concepts.split(',')
            for entry in term:
                label.append(entry)

        occur = Counter(label)
        top_terms = occur.most_common(5)
        top_terms_elements = [term[0] for term in top_terms]
        return ','.join(top_terms_elements).strip()

    to_label = list(set(comm_terms.community))

    comms = {}

    for item in to_label:
        comm_label = label_community(item)
        comms[item] = comm_label

    result['degree'] = result['degree'].str.extract('(\d+)')
    result['degree'] = result['degree'].astype(int)
    result['label_terms'] = result['community'].map(comms)
    result['color'] = pd.Categorical(result['community'])

    save_path_result = OUTPUTS_PATH + filename +'.csv'
    result.to_csv(save_path_result,index=False)
    return result

def group_communities(result, filename):
    """Groups communities in a summary dataframe.

    Parameters
    ----------
    result: pandas.core.frame.DataFrame
        Dataframe with communities, degree, labels columns
    filename: str
        Name to save plot

    Returns
    ------
    subgraph: networkx.classes.graph.Graph
        Graph
    """
    
    deg_sum = result.groupby(['community'])['degree'].sum().reset_index()
    result['members'] = result.groupby(['community'])['AUTHOR_ID'].transform('size')

    result = result.drop_duplicates(subset=['community'],keep="first")
    result.merge(deg_sum, on='community', how="inner")

    grouped = result[['community','members', 'degree','label_terms']]
    save_path_result = OUTPUTS_PATH + filename +'.csv'
    grouped.to_csv(save_path_result,index=False)
    return grouped

def plot_communities(G, result, filename):
    """Plots a subgraph with nodes colored by clique.

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        Graph
    result: pandas.core.frame.DataFrame
        Dataframe with communities and labels columns
    filename: str
        Name to save plot

    Returns
    ------
    """
    
    sub_nodes = result['AUTHOR_ID']
    subgraph = G.subgraph(sub_nodes)
    print(nx.info(subgraph))

    comm_dict = dict(zip(result.AUTHOR_ID, result.community))
    labels_dict = dict(zip(result.AUTHOR_ID, result.label_terms))
    colors_dict = dict(zip(result.AUTHOR_ID, result.color))
    
    nx.set_node_attributes(subgraph, comm_dict, 'community')
    nx.set_node_attributes(subgraph, labels_dict, 'label')
    nx.set_node_attributes(subgraph, colors_dict, 'color')
    
    fig = plt.figure(1, figsize=(20, 20), dpi=50)
    pos = nx.spring_layout(subgraph, k=0.9)
    node_color = [subgraph.nodes[v]['color'] for v in subgraph.nodes]
    degree = [subgraph.nodes[v]['degree'] for v in subgraph.nodes]
    
    plt.rcParams.update(plt.rcParamsDefault)
    
    nx.draw_networkx_nodes(subgraph,
                            pos=pos,
                            node_size=50,
                            node_color=node_color,
                            cmap=cc.cm.glasbey_bw)             
    
    nx.draw_networkx_edges(subgraph, 
                            pos=pos,
                            width=0.5,
                            edge_color="silver",
                            alpha=0.5) 
    
    pos_higher = {}
    y_off = .02

    for k, v in pos.items():
        pos_higher[k] = (v[0], v[1]+y_off)

    nx.draw_networkx_labels(subgraph, pos=pos_higher)
    
    # save_path = FIGURES_PATH + filename +'.png'
    # plt.savefig(save_path)
    plt.show()
    return subgraph

# DETECTING CLIQUES
def detect_cliques(G, k):
    """Applies k-clique percolation community detection algorithm to a graph.

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        Graph
    k: int
        Minimum clique size
    Returns
    ------
    communities: list
        A list of nodes belonging to each community.
    """
    
    kclique_comp = community.kclique.k_clique_communities(G, k)
    communities = sorted(kclique_comp, key=len, reverse=True)
    print(f"Communities detected at k of {k}: ", len(communities))
    return communities

def make_cliques_df(communities, identity, df, filename):    
    """Makes a dataframe of nodes and their cliques.

    Parameters
    ----------
    communities: list
        A list of nodes belonging to each community
    identity: str
        Name of the node identity type (options: "study", "data")
    df: pandas.core.frame.DataFrame
        Dataframe with node information columns
    filename: str
        Name to save plot

    Returns
    ------
    result: pandas.core.frame.DataFrame
        Dataframe of nodes, communities, and attribute information
        
    """
    
    df_communities = pd.DataFrame(communities)
    df_communities = df_communities.reset_index().rename(columns={'index':'community'})
    df_communities = pd.melt(df_communities, id_vars=['community'])
    
    if identity == "study":
        df_communities = df_communities[df_communities.value.notna()].rename(columns={'value':'STUDY'})
        df_communities = df_communities[['community','STUDY']].sort_values(by=['community']).reset_index().drop(columns=['index'])

        result = pd.merge(df_communities, df, on="STUDY", how="left")
        result = result[['community',
                         'degree',
                         'STUDY',
                         'NAME',
                         'SERIES',
                         'SERIES_TITLE',
                         'OWNER',
                         'FUNDINGAGENCY',
                         'GEO',
                         'TERMS',
                         'ORIGRELDATE']]
        
    else:
        df_communities = df_communities[df_communities.value.notna()].rename(columns={'value':'REF_DATA'})
        df_communities = df_communities[['community','REF_DATA']].sort_values(by=['community']).reset_index().drop(columns=['index'])

        result = pd.merge(df_communities, df, on="REF_DATA", how="left")
        result = result[['community',
                         'degree',
                         'REF_DATA',
                         'STUDY',
                         'NAME',
                         'SERIES',
                         'SERIES_TITLE',
                         'OWNER',
                         'FUNDINGAGENCY',
                         'GEO',
                         'TERMS',
                         'ORIGRELDATE']]
    
    comm_terms = result[result['TERMS'].notna()]
    
    def label_community(group):
        """Function within function to make clique labels.
        """
        
        comm = comm_terms[comm_terms.community == group]

        label = []

        for index, row in comm.iterrows():
            term = row.TERMS.split(';')
            for entry in term:
                label.append(entry)

        occur = Counter(label)
        top_terms = occur.most_common(3)
        top_terms_elements = [term[0] for term in top_terms]
        return ','.join(top_terms_elements).strip()

    to_label = list(set(comm_terms.community))

    comms = {}

    for item in to_label:
        comm_label = label_community(item)
        comms[item] = comm_label

    result['degree'] = result['degree'].str.extract('(\d+)')
    result['degree'] = pd.to_numeric(result['degree'], errors='coerce')
    result = result.dropna(subset=['degree'])
    result['degree'] = result['degree'].astype(int)
    result['label_terms'] = result['community'].map(comms)
    if identity == "study":
        result['multi'] = result.duplicated(subset='STUDY', keep=False)
    else:
        result['multi'] = result.duplicated(subset='REF_DATA', keep=False)
    result['color'] = np.where(result['multi']==True, -1, result['community'])
    result['color'] = pd.Categorical(result['color'])

    save_path_result = OUTPUTS_PATH + filename +'.csv'
    result.to_csv(save_path_result,index=False)
    return result

def group_cliques(result, identity, filename):
    """Groups cliques in a summary dataframe.

    Parameters
    ----------
    result: pandas.core.frame.DataFrame
        Dataframe with communities, degree, labels columns
    identity: str
        Name of node type (options: "study", "data")
    filename: str
        Name to save plot

    Returns
    ------
    subgraph: networkx.classes.graph.Graph
        Graph
    """
    
    deg_sum = result.groupby(['community'])['degree'].sum().reset_index()
    
    if identity == "study":
        result['members'] = result.groupby(['community'])['STUDY'].transform('size')
    else:
        result['members'] = result.groupby(['community'])['REF_DATA'].transform('size')

    result = result.drop_duplicates(subset=['community'],keep="first")
    result.merge(deg_sum, on='community', how="inner")

    grouped = result[['community','members','label_terms']]
    save_path_result = OUTPUTS_PATH + filename +'.csv'
    grouped.to_csv(save_path_result,index=False)
    return grouped

def plot_cliques(G, identity, result, filename):
    """Plots a subgraph with nodes colored by clique.

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        Graph
    identity: str
        Name of node type (options: "study", "data")
    result: pandas.core.frame.DataFrame
        Dataframe with communities and labels columns
    filename: str
        Name to save plot

    Returns
    ------
    """
    
    if identity == "study":
        sub_nodes = result['STUDY']
    else:
        sub_nodes = result['REF_DATA']
    
    subgraph = G.subgraph(sub_nodes)
    print(nx.info(subgraph))

    if identity == "study":
        comm_dict = dict(zip(result.STUDY, result.community))
        labels_dict = dict(zip(result.STUDY, result.label_terms))
        colors_dict = dict(zip(result.STUDY, result.color))
    else:
        comm_dict = dict(zip(result.REF_DATA, result.community))
        labels_dict = dict(zip(result.REF_DATA, result.label_terms))
        colors_dict = dict(zip(result.REF_DATA, result.color))
    
    nx.set_node_attributes(subgraph, comm_dict, 'community')
    nx.set_node_attributes(subgraph, labels_dict, 'label')
    nx.set_node_attributes(subgraph, colors_dict, 'color')
    
    fig = plt.figure(1, figsize=(30, 30), dpi=50)
    
    pos = nx.spring_layout(subgraph, k=3, iterations=1000, seed=42)
    
    node_color = [subgraph.nodes[v]['color'] for v in subgraph.nodes]
    degree = [subgraph.nodes[v]['degree'] for v in subgraph.nodes]
    plt.rcParams.update(plt.rcParamsDefault)
    
    nx.draw_networkx_nodes(subgraph,
                            pos=pos,
                            node_size=60,
                            node_color=node_color,
                            cmap=cc.cm.glasbey)              
    
    nx.draw_networkx_edges(subgraph, 
                            pos=pos,
                            width=0.5,
                            edge_color="#444444")
    
    pos_higher = {}
    y_off = .02
    
    for k, v in pos.items():
        pos_higher[k] = (v[0], v[1]+y_off)

    nx.draw_networkx_labels(subgraph, pos=pos_higher)
    
    # save_path = FIGURES_PATH + filename +'.png'
    # plt.savefig(save_path)
    plt.show()
    return subgraph

def save_gexf(G, filename):
    """Saves a gexf file for visualization in Gephi.

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        Graph
    filename: str
        Name to save plot

    Returns
    ------
    """

    save_path_result = OUTPUTS_PATH + filename +'.gexf'
    nx.write_gexf(G, save_path_result, encoding='utf-8', prettyprint=False, version='1.2draft')
