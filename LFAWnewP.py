import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import random
import datetime
import time
from collections import defaultdict
# from aquarel import load_theme
import math
import seaborn
import cdlib
from cdlib import viz, algorithms, ensemble, readwrite
import leidenalg
import igraph as ig
from IPython.display import display

#######################################
# HELPER FUNCTIONS FOR DISTRIBUTIONS  #
#######################################


def get_distributions(G):
    """
    Compute basic distributions from graph G.
    Returns a dictionary containing:
      - in_degrees: List of in-degrees for all nodes.
      - out_degrees: List of out-degrees for all nodes.
      - total_degrees: Sum of in- and out-degrees for each node.
      - cycle_lengths: List of sizes for strongly connected components (SCCs) of size > 1.
      - descendants_counts: Number of nodes reachable from each node.
      - num_of_nodes: Total number of nodes in G.
      - scc_size: Size (number of nodes) of the largest SCC.
    """
    in_degrees = [G.in_degree(n) for n in G.nodes()]
    out_degrees = [G.out_degree(n) for n in G.nodes()]
    total_degrees = [in_deg + out_deg for in_deg, out_deg in zip(in_degrees, out_degrees)]
    


    sccs = list(nx.strongly_connected_components(G))
    cycle_lengths = [len(scc) for scc in sccs if len(scc) > 1]
    largest_scc_size = max((len(scc) for scc in sccs), default=0)
    # descendants_counts = [len(nx.descendants(G, n)) for n in G.nodes()]
    
    return {
        'in_degrees': in_degrees,
        'out_degrees': out_degrees,
        'total_degrees': total_degrees,
        'cycle_lengths': cycle_lengths,
        # 'descendants_counts': descendants_counts,
        'num_of_nodes': G.number_of_nodes(),
        'largest_scc_size': largest_scc_size
    }

def build_signed_adjacency(G):
    """
    Given a DiGraph or MultiDiGraph G with edge-attribute 'relation' in {'help','harm'},
    return its signed adjacency matrix A (help=+1, harm=–1) and a node→index map.
    """
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=float)
    for u, v, d in G.edges(data=True):
        w =  1 if d.get('relation') == 'help' else -1
        A[idx[u], idx[v]] += w
    return A, idx


# threshold helper function

def pair_threshold(is_help: bool, bind_range: float, help_hurt: float) -> float:
    """Random threshold for *this* ordered pair (i,j)."""
    k = np.random.poisson(1)            # edge-wise heterogeneity
    base = bind_range * (1 - help_hurt) if is_help else bind_range * help_hurt
    return k * base                     
#######################################
# SIMULATION FUNCTION
#######################################

def simulate_network(num_steps, bind_range, seed, lifespan, help_hurt):
    """
    Builds an adaptive threshold network over num_steps.
    Each new node is assigned a 4-dimensional potential (help_in, help_out, harm_in, harm_out).
    Edges are added based on threshold comparisons, and nodes are removed based on lifespan and 
    the balance between harm and help.
    Returns the final graph and various simulation statistics.
    """
    random.seed(seed)
    np.random.seed(seed)

    G = nx.MultiDiGraph()
    HELP = 0
    HARM = 0
    # Lists to record dynamic statistics over simulation
    kappa_eff_series = []
    node_counts = []
    edge_counts = []
    spectral_radii = []
    spectral_radii_abs = []
    spectral_angles = []
    spectral_gaps = []
    cycle_fractions = []   # Fraction of nodes in cycles (SCCs with size > 1)
    largest_scc_fraction = []   # Fraction of nodes in the largest SCC
    largest_scc_size_series = []
    crossing_step = None
    v_prev_s = None

    #to help with death calculation
    help_in_cum = defaultdict(int)
    harm_in_cum = defaultdict(int)
    expiry_schedule = defaultdict(list)
    already_crossed = False

    # avg_descendants_over_time = []

    # Generate a threshold array from Poisson, then set separate thresholds for help and harm.
    # threshold = np.random.poisson(1, num_steps) * bind_range
    # help_threshold = threshold * (1 - help_hurt)
    # harm_threshold = threshold * help_hurt

    # starting_weight = random.random()

    # Main simulation loop: each iteration adds one new node and edges to existing nodes.
    for step in range(num_steps):
        new_node = step  # Unique node ID based on simulation step

        # Generate node potentials as random floats
        help_in = random.random()
        help_out = random.random()
        harm_in = random.random()
        harm_out = random.random()

        birth = step
        to_remove = []

        

        # Add new node with its attributes
        G.add_node(new_node, 
                   help_in=help_in, help_out=help_out,
                   harm_in=harm_in, harm_out=harm_out,
                   lifespan=lifespan)

        G.nodes[new_node]['help_in_cum'] = 0
        G.nodes[new_node]['harm_in_cum'] = 0
        G.nodes[new_node]['birth_step'] = step
        # print(G.nodes[new_node]['birth_step']) #debug

        # Choose a random number of edges to add between nodes
        multi_edge = random.randint(1, 6)

        # Iterate over all existing nodes (excluding the new node)
        for node in list(G.nodes()):
            if node == new_node or node not in G:
                continue

            node_data = G.nodes[node]
            new_node_data = G.nodes[new_node]

            # --- Add HARM edges if threshold met ---
            if abs(node_data['harm_in'] - new_node_data['harm_out']) \
       < pair_threshold(is_help=False, bind_range=bind_range, help_hurt=help_hurt):
                for _ in range(multi_edge):
                    G.add_edge(new_node, node, relation='harm', color='red')
                    harm_in_cum[node] += 1
                    HARM += 1
                age = step - G.nodes[node]['birth_step']
                # print(step-G.nodes[node]['birth_step']," ok ")
                if age >= lifespan:
                    total = help_in_cum[node] + harm_in_cum[node]
                    if total>0 and (harm_in_cum[node]/total) > 0.5:
                            to_remove.append(node)
            if abs(node_data['harm_out'] - new_node_data['harm_in']) \
       < pair_threshold(is_help=False, bind_range=bind_range, help_hurt=help_hurt):
                for _ in range(multi_edge):
                    G.add_edge(node, new_node, relation='harm', color='red')
                    HARM += 1

            # --- Add HELP edges if threshold met ---
            if abs(node_data['help_in'] - new_node_data['help_out']) \
       < pair_threshold(is_help=True, bind_range=bind_range, help_hurt=help_hurt):
                for _ in range(multi_edge):
                    G.add_edge(new_node, node, relation='help', color='blue')
                    help_in_cum[node] += 1
                    HELP +=1
                age = step - G.nodes[node]['birth_step']
                if age >= lifespan:
                    total = help_in_cum[node] + harm_in_cum[node]
                    if total>0 and (harm_in_cum[node]/total) > 0.5:
                            to_remove.append(node)
            if abs(node_data['help_out'] - new_node_data['help_in']) \
       < pair_threshold(is_help=True, bind_range=bind_range, help_hurt=help_hurt):
                for _ in range(multi_edge):
                    G.add_edge(node, new_node, relation='help', color='blue')
                    HELP +=1

            # --- Lifespan check and potential removal ---
            age = step - G.nodes[node]['birth_step']
            if age >= lifespan and (G.in_degree(node) == 0):
                to_remove.append(node)

            for node in to_remove:
                if node in G:
                    G.remove_node(node)




        
      





        # Record dynamic statistics for this step
        node_counts.append(G.number_of_nodes())
        edge_counts.append(G.number_of_edges())

        kappa_eff = (G.number_of_edges() / G.number_of_nodes())**2
        kappa_eff_series.append(kappa_eff)

        if kappa_eff >= 1 and not already_crossed:
            print(f"Critical Point crossed at step {step}")
            already_crossed = True


        #######################
        ###Spectral analysis###
        #######################
        
        A_t, idx_t = build_signed_adjacency(G)
        n_t = A_t.shape[0]

        #adaptive power-iteration: set tolerance here
        max_iters = 20
        tol = 1e-6



        vec = np.random.rand(n_t)
        prev_lam = 0.0

        for i in range(1, max_iters+1):
            vec = A_t.dot(vec)
            vec /= np.linalg.norm(vec)
            lam = vec.dot(A_t.dot(vec))
            if abs(lam - prev_lam) < tol:
                break
            prev_lam = lam
        lambda1_t = lam #rayleigh quotient

        #get 2nd eigenvalue

        B = A_t - lambda1_t * np.outer(vec,vec)
        vec2 = np.random.rand(n_t)
        prev_mu = 0.0
        for i in range(1, max_iters+1):
            vec2 = B.dot(vec2)
            vec2 /= np.linalg.norm(vec2)
            mu = vec2.dot(A_t.dot(vec2))
            if abs(mu - prev_mu) < tol:
                break
            prev_mu = mu
        lambda2_t = mu #rayleigh 2

        s_curr = pd.Series(vec, index = idx_t)

        if v_prev_s is None:
            theta_t = 0.0
        else:
            all_idx = v_prev_s.index.union(s_curr.index)
            u = v_prev_s.reindex(all_idx, fill_value=0.0).values
            w = s_curr.reindex(all_idx,    fill_value=0.0).values
            # compute principal angle
            cosθ = np.dot(u, w) / (np.linalg.norm(u)*np.linalg.norm(w))
            theta_t = np.arccos(np.clip(cosθ, -1.0, 1.0))

        v_prev_s = s_curr


        abs_lambda1 = abs(lambda1_t)

        # now record everything:
        spectral_radii.append(lambda1_t)
        spectral_radii_abs.append(abs_lambda1)
        spectral_angles.append(theta_t)
        spectral_gaps.append(abs_lambda1 - abs(lambda2_t))





        sccs = list(nx.strongly_connected_components(G))
        nodes_in_cycle = {n for scc in sccs if len(scc) > 2 for n in scc}
        cycle_fraction = len(nodes_in_cycle) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        cycle_fractions.append(cycle_fraction)
        
        if sccs and step > 20:
            largest = max(sccs, key=len)
            largest_scc_size = len(largest)
            largest_frac = len(largest) / G.number_of_nodes()
        else:
            largest_frac = 0
            largest_scc_size = 0

        largest_scc_fraction.append(largest_frac)
        largest_scc_size_series.append(largest_scc_size)

        # Record crossing step when largest SCC covers >= 50% of nodes
        if crossing_step is None and largest_frac >= 0.5 and step > 10:
            crossing_step = step
        print("Step:", step, " completed!")

    # --- Create Subgraphs for HELP and HARM edges ---
    help_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == 'help']
    harm_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == 'harm']

    help_subgraph = nx.MultiDiGraph()
    help_subgraph.add_edges_from(help_edges)

    harm_subgraph = nx.MultiDiGraph()
    harm_subgraph.add_edges_from(harm_edges)

    # --- Compute distributions over the final graphs ---
    dist_dict = get_distributions(G)
    dist_help = get_distributions(help_subgraph)
    dist_harm = get_distributions(harm_subgraph)

    print(" help edges: ", HELP)
    print(" harm edges: ", HARM)

    # Save graph files for Gephi visualization 
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"STATS_{timestamp}.txt"
    nx.write_gexf(G, "combined_graph_" + filename)
    nx.write_gexf(help_subgraph, "help_" + filename)
    # nx.write_gexf(harm_subgraph, "harm_" + filename)
    sorted_nodes = sorted(G.nodes(), key=lambda n: G.in_degree(n), reverse=True)




    with open(filename, "w") as f:
        f.write(f"numsteps = {num_steps}, lifespan={lifespan}, bind_range={bind_range}\n\n")
        for u in sorted_nodes:
            age = num_steps - G.nodes[u]['birth_step']
            f.write(f"{u} has age {age} and degree {G.in_degree(u)}\n" )
    print(f"Saved stats to {filename}")



    #-----Age Degree Histogram------
    # #1) Grab all nodes and sort by total degree (in+out) descending
    # all_nodes = list(G.nodes())
    # sorted_by_deg = sorted(all_nodes, key=lambda u: G.in_degree(u), reverse=True)

    # # 2) Compute top‐10% cutoff
    # N_total = len(all_nodes)
    # top_n   = max(1, math.ceil(0.10 * N_total))
    # top_nodes = sorted_by_deg[:top_n]

    # #3) Extract their degrees and ages
    # top_degrees = [G.in_degree(u) for u in top_nodes]
    # top_ages    = [num_steps - G.nodes[u]['birth_step'] for u in top_nodes]

    # plt.figure(figsize=(8,6))
    # plt.hist(top_degrees, bins=10, edgecolor='k', alpha=0.7)
    # plt.xlabel('Degree')
    # plt.ylabel('Count')
    # plt.title(f'Distribution of Nodes in Top 10% In-Degree (n={top_n})')
    # plt.tight_layout()
    # plt.savefig(f"HH={help_hurt}_LS={lifespan}_n={num_steps}_bc={bind_range}_top10degree.png")
    

    # # ——— Plot age histogram ———
    # plt.figure(figsize=(8,6))
    # plt.hist(top_ages, bins=10, edgecolor='k', alpha=0.7)
    # plt.xlabel('Age (steps)')
    # plt.ylabel('Count')
    # plt.title(f'Age dist. of Top 10% nodes (n={top_n})')
    # plt.tight_layout()
    # # plt.ylim(ymax=3.0)
    # # plt.xlim(xmin = 0, xmax=3.5)
    # plt.savefig(f"HH={help_hurt}_LS={lifespan}_n={num_steps}_bc={bind_range}_top10age.png")


    # theme = load_theme("scientific")
    # theme.apply()
    # ages   = [num_steps - G.nodes[u]['birth_step'] for u in sorted_nodes]
    # degrees = [G.in_degree(u)                         for u in sorted_nodes]

    # log_ages = np.log10(ages)
    # log_degrees = np.log10(degrees)

    # Make the scatter
    # plt.figure(figsize=(8,6))
    # plt.scatter(log_ages, log_degrees, alpha=0.6)
    # plt.xlabel('Age (steps since birth)')
    # plt.ylabel('Degree (in + out)')
    # plt.title(f'Age vs. Degree (n={num_steps}, life={lifespan}, bind={bind_range}), harm/help={help_hurt}')
    # plt.xlim(xmax=3.5)
    # plt.ylim(ymax=3.0)
    # plt.tight_layout()

    # filename2 = f"SCATTERplot_lifespan{lifespan}_bind{bind_range}_helphurt{help_hurt}_num_steps{num_steps}.png"
    # plt.savefig(filename2)
    
    # theem.apply_transforms()

    

    

    # #attach to dist_dict for return
    # dist_dict.update({
    #     'adjacency_matrix': A,
    #     'jaccard_distance_self': jaccard_dist,
    #     'eigvalues': eigvals,
    #     'principal_eigvec': principal_vec,
    #     'node_index_map': idx
    #     })

    return (G, node_counts, edge_counts, cycle_fractions,
            largest_scc_fraction, crossing_step, largest_scc_size_series,
            dist_dict, dist_harm, dist_help, spectral_radii, spectral_angles, spectral_radii_abs, spectral_gaps, kappa_eff_series)

#######################################
# BATCH RUNNER
#######################################

def run_batches(num_batches, num_steps, bind_range, seed, lifespan, help_hurt):
    """
    Runs multiple simulation batches and aggregates their statistics.
    Returns a dictionary of overall statistics.
    """
    all_node_counts = []
    all_edge_counts = []
    all_cycle_fractions = []
    all_largest_scc_fraction = []
    all_largest_scc_size     = []
    all_crossing_steps = []

    # Accumulators for distributions from main, help, and harm graphs
    all_in_degrees = []
    all_out_degrees = []
    all_total_degrees = []
    all_cycle_length_dist = []
    

    help_in_degrees = []
    help_out_degrees = []
    help_total_degrees = []
    help_cycle_length_dist = []
    # help_descendants_dist_final = []

    harm_in_degrees = []
    harm_out_degrees = []
    harm_total_degrees = []
    harm_cycle_length_dist = []
    # harm_descendants_dist_final = []
    all_spectral_radii = []
    all_spectral_radii_abs = []
    all_spectral_angles = []
    all_spectral_gaps = []
    all_kappa_eff = []

    for batch in range(num_batches):
        batch_seed = seed + batch + random.randrange(1, 100)
        (G, node_counts, edge_counts, cycle_fractions, largest_scc_fraction,
         crossing_step, largest_scc_size_series, dist_dict, dist_harm, dist_help, spectral_radii, spectral_angles, spectral_radii_abs, spectral_gaps, kappa_eff_series) = simulate_network(
            num_steps=num_steps, bind_range=bind_range, seed=batch_seed,
            lifespan=lifespan, help_hurt=help_hurt
        )
        all_node_counts.append(node_counts)
        all_edge_counts.append(edge_counts)
        all_cycle_fractions.append(cycle_fractions)
        all_largest_scc_fraction.append(largest_scc_fraction)
        all_crossing_steps.append(crossing_step if crossing_step is not None else np.nan)
        all_largest_scc_size.append(largest_scc_size_series)

        all_spectral_radii.append(spectral_radii)
        all_spectral_radii_abs.append(spectral_radii_abs)
        all_spectral_angles.append(spectral_angles)
        all_spectral_gaps.append(spectral_gaps)
        all_kappa_eff.append(kappa_eff_series)

        

        # Accumulate distributions from the main graph
        all_in_degrees.append(dist_dict['in_degrees'])
        all_out_degrees.append(dist_dict['out_degrees'])
        all_total_degrees.append(dist_dict['total_degrees'])
        all_cycle_length_dist.append(dist_dict['cycle_lengths'])
        # all_descendants_dist_final.append(dist_dict['descendants_counts'])
        # all_sccs_final.append(dist_dict['scc_size'])

        # Accumulate from the help subgraph
        help_in_degrees.append(dist_help['in_degrees'])
        help_out_degrees.append(dist_help['out_degrees'])
        help_total_degrees.append(dist_help['total_degrees'])
        help_cycle_length_dist.append(dist_help['cycle_lengths'])
        # help_descendants_dist_final.append(dist_help['descendants_counts'])

        # Accumulate from the harm subgraph
        harm_in_degrees.append(dist_harm['in_degrees'])
        harm_out_degrees.append(dist_harm['out_degrees'])
        harm_total_degrees.append(dist_harm['total_degrees'])
        harm_cycle_length_dist.append(dist_harm['cycle_lengths'])
        # harm_descendants_dist_final.append(dist_harm['descendants_counts'])

    # Convert aggregated data to NumPy arrays for statistics
    all_node_counts = np.array(all_node_counts)
    all_edge_counts = np.array(all_edge_counts)
    all_cycle_fractions = np.array(all_cycle_fractions)
    all_largest_scc_fraction = np.array(all_largest_scc_fraction)
    all_crossing_steps = np.array(all_crossing_steps)

    all_spectral_radii = np.array(all_spectral_radii)
    all_spectral_radii_abs = np.array(all_spectral_radii_abs)
    all_spectral_angles = np.array(all_spectral_angles)
    all_spectral_gaps = np.array(all_spectral_gaps)
    
    all_largest_scc_size = np.array(all_largest_scc_size)
    all_kappa_eff = np.array(all_kappa_eff)

    mean_node_counts = np.mean(all_node_counts, axis=0)
    std_node_counts = np.std(all_node_counts, axis=0)
    mean_edge_counts = np.mean(all_edge_counts, axis=0)
    std_edge_counts = np.std(all_edge_counts, axis=0)
    mean_cycle_fractions = np.mean(all_cycle_fractions, axis=0)
    std_cycle_fractions = np.std(all_cycle_fractions, axis=0)
    mean_largest_scc_fraction = np.mean(all_largest_scc_fraction, axis=0)
    std_largest_scc_fraction = np.std(all_largest_scc_fraction, axis=0)
    mean_crossing_step = np.mean(all_crossing_steps, axis=0)
    std_crossing_step = np.std(all_crossing_steps, axis=0)

    mean_scc_size = np.mean(all_largest_scc_size, axis = 0)
    std_scc_size = np.std(all_largest_scc_size, axis = 0)

    # A = dist_dict['adjacency_matrix']
    # print("Shape of A:", A.shape)
    # print("Top 5 eigenvalues:", dist_dict['eigvalues'][:5])
    # print("Principal eigenvector (first 5 components):",
    #       dist_dict['principal_eigvec'][:5])

    mean_lambda = np.mean(all_spectral_radii, axis = 0)
    std_lambda = np.std(all_spectral_radii, axis=0)

    mean_lambda_abs = np.mean(all_spectral_radii_abs, axis = 0)
    std_lambda_abs = np.std(all_spectral_radii_abs, axis=0)

    mean_angle = np.mean(all_spectral_angles, axis = 0)
    std_angle = np.std(all_spectral_angles, axis=0)

    mean_gaps = np.mean(all_spectral_gaps, axis = 0)
    std_gaps = np.mean(all_spectral_gaps)

    return {
        'mean_gaps': mean_gaps,
        'std_gaps': std_gaps,
        'mean_lambda_abs': mean_lambda_abs,
        'std_lambda_abs': std_lambda_abs,
        'mean_angle': mean_angle,
        'std_angle': std_angle,
        'mean_node_counts': mean_node_counts,
        'std_node_counts': std_node_counts,
        'mean_edge_counts': mean_edge_counts,
        'std_edge_counts': std_edge_counts,
        'mean_cycle_fractions': mean_cycle_fractions,
        'std_cycle_fractions': std_cycle_fractions,
        'mean_largest_scc_fraction': mean_largest_scc_fraction,
        'std_largest_scc_fraction': std_largest_scc_fraction,
        'mean_crossing_step': mean_crossing_step,
        'std_crossing_step': std_crossing_step,
        'all_in_degrees': all_in_degrees,
        'all_out_degrees': all_out_degrees,
        'all_total_degrees': all_total_degrees,
        'all_cycle_length_dist': all_cycle_length_dist,
        # 'all_descendants_dist_final': all_descendants_dist_final,
        'help_in_degrees': help_in_degrees,
        'help_out_degrees': help_out_degrees,
        'help_total_degrees': help_total_degrees, 
        'help_cycle_length_dist': help_cycle_length_dist,
        # 'help_descendants_dist_final': help_descendants_dist_final,
        'harm_in_degrees': harm_in_degrees,
        'harm_out_degrees': harm_out_degrees,
        'harm_cycle_length_dist': harm_cycle_length_dist,
        'num_nodes_remaining': dist_dict['num_of_nodes'],
        'mean_scc_size': mean_scc_size,
        'std_scc_size': std_scc_size,
        'mean_lambda': mean_lambda,
        'std_lambda': std_lambda,
        'all_kappa_eff': all_kappa_eff

    }

#######################################
# PLOTTING STATISTICS
#######################################

def plot_batch_statistics(stats):
    """
    Generate plots of network statistics over simulation steps.
    """
    steps = range(len(stats['mean_node_counts']))
    
    SMALL_SIZE = 11
    MEDIUM_SIZE = 11
    BIGGER_SIZE = 11
    # theme = load_theme("arctic_light")
    # theme.apply()
    # theme.apply_transforms()

    plt.style.use('fivethirtyeight')
    mpl.rcParams['lines.linewidth'] = 2
    # plt.style.use('seaborn-v0_8-bright')

    

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    




    f1 = plt.figure(figsize=(8.5,10.5))

    # # Plot: Nodes and Edges Over Time
    plt.subplot(5, 2, 1)
    plt.errorbar(steps, stats['mean_node_counts'], yerr=stats['std_node_counts'],
                 label='Nodes', color = '#E69F00', alpha=0.7)
    plt.errorbar(steps, stats['mean_edge_counts'],color = '#000000', yerr=stats['std_edge_counts'],
                 label='Edges', alpha=0.7)
    plt.xlabel('Steps')
    plt.ylabel('Count')
    plt.title('Nodes and Edges Over Time', fontsize = BIGGER_SIZE)
    plt.legend()
    # plt.ylim(0,16000)

    # Plot: Fraction of Nodes in Cycles (SCCs)
    plt.subplot(5, 2, 2)
    plt.errorbar(steps, stats['mean_cycle_fractions'], yerr=stats['std_cycle_fractions'],
                 color='#56B4E9', alpha=0.7)
    plt.xlabel('Steps')
    plt.ylabel('Fraction in Cycles')
    plt.title('Fraction of Nodes in Cycles (SCCs)', fontsize = BIGGER_SIZE)
    plt.ylim(ymin=-0.02,ymax=1.1)

    
    


    # Plot: Largest SCC Fraction Over Time
    plt.subplot(5, 2, 3)
    plt.errorbar(steps, stats['mean_largest_scc_fraction'], yerr=stats['std_largest_scc_fraction'],
                 color='#009E73', alpha=0.7)
    plt.xlabel('Steps')
    plt.ylabel('Largest SCC Fraction')
    plt.title('Largest Str Conn Comp', fontsize = BIGGER_SIZE)
    plt.ylim(ymin=-0.02,ymax=1.1)

    # #Plot: Largest SCC raw size
    plt.subplot(5, 2, 4)
    plt.errorbar(steps, stats['mean_node_counts'], yerr=stats['std_node_counts'],
                 label='Nodes', color = '#E69F00', alpha=0.7)
    plt.errorbar(steps, stats['mean_scc_size'], yerr=stats['std_scc_size'],
                 color='#0072B2', alpha=0.7)
    plt.xlabel('Steps')
    plt.ylabel('Largest SCC Size')
    plt.title('Largest SCC Size', fontsize = BIGGER_SIZE)
    plt.ylim(ymax = 2500)
    # plt.show()


    # plt.subplot(5, 2, 5)
    # plt.plot(steps, np.ravel(stats['all_kappa_eff']),
    #              color='#009E73', alpha=0.7)
    # plt.xlabel('Steps')
    # plt.ylabel('kappa_eff')
    # plt.title('Kappa Eff', fontsize = BIGGER_SIZE)
    # plt.ylim(ymin=-0.02,ymax=50.0)

    # Plot: In Degree Distribution of Main Graph
    # plt.subplot(5, 4, 6)
    # plt.hist(np.concatenate(stats['all_in_degrees']), bins=50, alpha=0.6,
    #          color='yellow', edgecolor='k')
    # plt.title('In Degree Distribution')
    # plt.xlabel('In Degree')
    # plt.ylabel('Frequency')
    # # plt.ylim(0,3500)
    # # plt.xlim(0,100)

    # # Plot: Out Degree Distribution of Main Graph
    # plt.subplot(4, 4, 7)
    # plt.hist(np.concatenate(stats['all_out_degrees']), bins=50, alpha=0.6,
    #          color='yellow', edgecolor='k')
    # plt.title("Out Degree Distribution")
    # plt.xlabel('Out Degree')
    # plt.ylabel('Frequency')
    # plt.ylim(0,3500)
    # plt.xlim(0,100)

        
    
   
    # Plot: In Degree Distribution for Help and Harm Subgraphs
    plt.subplot(5, 2, 5)
    n_bins = np.linspace(0,150,50)
    plt.hist(np.concatenate(stats['help_in_degrees']), bins=n_bins, alpha=0.5,
             color='#0072B2', edgecolor='#0072B2', label='Help')
    plt.hist(np.concatenate(stats['harm_in_degrees']), bins=n_bins, alpha=0.5,
             color='#D55E00', edgecolor='#D55E00', label='Harm')
    plt.title("Help and Harm In Degree Distribution", fontsize = BIGGER_SIZE)
    plt.xlabel('In Degree')
    plt.ylabel('Frequency')
    plt.legend()
    # plt.ylim(0,3500)
    plt.xlim(xmax=150)
    # # theme.apply_transforms()


    # # Plot: Out Degree Distribution for Help and Harm Subgraphs
    plt.subplot(5, 2, 6)
    plt.hist(np.concatenate(stats['help_out_degrees']), bins=n_bins, alpha=0.6,
             color='#0072B2', edgecolor='#0072B2', label='Help')
    plt.hist(np.concatenate(stats['harm_out_degrees']), bins=n_bins, alpha=0.6,
             color='#D55E00', edgecolor='#D55E00', label='Harm')
    plt.title("Help and Harm Out Degree Distribution", fontsize = BIGGER_SIZE)
    plt.xlabel('Out Degree')
    plt.ylabel('Frequency')
    plt.xlim(xmax=150)
    # # theme.apply_transforms()

    # # plt.legend()

    # # Plot: eigenvalue
    # plt.subplot(5, 2, 8)
    # plt.errorbar(steps, stats['mean_lambda'], yerr=stats['std_lambda'],
    #              color='#D55E00', alpha=0.7)
    # plt.xlabel('Steps')
    # plt.ylabel('Eigenvalue')
    # plt.title('Mean Dominant Eigenvalue', fontsize = BIGGER_SIZE)
    # plt.ylim(ymax=50)


    # plt.subplot(5, 2, 9)
    # plt.errorbar(steps, stats['mean_lambda_abs'], yerr=stats['std_lambda_abs'],
    #              color='#CC79A7', alpha=0.7)
    # plt.xlabel('Steps')
    # plt.ylabel('Eigenvalue')
    # plt.title('Abs Mean Dominant Eigenvalue', fontsize = BIGGER_SIZE)
    # plt.ylim(ymax=50)
   

    # # Plot: spectral angle
    # plt.subplot(5, 2, 9)
    # plt.errorbar(steps, stats['mean_angle'], yerr=stats['std_angle'],
    #              color='#648FFF', alpha=0.7)
    # plt.xlabel('Steps')
    # plt.ylabel('Theta_t')
    # plt.title('Principal Angle', fontsize = BIGGER_SIZE)
    # plt.ylim(ymax = 3.14)

    #  # Plot: spectral gaps
    # plt.subplot(5, 2, 10)
    # plt.errorbar(steps, stats['mean_gaps'], yerr=stats['std_gaps'],
    #              color='#FE6100', alpha=0.7)
    # plt.xlabel('Steps')
    # plt.ylabel('Gap')
    # plt.title('Spectral Gap', fontsize = BIGGER_SIZE)
    # plt.ylim(ymax = 45)
    # # theme.apply_transforms()
    plt.tight_layout()
    # f1.update_annotations()

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    figname = f"eigenanalysis_{timestamp}.png"
    # plt.show()
    f1.savefig(figname, dpi = 300)
    
    
    

    

#######################################
# MAIN FUNCTION
#######################################


def main():
    #Simulation parameters
    num_batches = 1       # Number of independent simulation runs
    num_steps = 1000     # Number of nodes to add in each simulation
    seed = 42              # Base seed for reproducibility
    
    # Parameters to explore
    lifespan_values = [100]            
    bind_range_values = [0.005]          
    help_hurt_values = [0.4]  # Harm/Help ratio values

    # help_hurt_values = [0.5]
    # Run simulations for each parameter combination
    for lifespan in lifespan_values:
        for bind_range in bind_range_values:
            for help_hurt in help_hurt_values:
                start_time = time.time()
                stats = run_batches(num_batches=num_batches, num_steps=num_steps,
                                    bind_range=bind_range, seed=seed,
                                    lifespan=lifespan, help_hurt=help_hurt)
                elapsed_time = time.time() - start_time
                
                # Create a plot for the current run
                # plt.figure(figsize=(24, 20))
                plot_batch_statistics(stats)
                
                # Save the plot to a file with parameter info in the filename
                filename = f"stu_plot_lifespan{lifespan}_bind{bind_range}_helphurt{help_hurt}_num_steps{num_steps}_{num_batches}batches.png"
                plt.savefig(filename, dpi = 300)
                # plt.show()
                plt.close()
                print(f"Saved plot to {filename}")
                print(f"Batch completed in {elapsed_time:.2f} seconds")
                
                # Save summary statistics to a text file
                text_filename = f"plot_lifespan{lifespan}_bind{bind_range}_helphurt{help_hurt}_num_steps{num_steps}_{num_batches}batches.txt"
                with open(text_filename, "w") as f:
                    f.write(f"numsteps = {num_steps}, lifespan={lifespan}, bind_range={bind_range}, "
                            f"mean_crossing_step={stats['mean_crossing_step']}, "
                            f"Harm/Help={help_hurt}, "
                            f"Num Final Nodes={stats['num_nodes_remaining']}, "
                            f"Mean SCC Size={stats['mean_scc_size']}\n")
                print(f"Saved stats to {text_filename}")

               



if __name__ == '__main__':
    main()
