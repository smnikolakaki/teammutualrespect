# encoding: utf-8

from __future__ import print_function
from __future__ import division
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter

import networkx as nx
import csv
import json
import sys
import operator
import time
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import numpy as np


def clean_graphs(category_graph,k_core,weighted):
	category_graph_clean = {}
#	# uncomment to read graphs from files 
	for category, graph in category_graph.iteritems():
		ebunch = graph.selfloop_edges()
		graph.remove_edges_from(ebunch)
		core_graph = nx.k_core(graph,k=k_core)
		if weighted == False:
			for u,v in core_graph.edges(data=False):
				core_graph[u][v]['weight'] = 1

		category_graph_clean[category] = core_graph

	return category_graph_clean

def compute_indegree(graph,weighted):
	'''
	Implementing indegree computation because it does not work in networkx for weighted/unweighted instances
	'''
	author_indegree = {}
	if weighted == False:
		for u,v in graph.edges(data=False):
			if v not in author_indegree:
				author_indegree[v] = 1
			else:
				author_indegree[v] += 1

		for u in graph.nodes():
			if u not in author_indegree:
				author_indegree[u] = 0  

	elif weighted == True:
		for u,v in graph.edges(data=False):
			weight = graph[u][v]['weight']
			if v not in author_indegree:
				author_indegree[v] = weight
			else:
				author_indegree[v] += weight


	author_indegree_list = []
	for author, indegree in author_indegree.iteritems():
		author_indegree_list.append((author, indegree))

	return author_indegree_list

def sort_category_list(tuple_list):
	tuple_sorted_list = sorted(tuple_list, key = lambda x: x[1], reverse = True)
	return tuple_sorted_list

def get_graph_statistics(DG,category,kcore):

	print('Graph statistics for category:',category)

	# number of connected components
	num_connected_components = nx.number_weakly_connected_components(DG)
	print('Number of weakly connected components:',num_connected_components)
	print()
	try:
		category_name_list = category.split(' ')
	except:
		category_name_list = str(category)

	# in-degree distribution
	print('In-degree distribution.')
	try:
		filename = "-".join(category_name_list)+'-in-degree-hist-'+str(kcore)+'.eps'
	except:
		filename = category_name_list+'-in-degree-hist-'+str(kcore)+'.eps'
	indegree_list = DG.in_degree().values()
	if indegree_list:
		print('Max value of in-degree',max(indegree_list))
		print()
		# print('Indegree list:',indegree_list)
		histogram(indegree_list,'plots/category_in_degree/'+filename,'in-degree','# nodes','Distribution of in-degree per category',100,None,1,500,True)

	# # edge weight distribution
	# print('Edge weight distribution.')
	# try:
	# 	filename = "-".join(category_name_list)+'-edge-weight-hist-'+str(kcore)+'.eps'
	# except:
	# 	filename = category_name_list+'-edge-weight-hist-'+str(kcore)+'.eps'

	# weight_list = [e[2] for e in DG.edges_iter(data='weight', default=1)]
	# if weight_list:
	# 	print('Max value of weight edges',max(weight_list))
	# 	print()
	# 	# print('Weight list:',weight_list)
	# 	histogram(weight_list,'plots/category_edge_weight/'+filename,'edge weight','# edges','Distribution of edge weights per category',60,None,1,500,True)

def get_authors_activity_per_graph(category_graph,kcore):
	authors_list = []
	authors_dict = {}

	for category, graph in category_graph.iteritems():
		nodes = nx.nodes(graph)
		authors_list += nodes

	authors_set = list(set(authors_list))

	for author in authors_set:
		for category, graph in category_graph.iteritems():
			if author in graph:
				if author not in authors_dict:
					authors_dict[author] = 1
				else:
					authors_dict[author] += 1

	# in how many categories are the authors active
	print('Active in multiple categories authors.')
	filename = 'synthetic-active-categories-hist-'+str(kcore)+'.eps'
	active_categories_list = authors_dict.values()
	if active_categories_list:
		print('Max value of active authors',max(active_categories_list))
		histogram(active_categories_list,'plots/active_users/'+filename,'# categories','# authors','Distribution of number of authors active in categories',60,None,1,500,True)

def histogram(data,filename,xlabel,ylabel,title,bins,yscale,min_val,max_val,log):
	width = 0.8
	# in-degree larger than 1
	data = [x for x in data if x >= min_val]
	# in-degree smaller than 1
	data = [x for x in data if x < max_val]
	data = sorted(data)

	min_value = min(data); max_value = max(data)
	num_of_bins = np.arange(bins+1)-0.5
	# You typically want your plot to be ~1.33x wider than tall.  
	# Common sizes: (10, 7.5) and (12, 9)  
	plt.figure(figsize=(20, 10))  

	# An "interface" to matplotlib.axes.Axes.hist() method
	n, bins, patches = plt.hist(x=data, bins=num_of_bins, width = width, color='#3F5D7D',
	                            normed=False, alpha=0.5, histtype='bar', ec='black')
	plt.grid(axis='y', alpha=0.75)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	# plt.text(23, 45, r'$\mu=15, b=3$')
	if yscale == "log":
		plt.yscale('log')

	maxfreq = n.max()
	plt.xlim([min(data) - 1, max(data) + 1])
	if log == True:
		plt.yscale("log")

	print('Saving at:',filename)
	plt.savefig(filename, 
	           bbox_inches='tight', 
	           transparent=True,
	           pad_inches=0, 
	           format='eps', 
	           dpi=1000)

	plt.close()

def histogram_nodes_edges(data,filename,xlabel,ylabel,title):
	# An "interface" to matplotlib.axes.Axes.hist() method
	n, bins, patches = plt.hist(x=data, bins=20, color='#0504aa',
	                            alpha=0.7, rwidth=0.85)

	plt.grid(axis='y', alpha=0.75)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	maxfreq = n.max()
	# Set a clean upper y-axis limit.
	plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
	plt.savefig(filename, 
	           bbox_inches='tight', 
	           transparent=True,
	           pad_inches=0, 
	           format='eps', 
	           dpi=1000)
	plt.close()

def get_all_statistics(category_graph):
	kcore = 1
	num_nodes = []; num_edges = []
	for category, DG in category_graph.iteritems():
		num_of_nodes = DG.number_of_nodes(); num_of_edges = DG.number_of_edges()
		num_nodes.append(num_of_nodes); num_edges.append(num_of_edges)
		get_graph_statistics(DG,category,kcore)

	get_authors_activity_per_graph(category_graph,kcore)

	if num_nodes:
		histogram_nodes_edges(num_nodes,'plots/graph_nodes/synthetic-num-nodes-hist-'+str(kcore)+'.eps','# graphs','# nodes','Node distribution in dataset')
	if num_edges:
		# histogram of # of edges in graph datasets
		histogram_nodes_edges(num_edges,'plots/graph_edges/synthetic-num-edges-hist-'+str(kcore)+'.eps','# graphs','# edges','Edge distribution in dataset')


def compute_indegree_outdegree(main_category, category_graph, weighted):
	'''
	Implementing indegree computation because it does not work in networkx for weighted/unweighted instances
	'''
	author_score = {}

	main_graph = category_graph[main_category]


	# outdegree
	for category, graph in category_graph.iteritems():

		if category != main_category:
			for edge in graph.edges():
				from_node = edge[0]; to_node = edge[1]
				if from_node not in author_score:
					author_score[from_node] = 1
				else:
					author_score[from_node] += 1

	# avg outdegree
	for node, outdegree in author_score.iteritems():
		author_score[node] = outdegree/(len(category_graph)-1)

	# indegree
	for edge in main_graph.edges():
		from_node = edge[0]; to_node = edge[1]
		if to_node not in author_score:
			author_score[to_node] = 1
		else:
			author_score[to_node] += 1


	# print('Author node outdegree + indegree:',author_score[0])
	author_score_list = []
	for author, score in author_score.iteritems():
		author_score_list.append((author, score))


	return author_score_list

def graph_ranking_heuristic(category_graph,weighted):
	category_graph_rankings_dict = {}
	set_of_nodes = []

	for category, graph in category_graph.iteritems():
		graph_nodes = list(graph)
		set_of_nodes += graph_nodes

	set_of_nodes = list(set(set_of_nodes))

	for category, graph in category_graph.iteritems():
		graph_copy = nx.Graph(graph)
		graph_copy.add_nodes_from(set_of_nodes)
		author_indegree_cat_list = compute_indegree(graph_copy,weighted)
		# author_indegree_outdegree_cat_list = compute_indegree_outdegree(category, category_graph, weighted)
		author_indegree_cat_sorted_list = sort_category_list(author_indegree_cat_list)
		print('In degree for category:',category,'sorted list',author_indegree_cat_sorted_list[:5])
		category_graph_rankings_dict[category] = author_indegree_cat_sorted_list


	skill_author_index_dict = create_skill_author_index(category_graph_rankings_dict)
	# solution_dict = half_approximation_per_rank_algorithm(category_graph_rankings_dict, weighted, skill_author_index_dict)
	max_score = -float("inf")
	counter = 0
	while counter < 100:
		print('In graph ranking:',counter)
		solution_dict = half_approximation_algorithm(category_graph_rankings_dict, weighted)
		score, edges = compute_graph_score(solution_dict,category_graph)
		if score > max_score:
			max_score = score
			best_solution = solution_dict
			print('New best score in graph ranking:',max_score)
		counter+=1

	# print('Solution dictionary for graph ranking heuristic is:',solution_dict)
	solution_score, solution_edges = compute_graph_score(best_solution,category_graph)
	print('Solution score for graph ranking heuristic is:',solution_score)
	return solution_dict,solution_score,solution_edges

def graph_random(category_graph,weighted):
	category_graph_rankings_dict = {}
	set_of_nodes = []
	max_score = -float("inf")
	counter = 0

	while counter < 200:
		print('In graph random choice:',counter)
		solution_dict = {}
		category_graphs_copy = dict(category_graph)

		while category_graphs_copy:
			# randomly select a skill 
			category = random.choice(category_graphs_copy.keys())
			graph = category_graphs_copy[category]
			random_node = random.choice(graph.nodes())
			solution_dict[category] = (random_node,0)
			solution_score, solution_edges = compute_graph_score(solution_dict,category_graph)

			if solution_score > max_score:
				best_solution_dict = solution_dict
				max_score = solution_score
				print('New best score in graph random choice:',max_score)

			del category_graphs_copy[category]
		counter += 1

	best_solution_score, best_solution_edges = compute_graph_score(best_solution_dict,category_graph)

	print('Solution score for graph random choice heuristic is:',best_solution_score)
	return best_solution_dict,best_solution_score,best_solution_edges

def half_approximation_random_algorithm(category_rankings, weighted):
 
    counter = 0
    best_score = -float("inf")
 
    while counter < 50:
        solution_dict = {}
        selected_authors = {}
        category_rankings_copy = dict(category_rankings)
        # randomly select a skill 
        category = random.choice(category_rankings_copy.keys())
        # print('Random Category is:',category)
        category_ranking = category_rankings_copy[category]
 
        # assign the author with the most citations as a solution for this category
        top_author_score = category_ranking[0]
        solution_dict[category] = (top_author_score[0],0)
        top_author = top_author_score[0]; top_score= top_author_score[1]
        # print('Category:',category,'Top author is:',top_author,'rank is:',0)
        # stores authors seen so far
        selected_authors[top_author] = 1
 
        # delete category from ranking since solution for it has been assigned
        del category_rankings_copy[category]
 
        while category_rankings_copy:
            highest_rank = float("inf");
            category = random.choice(category_rankings_copy.keys())
            category_ranking = category_rankings_copy[category]
            for ranking, author_score in enumerate(category_ranking):
                author = author_score[0]
                if weighted == False and author not in selected_authors:
                    if ranking < highest_rank:
                        top_author_score = author_score; highest_rank = ranking; solution_category = category
                        top_ranking = ranking
                        break
            top_author = top_author_score[0]; top_score= top_author_score[1]
            solution_dict[solution_category] = (top_author,highest_rank)
 
            # stores authors seen so far
            if top_author in selected_authors:
                print('No more available authors')
                sys.exit(0)
            else:
                selected_authors[top_author] = 1
 
            # delete category from ranking since solution for it has been assigned
            del category_rankings_copy[solution_category]
         
        if top_score > best_score:
            best_score = top_score
            best_solution = solution_dict
 
        counter += 1
 
    return best_solution

def half_approximation_algorithm(category_rankings, weighted):
	solution_dict = {}
	selected_authors = {}
	category_rankings_copy = dict(category_rankings)
	# randomly select a skill 
	category = random.choice(category_rankings_copy.keys())
	# print('Random Category is:',category)
	category_ranking = category_rankings_copy[category]

	# assign the author with the most citations as a solution for this category
	top_author_score = category_ranking[0]
	solution_dict[category] = (top_author_score[0],0)
	top_author = top_author_score[0]; top_score= top_author_score[1]
	# print('Category:',category,'Top author is:',top_author,'rank is:',0)
	# stores authors seen so far
	selected_authors[top_author] = 1

	# delete category from ranking since solution for it has been assigned
	del category_rankings_copy[category]

	while category_rankings_copy:
		highest_rank = float("inf"); highest_score = -float("inf")

		for category, category_ranking in category_rankings_copy.iteritems():
			highest_rank_local = float("inf"); highest_score_local = -float("inf")
			# print('category is:',category)
			for ranking, author_score in enumerate(category_ranking):
				author = author_score[0]
				if weighted == False and author not in selected_authors:
					if ranking < highest_rank_local:
						solution_local = author_score; highest_rank_local = ranking; category_local = category
						solution_ranking = ranking
						break
				elif weighted == True and author not in selected_authors:
					score = author_score[1]
					if score > highest_score_local:
						solution_local = author_score; highest_score_local = score; category_local = category
						solution_ranking = ranking
						break

			if weighted == False:
				if highest_rank_local < highest_rank:
					top_author_score = solution_local; highest_rank = highest_rank_local; solution_category = category_local
					top_ranking = solution_ranking



			elif weighted == True:
				score = author_score[1]
				if highest_score_local > highest_score:
					top_author_score = solution_local; highest_score = highest_score_local; solution_category = category_local
					top_ranking = solution_ranking

		# print('In half approximation: Next category with highest ranking is:',solution_category,'because of (author,score)',top_author_score,'that has ranking:',highest_rank)
		top_author = top_author_score[0]; top_score= top_author_score[1]
		solution_dict[solution_category] = (top_author,highest_rank)
		# print('In half approximation:Assigned new solution.')
		# print('In half approximation: Solution dict became:',solution_dict)
		# print('Rank heuristic - New solution dictionary is:',solution_dict,'best assignment:',top_author_score)

		# stores authors seen so far
		if top_author in selected_authors:
			print('No more available authors')
			sys.exit(0)
		else:
			selected_authors[top_author] = 1

		# delete category from ranking since solution for it has been assigned
		del category_rankings_copy[solution_category]
	
	return solution_dict

def half_approximation_per_rank_algorithm(category_rankings, weighted, skill_author_index_dict):

	counter = 0
	high_score = -float("inf")

	while counter < 1:

		solution_dict = {}
		selected_authors = {}
		count_node_observations = {}
		author_category_encounter = {}
		category_selection = []
		category_rankings_copy = dict(category_rankings)
		num_of_nodes = len(category_rankings_copy.items()[0][1])
		best_score = 0

		for i in range(0,50):
			# print('I is:',i)
			solution_dict = {}
			# selected_authors = {}
			category_rankings_copy = dict(category_rankings)
			# randomly select a skill 
			category = random.choice(category_rankings_copy.keys())
			# print('Random Category is:',category)
			category_ranking = category_rankings_copy[category]

			# assign the author with the most citations as a solution for this category
			top_author_score = category_ranking[0]
			
			for ranking, top_author_score in enumerate(category_ranking):
				top_author = top_author_score[0]; top_score= top_author_score[1]
				if top_author not in selected_authors:
					solution_dict[category] = (top_author,ranking)

			# print('Category BIG IS:',category,'Top author is:',top_author,'rank is:',0+i)
			# stores authors seen so far

			selected_authors[top_author] = 1
			if top_author not in count_node_observations:
				count_node_observations[top_author] = 1
			else:
				count_node_observations[top_author] += 1

			# delete category from ranking since solution for it has been assigned
			del category_rankings_copy[category]

			while category_rankings_copy:
				highest_rank = float("inf"); 

				for category, category_ranking in category_rankings_copy.iteritems():
					highest_rank_local = float("inf");
					# print('category is:',category)
					for ranking, author_score in enumerate(category_ranking):
						author = author_score[0]

						if author not in count_node_observations and (author,category) not in author_category_encounter:
							count_node_observations[author] = 1
						elif (author,category) not in author_category_encounter:
							count_node_observations[author] += 1

						author_category_encounter[(author,category)] = 1

						if weighted == False and author not in selected_authors:
							if ranking < highest_rank_local:
								solution_local = author_score; highest_rank_local = ranking; category_local = category
								solution_ranking = ranking
								break

					if weighted == False:
						if highest_rank_local < highest_rank:
							top_author_score = solution_local; highest_rank = highest_rank_local; solution_category = category_local
							top_ranking = solution_ranking

				top_author = top_author_score[0]; top_score= top_author_score[1]

				solution_dict[solution_category] = (top_author,highest_rank)

				# stores authors seen so far
				if top_author in selected_authors:
					return best_solution
				else:
					selected_authors[top_author] = 1

				# delete category from ranking since solution for it has been assigned
				# print('Deleting category:',solution_category)
				del category_rankings_copy[solution_category]

			for node in selected_authors.keys():
				if count_node_observations[node] == 1:
					del selected_authors[node]

			# print('count node observations:',count_node_observations)
			local_score = compute_ranking_score(solution_dict,category_rankings,skill_author_index_dict)
			if local_score > best_score:
				best_solution = dict(solution_dict)
				best_score = local_score
				# print('Best solution for i:',i,'is:',best_solution,'with score:',best_score)

		counter += 1
		if best_score > high_score:
			high_score = best_score
			high_solution = best_solution

	return high_solution

def greedy_algorithm(category_rankings, weighted):
	solution_dict = {}
	selected_authors = {}
	outer_highest_score = -float("inf")
	category_rankings_copy = dict(category_rankings)

	for outer_category, outer_category_ranking in category_rankings.iteritems():
		for outer_ranking, outer_author_score in enumerate(outer_category_ranking):
			selected_authors = {}; solution_dict = {}
			outer_author = outer_author_score[0]; outer_score = outer_author_score[1];
			solution_dict[outer_category] = outer_author_score
			print('Outer author score:',outer_author_score)
			print('Outer category is:',outer_category)
			selected_authors[outer_author] = 1
			del category_rankings_copy[outer_category]

			while category_rankings_copy:
				highest_score = 0

				for inner_category, inner_category_ranking in category_rankings_copy.iteritems():
					for inner_ranking, inner_author_score in enumerate(inner_category_ranking):
						inner_author = inner_author_score[0]
						solution_dict[inner_category] = inner_author_score
						if weighted == False and inner_author not in selected_authors:
							score = compute_ranking_score(solution_dict,category_rankings)

							if score > highest_score:
								same_highest_score_solutions = []
							if score >= highest_score:
								same_highest_score_solutions.append(inner_author_score + (inner_category,))
								highest_score = score

						del solution_dict[inner_category]

				selected_author_score_category = random.choice(same_highest_score_solutions)
				selected_author = selected_author_score_category[0]; selected_score = selected_author_score_category[1]
				selected_category = selected_author_score_category[2]
				selected_authors[selected_author] = 1
				solution_dict[selected_category] = (selected_author,selected_score)
				del category_rankings_copy[selected_category]

			category_rankings_copy = dict(category_rankings)
			score = compute_ranking_score(solution_dict,category_rankings)

			if score > outer_highest_score:
				outer_highest_score = score
				best_solution_dict = solution_dict
				best_score = score
				# print('Best solution dict so far is:',best_solution_dict)
				print('Best score so far is:',best_score)

	return best_solution_dict

def greedy_algorithm_random_skills(category_rankings, weighted):
	solution_dict = {}
	selected_authors = {}
	outer_highest_score = -float("inf")
	category_rankings_copy = dict(category_rankings)

	for outer_category, outer_category_ranking in category_rankings.iteritems():
		for outer_ranking, outer_author_score in enumerate(outer_category_ranking):
			selected_authors = {}; solution_dict = {}
			outer_author = outer_author_score[0]; outer_score = outer_author_score[1];
			solution_dict[outer_category] = outer_author_score
			print('Outer author score:',outer_author_score)
			print('Outer category is:',outer_category)
			selected_authors[outer_author] = 1
			del category_rankings_copy[outer_category]

			while category_rankings_copy:
				highest_score = 0
				inner_category = random.choice(category_rankings_copy.keys())
				inner_category_ranking = category_rankings_copy[inner_category]
				for inner_ranking, inner_author_score in enumerate(inner_category_ranking):
					inner_author = inner_author_score[0]
					solution_dict[inner_category] = inner_author_score
					if weighted == False and inner_author not in selected_authors:
						score = compute_ranking_score(solution_dict,category_rankings)

						if score > highest_score:
							same_highest_score_solutions = []
						if score >= highest_score:
							same_highest_score_solutions.append(inner_author_score + (inner_category,))
							highest_score = score

					del solution_dict[inner_category]

				selected_author_score_category = random.choice(same_highest_score_solutions)
				selected_author = selected_author_score_category[0]; selected_score = selected_author_score_category[1]
				selected_category = selected_author_score_category[2]
				selected_authors[selected_author] = 1
				solution_dict[selected_category] = (selected_author,selected_score)
				del category_rankings_copy[selected_category]

			category_rankings_copy = dict(category_rankings)
			score = compute_ranking_score(solution_dict,category_rankings)
			if score > outer_highest_score:
				outer_highest_score = score
				best_solution_dict = solution_dict
				best_score = score
				# print('Best solution dict so far is:',best_solution_dict)
				# print('Best score so far is:',best_score)

	return best_solution_dict

def greedy_algorithm_random_skills_stop(category_rankings, weighted, skill_author_index_dict):
	solution_dict = {}
	selected_authors = {}
	outer_highest_score = 0
	category_rankings_copy = dict(category_rankings)
	computed_solutions = {}

	for outer_category, outer_category_ranking in category_rankings.iteritems():
		# dictionary stores authors seen so far
		seen_skills_author = {}
		seen_skills_author_inner = {}

		seen_author_skills = {}
		seen_author_skills_inner = {}

		for outer_ranking, outer_author_score in enumerate(outer_category_ranking):

			selected_authors = {}; solution_dict = {}
			# print('In greedy: Considering rank:',outer_ranking,'for category:',outer_category)
			outer_author = outer_author_score[0]; outer_score = outer_author_score[1];
			solution_dict[outer_category] = (outer_author,outer_ranking)
 
			prev_score = 0

			selected_authors[outer_author] = 1

			if outer_category not in seen_skills_author:
				seen_skills_author[outer_category] = {}
			seen_skills_author[outer_category][outer_author] = 1
			if outer_category not in seen_skills_author_inner:
				seen_skills_author_inner[outer_category] = {}
			seen_skills_author_inner[outer_category][outer_author] = 1
	
			if outer_author not in seen_author_skills:
				seen_author_skills[outer_author] = 0
			seen_author_skills[outer_author] += 1
			if outer_author not in seen_author_skills_inner:
				seen_author_skills_inner[outer_author] = 0
			seen_author_skills_inner[outer_author] += 1

			del category_rankings_copy[outer_category]

			for category,value in seen_skills_author_inner.iteritems():
				if category != outer_category:
					seen_skills_author_inner[category] = {}

			for category,value in seen_skills_author.iteritems():
				if category != outer_category:
					seen_skills_author[category] = {}

			seen_author_skills = dict(seen_author_skills_inner)

			while category_rankings_copy:
				highest_score = 0;
				# change random.choice with locally optimum starting point
				# inner_category = random.choice(category_rankings_copy.keys())

				highest_rank = float("inf");

				for category, category_ranking in category_rankings_copy.iteritems():
					highest_rank_local = float("inf");

					for ranking, author_score in enumerate(category_ranking):
						author = author_score[0]
						if weighted == False and author not in selected_authors:
							if ranking < highest_rank_local:
								solution_local = author_score; highest_rank_local = ranking; category_local = category
								break

					if weighted == False:
						if highest_rank_local < highest_rank:
							top_author_score = solution_local; highest_rank = highest_rank_local; solution_category = category_local

				# print('In greedy: Next category with highest ranking is:',solution_category,'because of (author,score)',top_author_score,'that has ranking:',highest_rank)

				top_author = top_author_score[0]; top_score= top_author_score[1]

				inner_category_ranking = category_rankings_copy[solution_category]
				inner_category = solution_category
				max_indegree = len(solution_dict)

				for inner_ranking, inner_author_score in enumerate(inner_category_ranking):
					inner_author = inner_author_score[0]
					if inner_author in selected_authors:
						max_indegree -= 1

					if inner_category not in seen_skills_author_inner:
						seen_skills_author_inner[inner_category] = {}
					seen_skills_author_inner[inner_category][inner_author] = 1


					if weighted == False and inner_author not in selected_authors:
						# score in degree: how many of the solution nodes have an edge to inner author + how many edges there in the solution before there were before 
						# max indegree: 
						if inner_author in seen_author_skills:
							score_in_degree = len(solution_dict) - seen_author_skills_inner[inner_author] + prev_score
						else:
							score_in_degree = len(solution_dict) + prev_score

						if inner_author not in seen_author_skills_inner:
							seen_author_skills_inner[inner_author] = 0
						seen_author_skills_inner[inner_author] += 1

						solution_dict[solution_category] = (inner_author,inner_ranking)

						score = max_indegree + score_in_degree

						del solution_dict[solution_category]
						if score > highest_score:
							author_score_category = inner_author_score + (inner_category,inner_ranking)
							highest_score = score

							if inner_category not in seen_skills_author:
								seen_skills_author[inner_category] = {}
							seen_skills_author[inner_category] = dict(seen_skills_author_inner[inner_category])
							seen_author_skills_best = dict(seen_author_skills_inner)
						
				seen_skills_author_inner[inner_category] = seen_skills_author[inner_category]

				# from these solutions with same score select highest in list
				selected_author_score_category = author_score_category
				selected_author = selected_author_score_category[0]; selected_score = selected_author_score_category[1]
				selected_category = selected_author_score_category[2]; selected_ranking = selected_author_score_category[3]
				selected_authors[selected_author] = 1
				# seen_author_skills_inner[selected_author] = seen_author_skills[selected_author]
				solution_dict[selected_category] = (selected_author,selected_ranking)
				# print('In greedy: Solution dict became:',solution_dict)
				# print('In greedy: Assigned new solution:',selected_author)
				seen_author_skills_inner = dict(seen_author_skills_best)
				del category_rankings_copy[selected_category]

				prev_score = highest_score

			seen_author_skills_inner = dict(seen_author_skills)
			category_rankings_copy = dict(category_rankings)
			# print('Score:',highest_score,'outer_highest_score',outer_highest_score)
			if highest_score > outer_highest_score:
				outer_highest_score = highest_score
				best_solution_dict = solution_dict
				best_score = highest_score
				# print('Best solution dict so far is:',best_solution_dict)
				print('Best score so far is:',best_score)

	return best_solution_dict

def max_score_algo(category_rankings, weighted):
	F_category_author = {};
	F_author_category = {}; 
	D = {}
	solution_dict = {}; selected_authors = {}
	available_categories = {}
	num_of_skills = len(category_rankings.keys())
	num_of_workers = len(category_rankings.items()[0][1])
	solution_size = len(F_category_author)

	category_rankings_copy = dict(category_rankings)
	num_iterations = 0

	for category,category_ranking in category_rankings_copy.iteritems():
		available_categories[category] = 1

	while solution_size < num_of_skills:
		category = random.choice(available_categories.keys())

		category_ranking = category_rankings_copy[category]
		for ranking, author_score in enumerate(category_ranking):
			author = author_score[0]
			# print('Category:',category,'author:',author,'ranking:',ranking)
			if author in F_author_category:
				remove_category = F_author_category[author]
				# print('Removing category:',remove_category,'because of author:',author,'but also in category:',category)
				available_categories[remove_category] = 1
				D[author] = 1
				del F_category_author[remove_category]
				del F_author_category[author]

			elif (author not in F_author_category) and (author not in D):
				# print('Adding author:',author,'to category:',category)
				F_category_author[category] = (author,ranking)
				F_author_category[author] = category
				# print('Current solution is:',F_category_author)
				del available_categories[category]
				break

		solution_size = len(F_category_author)
		# print('solution size is:',solution_size,'length D:',len(D),'num workers:',num_of_workers)
		if len(D) >= num_of_workers-1:
			break

	# print('Solution is:',F_category_author)
	return F_category_author

def compute_seen_ranking_score_author(solution_dict,seen_author_skills,author,new_category,prev_score):
	score = prev_score
	authors_set = [x[0] for x in solution_dict.values()]
	ranking_of_new_author = seen_author_skills[new_category]
 
	for category,author_score in solution_dict.iteritems():
		author_from = author_score[0]
		if (author not in seen_author_skills[category]):
			score +=1
		# if (author != author_from) and (author_from not in ranking_of_new_author):
		# 	score += 1

	return score

def compute_seen_ranking_score(solution_dict,seen_author_skills):
	score = 0
	authors_set = [x[0] for x in solution_dict.values()]
	# print('Solution dict is:',solution_dict)
	# print('Seen author skill:',seen_author_skills)
	for category, author_score in solution_dict.iteritems():
		author_to = author_score[0];
		for author_from in authors_set:
			if (author_to != author_from) and (author_from not in seen_author_skills[category]):
				score += 1
	# print('Score is:',score)
	return score


def create_skill_author_index(category_rankings):
	skill_author_index_dict = {}
	for category, category_ranking in category_rankings.iteritems():
		for ranking, author_score in enumerate(category_ranking):
			author = author_score[0]
			tup = (category,author)
			if tup not in skill_author_index_dict:
				skill_author_index_dict[tup] = ranking

	return skill_author_index_dict

def compute_graph_score(solution_dict,category_graph):
	score = 0
	existing_edges = []
	authors_set = [x[0] for x in solution_dict.values()]

	for category, author_score in solution_dict.iteritems():
		graph = category_graph[category]
		author_to = author_score[0]
		for author_from in authors_set:
			if author_to != author_from and graph.has_edge(author_from,author_to):
				score += graph[author_from][author_to]['weight']
				existing_edges.append((author_from,author_to))

	return score, existing_edges

def compute_ranking_score(solution_dict,category_ranking,skill_author_index_dict):
	score = 0
	indegree = {}
	outdegree = {}
	# print('Skill author index:',skill_author_index_dict)
	authors_set = [x[0] for x in solution_dict.values()]
	for category, author_score in solution_dict.iteritems():
		author_to = author_score[0];
		idx_author_to = skill_author_index_dict[(category,author_to)]
		for author_from in authors_set:
			idx_author_from = skill_author_index_dict[(category,author_from)]
			if author_to != author_from and idx_author_to < idx_author_from:
				if author_from not in outdegree:
					outdegree[author_from] = 0
				outdegree[author_from] += 1

				if author_to not in indegree:
					indegree[author_to] = 0
				indegree[author_to] += 1

				score += 1
	return score

def create_graph_indegree_rankings(category_graph,weighted):
	category_graph_rankings_dict = {}

	for category, graph in category_graph.iteritems():
		author_indegree_cat_list = compute_indegree(graph,weighted)
		author_indegree_cat_sorted_list = sort_category_list(author_indegree_cat_list)
		category_graph_rankings_dict[category] = author_indegree_cat_sorted_list

	return category_graph_rankings_dict

def store_category_ranks(category_ranks,store,kcore):

	if store == True:
		for category, ranking in category_ranks.iteritems():
			try:
				category_name_list = category.split(' ')
				filename = "-".join(category_name_list)+'-weighted-total-rank-'+str(kcore)+'.csv'
			except:
				category_name_list = str(category)
				filename = category_name_list+'-weighted-total-rank-'+str(kcore)+'.csv'

			with open('category_rankings/'+filename,'wb') as out:
			    csv_out=csv.writer(out)
			    for row in ranking:
			        csv_out.writerow(row)

def compute_indegree_per_category(graphs,weighted):
	'''
	Input: Dictionary with key:category, value: graph of category
	Output: Dictionary with key:(category,node), value: node's indegree for the specific category
	'''
	indegree_per_cat_dict = {}

	for category,graph in graphs.iteritems():

		for edge in graph.edges():
			from_node = edge[0]; to_node = edge[1]
			# print('Category is:',category,'From node:',from_node,'to node:',to_node)
			key = (category,to_node)
			if key not in indegree_per_cat_dict:
				indegree_per_cat_dict[key] = 1
			else:
				indegree_per_cat_dict[key] += 1

	return indegree_per_cat_dict

def compute_outdegree(graphs,weighted):
	'''
	Input: Dictionary with key:category, value: graph of category
	Output: Dictionary with key: node, value: node's outdegree in all categories
	'''
	outdegree_dict = {}
	for category,graph in graphs.iteritems():

		for edge in graph.edges():
			from_node = edge[0]; to_node = edge[1]
			key = from_node
			if key not in outdegree_dict:
				outdegree_dict[key] = 1
			else:
				outdegree_dict[key] += 1

	return outdegree_dict

def initialize_indegree_outdegree_dictionaries(first_assignment,outdegree_main_dict,graphs):
	'''
	Input: First assignment (category,node), Dictionary with nodes' total outdegree, graph dictionary key:category value:graph
	Output: Indegree Solution dictionary key: node, value: edges from node to a solution node in their corresponding graph, 
			Outdegree solution dictionary key: (category, node) value: edges from solution nodes to a node in a specific category (not solution's category)
			Outdegree nodes dictionary key: node, value: outdegree to categories not in the solution
			Selected categories dictionary key:category, value: has been assigned with node
	'''
	indegree_solution_dict = {}
	outdegree_solution_dict = {}
	selected_category_dict = {}
	selected_node_dict = {}

	assignment_category = first_assignment[0]; assignment_node = first_assignment[1]

	outdegree_dict = dict(outdegree_main_dict)

	# Creating indegree solution dictionary where key: node, value: edges from node to a solution node in their corresponding graph
	assignment_graph = graphs[assignment_category]
	for edge in assignment_graph.edges():
		from_node = edge[0]; to_node = edge[1]
		if to_node == assignment_node:
			if from_node not in indegree_solution_dict:
				indegree_solution_dict[from_node] = 1
			else:
				indegree_solution_dict[from_node] += 1

	# print('In degree solution dictionary is:',indegree_solution_dict)

	# Creating outdegree solution dictionary where key: (category,node), value: edges from solution nodes to a node in a specific category (not solution's category)
	for category, graph in graphs.iteritems():
		for edge in graph.edges():
			from_node = edge[0]; to_node = edge[1]
			if from_node == assignment_node and category != assignment_category:
				key = (category,to_node)
				# print('Edge is:',edge,'and key is:',key)
				if key not in outdegree_solution_dict:
					outdegree_solution_dict[key] = 1
				else:
					outdegree_solution_dict[key] += 1

	# print('Outdegree solution dictionary is:',outdegree_solution_dict)

	# print('Category is:',assignment_category)
	# print('Out degree solution dictionary before is:',outdegree_dict)
	# Updating total outdegree to not include outdegree of nodes in a solution
	for node,outdegree in outdegree_dict.iteritems():
		if assignment_graph.has_node(node):
			assignment_graph_node_outdegree = assignment_graph.out_degree(node)
		else:
			assignment_graph_node_outdegree = 0

		# print('assignment graph node value:',assignment_graph_node_outdegree,'for node:',node)
		outdegree_dict[node] -= assignment_graph_node_outdegree

	# print('Out degree solution dictionary after is:',outdegree_dict)
	selected_category_dict[assignment_category] = 1
	selected_node_dict[assignment_node] = 1

	return indegree_solution_dict,outdegree_solution_dict,outdegree_dict,selected_category_dict,selected_node_dict


def find_greedy_solution(indegree_solution_dict, outdegree_solution_dict, outdegree_dict, graphs, solution_dict,selected_category_dict,selected_node_dict):
	'''
	Input: Indegree Solution dictionary key: node, value: edges from node to a solution node in their corresponding graph, 
			Outdegree solution dictionary key: (category, node) value: edges from solution nodes to a node in a specific category (not solution's category)
			Outdegree nodes dictionary key: node, value: outdegree to categories not in the solution
	Output: Solution dictionary with final solution
	'''
	assigned_nodes = len(solution_dict)
	graphs_copy = dict(graphs)

	while assigned_nodes < len(graphs):
		max_score = -float('inf')
		# print('New assignment.')

		# Rescaling outdegree to other categories to be rescaled to the range of max (indegree solution + outdegree solution)
		outdegree_to_others_values_list = []
		for category, graph in graphs.iteritems():
			if category not in selected_category_dict:
				# For each available category, node consideration
				for node in graph.nodes():
					indegree_solution_score = 0; outdegree_solution_score = 0
					outdegree_score = 0
					if node not in selected_node_dict:
						outdegree_in_graph_score = graph.out_degree(node)
						if node in indegree_solution_dict:
							indegree_solution_score = indegree_solution_dict[node]
						if (category,node) in outdegree_solution_dict:
							outdegree_solution_score = outdegree_solution_dict[(category,node)]
						if node in outdegree_dict:
							outdegree_score = outdegree_dict[node]

						outdegree_to_others_value = (outdegree_score - outdegree_in_graph_score)/(len(graphs) - len(solution_dict))
						outdegree_to_others_values_list.append(outdegree_to_others_value)

		oldmax = max(outdegree_to_others_values_list); oldmin = min(outdegree_to_others_values_list)
		oldrange = oldmax - oldmin

		newmax = 2*len(solution_dict); newmin = 0
		newrange = newmax - newmin
		# print('new max',newmax,'old max:',oldmax,'old min:',oldmin)

		# Compute next solution
		for category, graph in graphs.iteritems():
			if category not in selected_category_dict:

				# For each available category, node consideration
				for node in graph.nodes():
					indegree_solution_score = 0; outdegree_solution_score = 0
					outdegree_score = 0
					if node not in selected_node_dict:
						outdegree_in_graph_score = graph.out_degree(node)
						if node in indegree_solution_dict:
							indegree_solution_score = indegree_solution_dict[node]
						if (category,node) in outdegree_solution_dict:
							outdegree_solution_score = outdegree_solution_dict[(category,node)]
						if node in outdegree_dict:
							outdegree_score = outdegree_dict[node]

						oldvalue = (outdegree_score - outdegree_in_graph_score)/(len(graphs) - len(solution_dict))

						if oldmax-oldmin == 0:
							new_value = oldvalue
						else:
							newvalue = ((oldvalue - oldmin) / (oldmax - oldmin)) * (newmax - newmin) + newmin

						score = indegree_solution_score + outdegree_solution_score + newvalue # + (outdegree_score - outdegree_in_graph_score)/(len(graphs) - len(solution_dict))
						# print('Indegree solution score:',indegree_solution_score,'Outdegree solution score:',outdegree_solution_score,'Total avg outdegree score:',(outdegree_score - outdegree_in_graph_score)/(len(graphs) - len(solution_dict)))
						
						if score > max_score:
							max_score = score
							best_solution = (category,node); best_indegree_sol_score = indegree_solution_score
							best_outdegree_sol_score = outdegree_solution_score; best_outdegree_score = outdegree_score
							best_oudegree_in_graph_score = outdegree_in_graph_score
							# print('Best Solution is:',best_solution,'max score is:',max_score,'in degree solution score:',best_indegree_sol_score,'out degree solution score:',\
							# 		best_outdegree_sol_score,'out degree score:',best_outdegree_score,'out degree in graph score:',best_oudegree_in_graph_score)

		# Assign best solution to solution dictionary
		best_category = best_solution[0]; best_node = best_solution[1]
		best_assignment = (best_category,best_node)
		solution_dict[best_category] = (best_node,max_score)
		# print('Best Solution is:',best_assignment,'best_outdegree_sol_score',best_outdegree_sol_score,'best_indegree_sol_score',best_indegree_sol_score,'best_outdegree_score',best_outdegree_score)

		# print('Find greedy - New solution dictionary is:',solution_dict,'best assignment:',best_assignment)

		# Update dictionaries
		selected_category_dict[best_category] = 1
		selected_node_dict[best_node] = 1
		(indegree_solution_dict, outdegree_solution_dict, outdegree_dict) = update_solution_dictionaries(best_assignment,indegree_solution_dict, outdegree_solution_dict, outdegree_dict, graphs)
		assigned_nodes = len(solution_dict)
		# break

	return solution_dict

def find_random_solution(indegree_solution_dict, outdegree_solution_dict, outdegree_dict, graphs, solution_dict,selected_category_dict,selected_node_dict):
	'''
	Input: Indegree Solution dictionary key: node, value: edges from node to a solution node in their corresponding graph, 
			Outdegree solution dictionary key: (category, node) value: edges from solution nodes to a node in a specific category (not solution's category)
			Outdegree nodes dictionary key: node, value: outdegree to categories not in the solution
	Output: Solution dictionary with final solution
	'''
	assigned_nodes = len(solution_dict)
	graphs_copy = dict(graphs)
	while assigned_nodes < len(graphs):
		max_score = -float('inf')
		# print('New assignment.')
		category = random.choice(graphs_copy.keys())
		graph = graphs_copy[category]
		# Rescaling outdegree to other categories to be rescaled to the range of max (indegree solution + outdegree solution)
		outdegree_to_others_values_list = []

		for node in graph.nodes():
			indegree_solution_score = 0; outdegree_solution_score = 0
			outdegree_score = 0
			if node not in selected_node_dict:
				outdegree_in_graph_score = graph.out_degree(node)
				if node in indegree_solution_dict:
					indegree_solution_score = indegree_solution_dict[node]
				if (category,node) in outdegree_solution_dict:
					outdegree_solution_score = outdegree_solution_dict[(category,node)]
				if node in outdegree_dict:
					outdegree_score = outdegree_dict[node]

				outdegree_to_others_value = (outdegree_score - outdegree_in_graph_score)/(len(graphs) - len(solution_dict))
				outdegree_to_others_values_list.append(outdegree_to_others_value)

		oldmax = max(outdegree_to_others_values_list); oldmin = min(outdegree_to_others_values_list)
		oldrange = oldmax - oldmin

		newmax = 2*len(solution_dict); newmin = 0
		newrange = newmax - newmin
		# print('new max',newmax,'old max:',oldmax,'old min:',oldmin)

		# For each available category, node consideration
		for node in graph.nodes():
			indegree_solution_score = 0; outdegree_solution_score = 0
			outdegree_score = 0
			if node not in selected_node_dict:
				outdegree_in_graph_score = graph.out_degree(node)
				if node in indegree_solution_dict:
					indegree_solution_score = indegree_solution_dict[node]
				if (category,node) in outdegree_solution_dict:
					outdegree_solution_score = outdegree_solution_dict[(category,node)]
				if node in outdegree_dict:
					outdegree_score = outdegree_dict[node]

				oldvalue = (outdegree_score - outdegree_in_graph_score)/(len(graphs) - len(solution_dict))

				if oldmax-oldmin == 0:
					new_value = oldvalue
				else:
					newvalue = ((oldvalue - oldmin) / (oldmax - oldmin)) * (newmax - newmin) + newmin

				score = indegree_solution_score + outdegree_solution_score + newvalue # + (outdegree_score - outdegree_in_graph_score)/(len(graphs) - len(solution_dict))
				# print('Score is:',score,'indegree solution score:',indegree_solution_score,'outdegree solution score:',outdegree_solution_score,'new value:',newvalue)
				if score > max_score:
					max_score = score
					best_solution = (category,node); best_indegree_sol_score = indegree_solution_score
					best_outdegree_sol_score = outdegree_solution_score; best_outdegree_score = outdegree_score
					best_oudegree_in_graph_score = outdegree_in_graph_score
					# print('Best Solution is:',best_solution,'max score is:',max_score,'in degree solution score:',best_indegree_sol_score,'out degree solution score:',\
					# 		best_outdegree



		del graphs_copy[category]
		# Assign best solution to solution dictionary
		best_category = best_solution[0]; best_node = best_solution[1]
		best_assignment = (best_category,best_node)
		solution_dict[best_category] = (best_node,max_score)

		# print('Find random - New solution dictionary is:',solution_dict,'best assignment:',best_assignment)

		# Update dictionaries
		selected_category_dict[best_category] = 1
		selected_node_dict[best_node] = 1
		(indegree_solution_dict, outdegree_solution_dict, outdegree_dict) = update_solution_dictionaries(best_assignment,indegree_solution_dict, outdegree_solution_dict, outdegree_dict, graphs)
		assigned_nodes = len(solution_dict)
		# break

	return solution_dict

def update_solution_dictionaries(new_assignment,indegree_solution_dict, outdegree_solution_dict, outdegree_dict, graphs):
	'''
	Input: Indegree Solution dictionary key: node, value: edges from node to a solution node in their corresponding graph, 
			Outdegree solution dictionary key: (category, node) value: edges from solution nodes to a node in a specific category (not solution's category)
			Outdegree nodes dictionary key: node, value: outdegree to categories not in the solution
	Output: Updated Indegree Solution dictionary key: node, value: edges from node to a solution node in their corresponding graph, 
			Updated Outdegree solution dictionary key: (category, node) value: edges from solution nodes to a node in a specific category (not solution's category)
			Updated Outdegree nodes dictionary key: node, value: outdegree to categories not in the solution
	'''

	assignment_category = new_assignment[0]; assignment_node = new_assignment[1]

	# Creating indegree solution dictionary where key: node, value: edges from node to a solution node in their corresponding graph
	assignment_graph = graphs[assignment_category]
	for edge in assignment_graph.edges():
		from_node = edge[0]; to_node = edge[1]
		if to_node == assignment_node:
			if from_node not in indegree_solution_dict:
				indegree_solution_dict[from_node] = 1
			else:
				indegree_solution_dict[from_node] += 1

	# Creating outdegree solution dictionary where key: (category,node), value: edges from solution nodes to a node in a specific category (not solution's category)
	for category, graph in graphs.iteritems():
		for edge in graph.edges():
			from_node = edge[0]; to_node = edge[1]
			if from_node == assignment_node and category != assignment_category:
				key = (category,to_node)
				# print('Edge is:',edge,'and key is:',key)
				if key not in outdegree_solution_dict:
					outdegree_solution_dict[key] = 1
				else:
					outdegree_solution_dict[key] += 1

	# Updating total outdegree to not include outdegree of nodes in a solution
	for node,outdegree in outdegree_dict.iteritems():
		assignment_graph_node_outdegree = assignment_graph.out_degree(node)
		if assignment_graph.has_node(node):
			assignment_graph_node_outdegree = assignment_graph.out_degree(node)
		else:
			assignment_graph_node_outdegree = 0

		# print('update assignment graph node value:',assignment_graph_node_outdegree,'for node:',node)
		outdegree_dict[node] -= assignment_graph_node_outdegree

	return indegree_solution_dict,outdegree_solution_dict,outdegree_dict

def find_first_assignments(indegree_per_category_dict,outdegree_dict,graphs_copy):
	'''
	Input: Dictionary with indegree of each node for each category, Dictionary with outdegree, Dictionary with categories and corresponding graphs
	Output: Dictionary with key: category and value: the node assignment, First assignment pair (category,node), Dictionary with categories assigned so far
	'''
	ordered_first_assignment = []

	score_dict = {}
	solution_dict = {}

	max_score = -float("inf")
	# Score is the sum of a node's indegree in a specific category and 
	# outdegree of node in the rest of the categories

	# copute outdegree per category
	outdegree_per_category_dict = {}
	for category, graph in graphs_copy.iteritems():
		for edge in graph.edges():
			from_node = edge[0]; to_node = edge[1]
			key = (category,from_node)
			if key not in outdegree_per_category_dict:
				outdegree_per_category_dict[key] = 1
			else:
				outdegree_per_category_dict[key] += 1


	for indegree_key,indegree_value in indegree_per_category_dict.iteritems():
		indegree_category = indegree_key[0]; indegree_node = indegree_key[1]
		subtract_node_outdegree_of_category = 0; total_node_outdegree = 0
		# print('Indegree key:',indegree_key)
		# print('Indegree is:',indegree_value)
		if indegree_key in outdegree_per_category_dict:
			subtract_node_outdegree_of_category = outdegree_per_category_dict[indegree_key]
			# print('Outdegree for this category is:',subtract_node_outdegree_of_category)
		if indegree_node in outdegree_dict:
			total_node_outdegree = outdegree_dict[indegree_node]
	

		score = indegree_value + (total_node_outdegree - subtract_node_outdegree_of_category)/(len(graphs_copy)-1)

		if indegree_key not in score_dict:
			score_dict[indegree_key] = score
		else:
			print('Should never be here.')
			score_dict[indegree_key] += score


	for assignment, score in score_dict.iteritems():
		ordered_first_assignment.append(assignment + (score,))


	ordered_first_assignment = sorted(ordered_first_assignment, key = lambda x: x[2], reverse = True)
	# print('In find first assignments:',ordered_first_assignment[:10])
	return ordered_first_assignment

def random_first_assignment(indegree_per_category_dict,outdegree_dict,graphs_copy):

	'''
	Input: Dictionary with indegree of each node for each category, Dictionary with outdegree, Dictionary with categories and corresponding graphs
	Output: Dictionary with key: category and value: the node assignment, First assignment pair (category,node), Dictionary with categories assigned so far
	'''
	ordered_first_assignment = []

	score_dict = {}
	solution_dict = {}

	ordered_first_assignment_categories_dict = {}

	max_score = -float("inf")
	# Score is the sum of a node's indegree in a specific category and 
	# outdegree of node in the rest of the categories

	# copute outdegree per category
	outdegree_per_category_dict = {}
	for category, graph in graphs_copy.iteritems():
		for edge in graph.edges():
			from_node = edge[0]; to_node = edge[1]
			key = (category,from_node)
			if key not in outdegree_per_category_dict:
				outdegree_per_category_dict[key] = 1
			else:
				outdegree_per_category_dict[key] += 1


	for indegree_key,indegree_value in indegree_per_category_dict.iteritems():
		indegree_category = indegree_key[0]; indegree_node = indegree_key[1]
		subtract_node_outdegree_of_category = 0; total_node_outdegree = 0
		# print('Indegree key:',indegree_key)
		# print('Indegree is:',indegree_value)
		if indegree_key in outdegree_per_category_dict:
			subtract_node_outdegree_of_category = outdegree_per_category_dict[indegree_key]
			# print('Outdegree for this category is:',subtract_node_outdegree_of_category)
		if indegree_node in outdegree_dict:
			total_node_outdegree = outdegree_dict[indegree_node]
	

		score = indegree_value + (total_node_outdegree - subtract_node_outdegree_of_category)/(len(graphs_copy)-1)

		if indegree_key not in score_dict:
			score_dict[indegree_key] = score
		else:
			print('Should never be here.')
			score_dict[indegree_key] += score

	for assignment, score in score_dict.iteritems():
		ordered_first_assignment.append(assignment + (score,))


	ordered_first_assignment = sorted(ordered_first_assignment, key = lambda x: x[2], reverse = True)
	for item in ordered_first_assignment:
		cat = item[0]; node = item[1]; score = item[2]
		if cat not in ordered_first_assignment_categories_dict:
			ordered_first_assignment_categories_dict[cat] = []
		ordered_first_assignment_categories_dict[cat].append(item)

	# print('In find first assignments:',ordered_first_assignment[:10])
	return ordered_first_assignment_categories_dict

def graph_algorithm_random(graphs,weighted):
	counter = 0
	best_score = -float("inf")
	scores =[]

	graphs_copy = dict(graphs)
	# key is (category,node)
	# print('Computing in degree per category')
	indegree_per_category_dict = compute_indegree_per_category(graphs_copy,weighted)
	# print('In degree per category:',indegree_per_category_dict)
	# print('Computing in outdegree')
	outdegree_main_dict = compute_outdegree(graphs_copy,weighted)
	# print('Total Out degree of graph nodes:',outdegree_main_dict)
	# print('Finding first assignment')
	# ordered_first_assignment = find_first_assignments(indegree_per_category_dict,outdegree_main_dict,graphs_copy)
	ordered_first_assignment_categories_dict = random_first_assignment(indegree_per_category_dict,outdegree_main_dict,graphs_copy)
	while counter < 10:
		start = time.time()
		solution_dict = {}
		random_category = random.choice(graphs_copy.keys())
		random_assignment = ordered_first_assignment_categories_dict[random_category][0]
		best_category = random_assignment[0]; best_node = random_assignment[1]; max_score = random_assignment[2]
		solution_dict[best_category] = (best_node,max_score)
		first_assignment = (best_category,best_node)
		print('Iteration of graph random:',counter)
		(indegree_solution_dict, outdegree_solution_dict, outdegree_dict,selected_category_dict,selected_node_dict) = initialize_indegree_outdegree_dictionaries(first_assignment,outdegree_main_dict,graphs_copy)
		final_solution_dict = find_random_solution(indegree_solution_dict, outdegree_solution_dict, outdegree_dict, graphs_copy, solution_dict,selected_category_dict,selected_node_dict)
		# print('Solution dictionary for graph algorithm random is:',final_solution_dict)
		final_score, final_edges = compute_graph_score(final_solution_dict,graphs)
		# print('Solution score for graph algorithm random is:',final_score)
		scores.append(final_score)

		if final_score > best_score:
			best_score = final_score
			best_solution = final_solution_dict
			best_edges = final_edges
			print('Found best solution in algorithms random:',final_score)
		print('Time to run one loop of graph algorithm random:',time.time() - start)

		counter += 1

	score_mean = np.mean(scores)
	score_std = np.std(scores)
	print('Best Random solution is:',best_solution)
	return best_solution,best_score,best_edges,score_mean,score_std

def graph_algorithm_greedy(graphs,weighted):
	counter = 0
	best_score = -float("inf")
	scores =[]

	graphs_copy = dict(graphs)
	# key is (category,node)
	indegree_per_category_dict = compute_indegree_per_category(graphs_copy,weighted)
	# print('In degree per category:',indegree_per_category_dict)
	outdegree_main_dict = compute_outdegree(graphs_copy,weighted)
	# print('Total Out degree of graph nodes:',outdegree_main_dict)
	ordered_first_assignment = find_first_assignments(indegree_per_category_dict,outdegree_main_dict,graphs_copy)
	# sprint('Ordered first assignments:',ordered_first_assignment)
	# ordered_first_assignment_categories_dict = random_first_assignment(indegree_per_category_dict,outdegree_main_dict,graphs_copy)
	while counter < 1:
		start = time.time()
		solution_dict = {}
		random_assignment = ordered_first_assignment[0]
		best_category = random_assignment[0]; best_node = random_assignment[1]; max_score = random_assignment[2]
		# print('Greedy starting with:',best_category,best_node)
		solution_dict[best_category] = (best_node,max_score)
		first_assignment = (best_category,best_node)

		(indegree_solution_dict, outdegree_solution_dict, outdegree_dict,selected_category_dict,selected_node_dict) = initialize_indegree_outdegree_dictionaries(first_assignment,outdegree_main_dict,graphs_copy)
		final_solution_dict = find_greedy_solution(indegree_solution_dict, outdegree_solution_dict, outdegree_dict, graphs_copy, solution_dict,selected_category_dict,selected_node_dict)
		# print('Solution dictionary for graph algorithm greedy is:',final_solution_dict)
		final_score, final_edges = compute_graph_score(final_solution_dict,graphs)
		# print('Solution score for graph algorithm greedy is:',final_score)
		scores.append(final_score)

		if final_score > best_score:
			best_score = final_score
			best_solution = final_solution_dict
			best_edges = final_edges
		print('Time to run one loop of graph algorithm greedy:',time.time()-start)
		counter += 1

	score_mean = np.mean(scores)
	score_std = np.std(scores)
	print('Best Greedy solution is:',best_solution)
	return best_solution,best_score,best_edges,score_mean,score_std

