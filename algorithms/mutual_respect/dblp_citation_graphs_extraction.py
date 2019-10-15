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
import mutual_respect_algorithms as mra
import pandas as pd


# Other options for categories
# customized_categories_list = [
# {'artificial intelligence':1,'natural language processing':1,'robotics':1, 'neural networks':1},
# {'web and information systems':1, 'human-computer interaction':1, 'artificial intelligence':1, 'computer graphics':1 },
# {'cryptography':1, 'security and privacy':1, 'algorithms and theory':1, 'information systems':1, 'information retrieval': 1},
# {'bioinformatics':1,'computational biology':1, 'evolutionary computation': 1,'high-performance computing':1,'artificial intelligence':1},
# {'formal methods':1,'programming languages':1, 'algorithms and theory': 1,'algorithms':1,'computational complexity':1,'geometric algorithms':1},
# {'software engineering':1,'operating systems':1, 'high-performance computing': 1,'distributed and parallel computing':1},
# {'signal processing':1, 'information systems':1, 'wireless networks and mobile computing':1, 'computer networking':1, 'information retrieval': 1},
# {'data mining':1, 'databases':1, 'algorithms and theory':1, 'algorithms': 1,'artificial intelligence':1},
# {'high-performance computing':1, 'computer architecture':1, 'distributed and parallel computing':1, 'computer hardware':1, 'logic':1},
# {'artificial intelligence':1,'natural language processing':1,'robotics':1, 'neural networks':1, 'data mining':1, 'databases':1, 'algorithms and theory':1, 'algorithms': 1},
# {'cryptography':1, 'security and privacy':1, 'algorithms and theory':1, 'information systems':1, 'information retrieval': 1,'high-performance computing':1, 'computer architecture':1, 'distributed and parallel computing':1, 'computer hardware':1, 'logic':1},
# {'web and information systems':1, 'human-computer interaction':1, 'artificial intelligence':1, 'computer graphics':1,'artificial intelligence':1,'natural language processing':1,'robotics':1, 'neural networks':1, 'control systems':1},
# {'web and information systems':1, 'human-computer interaction':1, 'artificial intelligence':1, 'computer graphics':1, 'data mining':1, 'data management':1, 'databases':1, 'algorithms and theory':1, 'algorithms': 1},
# {'software engineering':1,'operating systems':1, 'high-performance computing': 1,'distributed and parallel computing':1,'high-performance computing':1, 'computer architecture':1, 'distributed and parallel computing':1, 'computer hardware':1, 'logic':1}
# ]

customized_categories_list = [
{'artificial intelligence':1,'natural language processing':1,'robotics':1, 'neural networks':1},
{'data mining':1, 'databases':1, 'algorithms and theory':1, 'algorithms': 1},
{'data mining':1, 'databases':1, 'algorithms and theory':1, 'algorithms': 1,'artificial intelligence':1,'natural language processing':1,'robotics':1, 'neural networks':1},
{'software engineering':1,'operating systems':1, 'high-performance computing': 1,'distributed and parallel computing':1},
{'signal processing':1,'wireless networks and mobile computing':1, 'computer networking':1, 'information retrieval': 1},
{'signal processing':1,'wireless networks and mobile computing':1, 'computer networking':1, 'information retrieval': 1,'software engineering':1,'operating systems':1, 'high-performance computing': 1,'distributed and parallel computing':1},
{'software engineering':1,'operating systems':1, 'high-performance computing': 1},
{'operating systems':1, 'high-performance computing': 1},
]




def transform_raw_file_to_list(file_path,file_names):

	'''
	Input: file path, file name
	Output: Returns a list with the json format of the papers in the available dblp dataset files 
	'''
	print('Dataset path file name is:',file_path)

	# store all papers of dblp dataset files
	papers_json_format = []

	for j,file_name in enumerate(file_names):

		file_path_name = file_path + '/' + file_name
		print('Loading file ',j,':',file_name)

		with open(file_path_name) as f:
			file_content = f.readlines()

		# number of papers for dblp-ref-3.json: 79007, number of papers for other dblp-ref-{0,1,2}.json: 1000000
		print('Number of json objects:',len(file_content))

		for i,json_object in enumerate(file_content):
			# print to show which paper is being read from the file
			if i%10000 == 0 and i>0:
				print('# Json object:',i,' ',len(file_content)-i,'more objects to read')

			papers_json_format.append(json.loads(json_object))

		print('Loading completed.')

	return papers_json_format

def print_statistics(papers_json_format,venue_dict,venue_counter_dict,num_useful_papers,no_references_counter,no_venue_counter,no_authors_counter,no_paper_id_counter,\
					duplicates_counter,no_references_paper_ids_list,no_venue_paper_ids_list):
	
	'''
	Input: Dictionaries and counters created during the extraction of useful entries from the dblp datasets
	Output: None
	Purpose of this function is to prin some statistics about the dblp dataset paper id, venue, and other values
	'''

	num_overlapping_no_citations_venues = len(list(set(no_references_paper_ids_list) & set(no_venue_paper_ids_list)))
	print()
	print('Some statistics')
	print('-'*40)
	print('Number of papers with useful information:',num_useful_papers)
	print('% of useful papers in the dataset:',((num_useful_papers)/len(papers_json_format))*100,'%')
	print('Number of papers with no citations:',no_references_counter)
	print('Number of papers with no venue:',no_venue_counter)
	print('Number of papers with no authors:',no_authors_counter)
	print('Number of papers with no paper_id:',no_paper_id_counter)
	print('Duplicates in dataset',duplicates_counter)
	print('% of papers with no citations and no references:',((no_references_counter+no_paper_id_counter-num_overlapping_no_citations_venues)/len(papers_json_format))*100,"% of the dataset.")
	print('Overlapping paper objects with no references and no venue:',num_overlapping_no_citations_venues,'length of no citations:',len(no_references_paper_ids_list),'length of no venue:',len(no_venue_paper_ids_list))
	print('Number of found venues',len(venue_dict))
	print('% of venues in the papers with useful information:',((len(venue_dict))/num_useful_papers)*100,'%')
	venue_counter_sorted_dict = sorted(venue_counter_dict.items(), key=operator.itemgetter(1),reverse=True)
	print('Top 200 Venues from most popular to least popular:',venue_counter_sorted_dict[:200])
	print('-'*40,'\n')

def create_dictionaries_from_dataset_list(papers_json_format):
	'''
	Input: List with the json format of the papers in the avilable dblp dataset files
	Output: 1. A multi-index dictionary - key1: conference venue, key2: author that published in this venue, value: list of citations (paper ids) the author has given to papers of the same venue
			2. A dictionary - key: paper id, value: list of paper authors
			3. A dictionary - key: conference venue, value: number of papers published in this venue
			4. A dictionary - key: paper id, value: venue where it was published
	'''

	venue_dict = {}; venue_counter_dict = {}; paper_id_authors_dict = {}; paper_id_venue_dict = {}

	print('Number of papers in dblp files:',len(papers_json_format))

	# number of papers with no references
	no_references_counter = 0; no_references_paper_ids_list = []
	# number of papers with no venue value
	no_venue_counter = 0; no_venue_paper_ids_list = []
	# number of papers with no authors value
	no_authors_counter = 0
	# number of papers with no paper id value
	no_paper_id_counter = 0
	# duplications in the dataset
	duplicates_counter = 0
	# number of actually used papers
	num_useful_papers = 0

	print('Number of papers to read:',len(papers_json_format))

	for i,json_object in enumerate(papers_json_format):

		flag = False
		if i%10000 == 0 and i>0:
			print('# Json object:',i,' ',len(papers_json_format)-i,'more objects to read')

		# print('Json object keys:',json_object.keys())

		# increase counter of corresponding attribute if it is missing
		try:
			references = json_object['references']
		except:
			no_references_counter+=1; no_references_paper_ids_list.append(json_object['id'])
			references = []

		try:
			venue = json_object['venue']; 
			if venue == '':
				no_venue_counter += 1; no_venue_paper_ids_list.append(json_object['id'])
				flag = True
		except:
			no_venue_counter += 1; no_venue_paper_ids_list.append(json_object['id'])
			venue = ''
			flag = True

		try:
			authors = json_object['authors']
		except:
			no_authors_counter += 1
			authors = []
			flag = True

		try:
			paper_id = json_object['id']
		except:
			no_paper_id_counter += 1
			paper_id = ''
			flag = True

		if flag == True:
			continue


		# creating dictionaries only if the entry is complete
		num_useful_papers += 1
		# creating venue dictionary  
		if venue not in venue_dict:
			venue_dict[venue] = {}
			venue_counter_dict[venue] = 1
		else:
			venue_counter_dict[venue] += 1

		for author in authors:
			if author not in venue_dict[venue]:
				venue_dict[venue][author] = references
			else:
				venue_dict[venue][author] += references

		# creating paper id dictionary with authors
		if paper_id not in paper_id_authors_dict:
			paper_id_authors_dict[paper_id] = authors
		else:
			duplicates_counter += 1

		# creating paper id dictionary with venues
		if paper_id not in paper_id_venue_dict:
			paper_id_venue_dict[paper_id] = venue.replace(",","").lower()
		else:
			duplicates_counter += 1

	# # printing some statistics about created dictionaries
	# print_statistics(papers_json_format,venue_dict,venue_counter_dict,num_useful_papers,no_references_counter,no_venue_counter,no_authors_counter,no_paper_id_counter,\
	# 				duplicates_counter,no_references_paper_ids_list,no_venue_paper_ids_list)

	# format name of venues in the dictionaries: remove commas and transform to lower case 
	venue_dict = {i.replace(",","").lower():val for i,val in venue_dict.iteritems()}
	venue_counter_dict = {i.replace(",","").lower():val for i,val in venue_counter_dict.iteritems()}

	return venue_dict,venue_counter_dict,paper_id_authors_dict,paper_id_venue_dict

def store_list_to_file(file_name,name_list):

	name_list_sorted = sorted(name_list)
	print('Writing list to txt file:',file_name)
	with open(file_name, 'w') as f:
		for s in name_list_sorted:
			f.write((s.lower() + u'\n').encode('unicode-escape'))

	print('Writing completed.')


def store_dict_to_file(file_name,name_dict):

	with open(file_name, 'w') as file:
		file.write(json.dumps(name_dict)) # use `json.loads` to do the reverse

def store_dict_to_pkl_file(file_name,name_dict):
	# store hierarchical dictionary
	print('Store hierarchical dictionary to file.')
	with open(file_name, 'wb') as f:
		pickle.dump(name_dict, f, pickle.HIGHEST_PROTOCOL)
	print('Finished storing hierarchical dictionary to file.')

def read_dict_to_pkl_file(file_name):
	# load hierarchical dictionary
	with open(file_name, 'rb') as f:
		name_dict = pickle.load(f)

	return name_dict

def read_venue_assignments_csv(file_name):
	venue_cat_dict = {}
	with open(file_name, "rb") as f:
		reader = csv.reader(f, delimiter="\t")
		for i, line in enumerate(reader):
			csv_line_split = line[0].strip().split(',')	
			venue_name = csv_line_split[0].strip(); cat_name = csv_line_split[2].strip();
			venue_cat_dict[venue_name] = cat_name 

	return venue_cat_dict

def get_authors(venue_author_dict,venue_cat_dict,customized_categories):
	'''
	Input: dictionary with venues and corresponding authors, dictionary with venues and corresponding categories
	Output: dictionary key: category value: authors who published in this category, set of authors that have published in the venues of the selected customized categoriess
	'''

	cat_authors_dict = {}
	authors_list = []
	for venue, category in venue_cat_dict.iteritems():
		if category != 'category':
			if (category in customized_categories) and (venue in venue_author_dict):
				authors = venue_author_dict[venue].keys()
				if category not in cat_authors_dict:
					cat_authors_dict[category] = []
				cat_authors_dict[category] += authors
				cat_authors_dict[category] = list(set(cat_authors_dict[category]))

	for cat,authors in cat_authors_dict.iteritems():
		authors_list += authors

	authors_set = list(set(authors_list))

	return cat_authors_dict, authors_set

def create_cat_venue_dictionary(venue_cat_dict):
	cat_venue_dict = {}

	for venue, cat in venue_cat_dict.iteritems():
		if cat not in cat_venue_dict:
			cat_venue_dict[cat] = []
		cat_venue_dict[cat].append(venue)

	return cat_venue_dict

def read_dict_to_list(file_name):

	# print('Reading txt file:',file_name)

	venue_list = json.load(open(file_name))

	# print('Reading completed.')
	return venue_list

def create_graph(category,cat_venue_dict,venue_author_citations_dict,venue_cat_dict,paper_id_authors_dict,paper_id_venue_dict,category_remaining_edges):
	'''
	Input:  name of graph category, 
			dictionary key: category val: list of venues, 
			dictionary key: venue key: author val: list of citations (paper ids),
			dictionary key: paper id and authors of the paper
			dictionary key: paper id and venue of the paper
	Output: a graph 
	'''
	# given the category of the graph
	# find the corresponding venues of this category from the cat, venues dictionary
	# for each of the venues, find authors that published in this venue - add the author as a node to the graph
	# 	for each author published in this venue find the citations the author has from this venue.
	# 		for each citation, i) see if paper that is being cited belongs to the same category, 1 i) if yes create an edge between author and authors of the paper, ii) if not continue


	DG=nx.DiGraph()
	edges = []
	category_venues_list = cat_venue_dict[category]
	for venue in category_venues_list:
		for from_author, citations in venue_author_citations_dict[venue].iteritems():
			for citation in citations:
				if citation in paper_id_venue_dict:
					citation_venue = paper_id_venue_dict[citation]
					if citation_venue in venue_cat_dict:
						citation_category = venue_cat_dict[citation_venue]
						citation_authors = paper_id_authors_dict[citation]
						if citation_category == category:	
							for to_author in citation_authors:
								if DG.has_edge(from_author, to_author):
									DG[from_author.encode('ascii','ignore')][to_author.encode('ascii','ignore')]['weight'] += 1
								else:
									DG.add_edge(from_author.encode('ascii','ignore'), to_author.encode('ascii','ignore'), weight=1)
						else:
							if citation_category not in category_remaining_edges:
								category_remaining_edges[citation_category] = []
							for to_author in citation_authors:
								tup = (from_author.encode('ascii','ignore'),to_author.encode('ascii','ignore'))
								category_remaining_edges[citation_category].append(tup)


	return DG, category_remaining_edges

def store_graph(file_name,DG):
	nx.write_gml(DG, file_name)

def add_cross_edges(DG,remaining_edges_list):

	for edge in remaining_edges_list:
		from_author = edge[0]; to_author = edge[1]

		if DG.has_edge(from_author, to_author):
			DG[from_author][to_author]['weight'] += 1
		else:
			DG.add_edge(from_author, to_author, weight=1)

	return DG

def parse_dataset_and_store_to_files(file_path,file_names,file_name_venue_paper_counts,file_name_pid_venue,file_name_pid_authors,file_name_venues_authors_citations):

	###### Loading the dblp dataset and storing it into sub-datasets STARTS here. ######

	# parse dblp .json files and store dataset to a list of .json objects 
	papers_json_format = transform_raw_file_to_list(file_path,file_names)

	# transform list with dblp .json objects to three dictionaries:
	# 1. A multi-index dictionary - key1: conference venue, key2: author that published in this venue, value: list of citations (paper ids) the author has given to papers of the same venue
	# 2. A dictionary - key: paper id, value: list of paper authors
	# 3. A dictionary - key: conference venue, value: number of papers published in this venue 
	(venue_author_citations_dict,venue_counter_dict,paper_id_authors_dict,paper_id_venue_dict) = create_dictionaries_from_dataset_list(papers_json_format)

	print('Storing venue paper counter dictionary to .txt file.')
	# store venue names with paper counts to .txt file
	store_dict_to_file(file_name_venue_paper_counts,venue_counter_dict)
	print('Storing completed.')

	print('Storing paper id venue dictionary to .txt file.')
	# store pid with venue to .txt file
	store_dict_to_file(file_name_pid_venue,paper_id_venue_dict)
	print('Storing completed.')

	print('Storing paper id authors dictionary to .pkl file.')
	# store paper id with authors to .pkl file
	store_dict_to_pkl_file(file_name_pid_authors,paper_id_authors_dict)
	print('Storing completed.')

	print('Storing venues hierarchical dictionary to .pkl file.')
	# # store hierarchical dictionary with venue, authors, citations to a file
	store_dict_to_pkl_file(file_name_venues_authors_citations,venue_author_citations_dict)
	print('Storing completed.')

	##### Loading the dblp dataset and storing it into sub-datasets ENDS here. ######

def store_all_graphs(category_graph,store,kcore,customized_categories):

	if store == True:
		print('Storing graphs to .gml files.')
		# storing graphs to .gml files
		for category, category_selected in customized_categories.iteritems():
			print('Category is:',category)
			if customized_categories[category] == 1:
				DG = category_graph[category]
				category_name_list = category.split(' ')
				file_name_graph = "-".join(category_name_list)+'-'+str(kcore)+'.gml'
				print('Storing graph of category:',category,'to file:',file_name_graph)
				store_graph('category_graphs/'+file_name_graph,DG)


def create_graph_gml_files(file_name_pid_venue,file_name_venues_authors_citations,file_name_pid_authors,file_name_venues_assignment_csv,customized_categories):

	# read paper id and venue .txt file
	paper_id_venue_dict = read_dict_to_list(file_name_pid_venue)

	# read hierarchical dictionary with venue, authors, citations to a file
	venue_author_citations_dict = read_dict_to_pkl_file(file_name_venues_authors_citations)

	# read paper id and corresponding authors from file
	paper_id_authors_dict = read_dict_to_pkl_file(file_name_pid_authors)

	# read venues with manually assigned category
	venue_cat_dict = read_venue_assignments_csv(file_name_venues_assignment_csv)

	# create a dictionary with categories and their corresponding venues
	cat_venue_dict = create_cat_venue_dictionary(venue_cat_dict)

	# get set of authors published in customized category venues
	cat_authors_dict, authors_set = get_authors(venue_author_citations_dict,venue_cat_dict,customized_categories)

	# dictionary that stores cross-edges that belong to different graphs
	category_remaining_edges = {}
	# dictionary that stores graph of each category
	category_graph = {}

	start_graph = time.time(); counter = 0
	print('Creating graphs.')
	# creating graph for each of the customized categories
	for category,category_selected in customized_categories.iteritems():
		if customized_categories[category] == 1:
			DG,category_remaining_edges = create_graph(category,cat_venue_dict,venue_author_citations_dict,venue_cat_dict,paper_id_authors_dict,paper_id_venue_dict,category_remaining_edges)
			# storing graphs of each category in a dictionary
			category_graph[category] = DG
			counter+=1
			# if counter == 3:
			# 	break

	end_graph = time.time()
	# time duration of main function
	print('Time required to create graphhs is:',end_graph - start_graph,'sec')

	start_graph = time.time()
	print('Adding cross-edges to the graph.')
	for category, category_selected in customized_categories.iteritems():
		print('Category is:',category)
		if customized_categories[category] == 1:
			DG = category_graph[category]; remaining_edges_list = category_remaining_edges[category]

			DG = add_cross_edges(DG,remaining_edges_list)
			category_graph[category] = DG
			print('New number of nodes of graph:',DG.number_of_nodes())
			print('New number of edges of graph:',DG.number_of_edges())

	end_graph = time.time()
	# time duration of main function
	print('Time required to add cross-edges is:',end_graph - start_graph,'sec')

	return category_graph

def read_graph(file_name):
	DG = nx.read_gml(file_name)
	return DG

def read_graph_gml_files(customized_categories):
	# loading graphs to .gml files
	category_graph = {}
	for category, category_selected in customized_categories.iteritems():
		print('Category is:',category)
		if customized_categories[category] == 1:
			start_graph = time.time()
			category_name_list = category.split(' ')
			file_name_graph = "-".join(category_name_list)+'.gml'
			print('Reading graph of category:',category,'to file:',file_name_graph)
			DG = read_graph('category_graphs/original_graphs/'+file_name_graph)
			category_graph[category] = DG
			end_graph = time.time()
			# time duration of main function
			print('Time required to read',file_name_graph,'is:',end_graph - start_graph,'sec')

	return category_graph

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

	plt.savefig(filename, 
	           bbox_inches='tight', 
	           transparent=True,
	           pad_inches=0, 
	           format='eps', 
	           dpi=1000)

	plt.close()

def get_graph_statistics(DG,category,kcore):

	print('Graph statistics for category:',category)

	# number of connected components
	num_connected_components = nx.number_weakly_connected_components(DG)
	print('Number of weakly connected components:',num_connected_components)
	print()
	category_name_list = category.split(' ')

	# in-degree distribution
	print('In-degree distribution.')
	filename = "-".join(category_name_list)+'-in-degree-hist-'+str(kcore)+'.eps'
	indegree_list = DG.in_degree().values()
	if indegree_list:
		print('Max value of in-degree',max(indegree_list))
		print()
		# print('Indegree list:',indegree_list)
		histogram(indegree_list,'plots/category_in_degree/'+filename,'in-degree','# nodes','Distribution of in-degree per category',100,None,1,500,True)

	# edge weight distribution
	print('Edge weight distribution.')
	filename = "-".join(category_name_list)+'-edge-weight-hist-'+str(kcore)+'.eps'
	weight_list = [e[2] for e in DG.edges_iter(data='weight', default=1)]
	if weight_list:
		print('Max value of weight edges',max(weight_list))
		print()
		# print('Weight list:',weight_list)
		histogram(weight_list,'plots/category_edge_weight/'+filename,'edge weight','# edges','Distribution of edge weights per category',60,None,1,500,True)

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
	filename = 'active-categories-hist-'+str(kcore)+'.eps'
	active_categories_list = authors_dict.values()
	if active_categories_list:
		print('Max value of active authors',max(active_categories_list))
		histogram(active_categories_list,'plots/active_users/'+filename,'# categories','# authors','Distribution of number of authors active in categories',60,None,1,500,True)

def jaccard_similarity_score(category_graph):
	jaccard_similarity_score = {}
	jaccard_similarity_score_lists = {}
	category_graph_pairs_list = list([{j for j in i} for i in combinations(category_graph, 2)])

	for pair in category_graph_pairs_list:
		pair = list(pair); 
		category1 = pair[0]; category2 = pair[1]
		print('Pair 1:',pair[0],'pair 2:',pair[1])
		DG1 = category_graph[category1]; DG2 = category_graph[category2]
		DG1nodes = list(DG1); DG2nodes = list(DG2)
		jaccard_similarity = (len(list((set(DG1nodes) & set(DG2nodes)))))/float(len(list((set(DG1nodes) | set(DG2nodes)))))
		tup = (category1,category2)
		jaccard_similarity_score[tup] = jaccard_similarity

		tup2 = (category2,jaccard_similarity)
		if category1 not in jaccard_similarity_score_lists:
			jaccard_similarity_score_lists[category1] = []
		jaccard_similarity_score_lists[category1].append(tup2)
		tup3 = (category1,jaccard_similarity)
		if category2 not in jaccard_similarity_score_lists:
			jaccard_similarity_score_lists[category2] = []
		jaccard_similarity_score_lists[category2].append(tup3)

	for key,alist in jaccard_similarity_score_lists.iteritems():
		alist_sorted = sorted(alist,key=itemgetter(1), reverse = True)
		jaccard_similarity_score_lists[key] = alist_sorted


	for key, alist in jaccard_similarity_score_lists.iteritems():
		print('Category:',key,' top-10 most similar:',alist[:10])


	# end time duration of main function
	end = time.time()
	# time duration of main function
	print('Time required to create graphs:',end - start,'sec')

def clean_graphs(category_graph,k_core,weighted):
	category_graph_clean = {}
#	# uncomment to read graphs from files 
	for category, graph in category_graph.iteritems():
		ebunch = graph.selfloop_edges()
		graph.remove_edges_from(ebunch)
		print('Creating core graph.')
		core_graph = nx.k_core(graph,k=k_core)
		if weighted == False:
			for u,v in core_graph.edges(data=False):
				core_graph[u][v]['weight'] = 1

		category_graph_clean[category] = core_graph

	return category_graph_clean

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

def get_all_statistics(category_graph):

	num_nodes = []; num_edges = []
	for category, DG in category_graph.iteritems():
		num_of_nodes = DG.number_of_nodes(); num_of_edges = DG.number_of_edges()
		num_nodes.append(num_of_nodes); num_edges.append(num_of_edges)
		get_graph_statistics(DG,category,kcore)

	get_authors_activity_per_graph(category_graph,kcore)

	if num_nodes:
		histogram_nodes_edges(num_nodes,'plots/graph_nodes/num-nodes-hist-'+str(kcore)+'.eps','# graphs','# nodes','Node distribution in dataset')
	if num_edges:
		# histogram of # of edges in graph datasets
		histogram_nodes_edges(num_edges,'plots/graph_edges/num-edges-hist-'+str(kcore)+'.eps','# graphs','# edges','Edge distribution in dataset')

def compute_indegree_outdegree(main_category, category_graph, weighted):
	'''
	Implementing indegree computation because it does not work in networkx for weighted/unweighted instances
	'''
	author_score = {}

	main_graph = category_graph[main_category]

	for edge in main_graph.edges():
		from_node = edge[0]; to_node = edge[1]
		if to_node not in author_score:
			author_score[to_node] = 1
		else:
			author_score[to_node] += 1

	for category, graph in category_graph.iteritems():

		if category != main_category:
			for edge in graph.edges():
				from_node = edge[0]; to_node = edge[1]
				if from_node not in author_score:
					author_score[from_node] = 1
				else:
					author_score[from_node] += 1

	author_score_list = []
	for author, score in author_score.iteritems():
		author_score_list.append((author, score))

	return author_score_list

def graph_ranking_heuristic(category_graph,weighted):
	category_graph_rankings_dict = {}

	for category, graph in category_graph.iteritems():
		# author_indegree_cat_list = compute_indegree(graph,weighted)
		author_indegree_outdegree_cat_list = compute_indegree_outdegree(category, category_graph, weighted)
		author_indegree_cat_sorted_list = sort_category_list(author_indegree_outdegree_cat_list)
		category_graph_rankings_dict[category] = author_indegree_cat_sorted_list

	# for synthetic example
	# category_graph_rankings_dict = synthetic_category_graph_rankings_dict

	solution_dict = half_approximation_algorithm(category_graph_rankings_dict, weighted)
	print('Solution dictionary:',solution_dict)

	return solution_dict

def half_approximation_algorithm(category_graph_rankings, weighted):
	solution_dict = {}
	selected_authors = {}

	# randomly select a skill 
	category = random.choice(category_graph_rankings.keys())
	category_ranking = category_graph_rankings[category]

	# assign the author with the most citations as a solution for this category
	top_author_score = category_ranking[0]

	solution_dict[category] = top_author_score
	top_author = top_author_score[0]; top_score= top_author_score[1]

	# stores authors seen so far
	selected_authors[top_author] = 1

	# delete category from ranking since solution for it has been assigned
	del category_graph_rankings[category]

	while category_graph_rankings:
		highest_rank = float("inf"); highest_score = -float("inf")

		for category, category_ranking in category_graph_rankings.iteritems():
			highest_rank_local = float("inf"); highest_score_local = -float("inf")

			for ranking, author_score in enumerate(category_ranking):
				author = author_score[0]
				if weighted == False and author not in selected_authors:
					if ranking < highest_rank_local:
						solution_local = author_score; highest_rank_local = ranking; category_local = category
						break
				elif weighted == True and author not in selected_authors:
					score = author_score[1]
					if score > highest_score_local:
						solution_local = author_score; highest_score_local = score; category_local = category
						break

			if weighted == False:
				if highest_rank_local < highest_rank:
					top_author_score = solution_local; highest_rank = highest_rank_local; solution_category = category_local

			elif weighted == True:
				score = author_score[1]
				if highest_score_local > highest_score:
					top_author_score = solution_local; highest_score = highest_score_local; solution_category = category_local

		solution_dict[solution_category] = top_author_score
		top_author = top_author_score[0]; top_score= top_author_score[1]

		# stores authors seen so far
		if top_author in selected_authors:
			print('This SHOULD NOT HAPPEN')
			sys.exit(0)
		else:
			selected_authors[top_author] = 1

		# delete category from ranking since solution for it has been assigned
		del category_graph_rankings[solution_category]

	return solution_dict

def compute_graph_score(solution_dict,category_graph):
	score = 0
	authors_set = [x[0] for x in solution_dict.values()]

	for category, author_score in solution_dict.iteritems():
		graph = category_graph[category]
		author_to = author_score[0]
		for author_from in authors_set:
			if author_to != author_from and graph.has_edge(author_from,author_to):
				score += graph[author_from][author_to]['weight']

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
			category_name_list = category.split(' ')
			filename = "-".join(category_name_list)+'-weighted-total-rank-'+str(kcore)+'.csv'
			with open('category_rankings/'+filename,'wb') as out:
			    csv_out=csv.writer(out)
			    for row in ranking:
			        csv_out.writerow(row)



def get_overlapping_graph_nodes(category_graphs):

	set_of_nodes = []
	category_num_nodes = []
	counter = 0
	for category,graph in category_graphs.iteritems():
		graph_nodes = list(graph.nodes())
		if counter == 0:
			intersecting_nodes = set(graph_nodes)

		intersecting_nodes = intersecting_nodes.intersection(set(graph_nodes))
		counter += 1


	print('Number of intersecting nodes:',len(intersecting_nodes))
	return intersecting_nodes

def main(args):

	# dblp dataset names dblp-ref-0.json dblp-ref-1.json dblp-ref-2.json dblp-ref-3.json
	file_path = args[0]; file_names = args[1:]; 

	file_name_venues_authors_citations = "venues_authors_citations.pkl"
	file_name_pid_authors = "pid_authors.pkl"
	file_name_pid_venue = "pid_venue.txt"
	file_name_venue_paper_counts = "venue_paper_counts.txt.txt"
	file_name_venues_assignment_csv = "venues_cat_assign_refined.csv"
	kcore = 5
	weighted = False

	# start time of main function
	start = time.time()

	# uncomment to parse dblp raw dataset and store to file
	# parse_dataset_and_store_to_files(file_path,file_names,file_name_venue_paper_counts,file_name_pid_venue,file_name_pid_authors,file_name_venues_authors_citations)
	
	num_of_skills_list = []; 
	score_graph_greedy_list = []; score_graph_ranking_list = []; score_graph_random_list = [];
	solution_graph_greedy_list = []; solution_graph_ranking_list = []; solution_graph_random_list = []
	time_list = []; loop_graph_greedy_time_list = []; loop_graph_ranking_time_list = []; loop_graph_random_time_list = [];
	max_score_list = []; categories_list = [] 

	i = 1

	for customized_categories in customized_categories_list:
		nodes = []
		edges = []
		max_endorsement = -float("inf")
		nodes_overlap = []
		total_start_time = time.time()
		max_score = (len(customized_categories))*(len(customized_categories)-1)
		num_of_skills = len(customized_categories)
		print('Considering the following customized categories:',customized_categories)
		# uncomment to create and store category graphs
		print('Creating graphs.')
		category_graph = create_graph_gml_files(file_name_pid_venue,file_name_venues_authors_citations,file_name_pid_authors,file_name_venues_assignment_csv,customized_categories)
		print('Cleaning graphs.')
		category_graph_clean = clean_graphs(category_graph,kcore,weighted)

		for category, graph in category_graph_clean.iteritems():
			# print('Category:',category,sorted(graph.in_degree(), key=lambda x: x[1], reverse=True)[:1])
			nodes_list = graph.nodes()
			edge_list = graph.edges()
			print('Length of nodes list:',len(nodes_list),'length of nodes',len(nodes))
			nodes = nodes + nodes_list
			nodes_overlap = list(set(nodes) & set(nodes_list))
			# nodes_overlap = set(nodes_overlap).intersection(nodes_list)
			edges = edges + edge_list
			nd = sorted(graph.degree_iter(),key=itemgetter(1),reverse=True)[0]
			in_degree = graph.in_degree(nd)
			if in_degree > max_endorsement:
				max_endorsement = in_degree


		print('Number of unique nodes',len(set(nodes)),'number of all nodes:',len(nodes))
		print('Avg End./Role:',len(edges)/len(category_graph_clean))
		print('Avg End./Expert:',len(edges)/len(set(nodes)))
		print('Max End./Expert',max_endorsement)
		print('Number of overlapping nodes:',len(set(nodes_overlap)))

		category_graph_clean = get_overlapping_graph_nodes(category_graph_clean)

		loop_start_time = time.time()
		random_solution, random_score, random_edges, random_score_mean, random_score_std = mra.graph_algorithm_random(category_graph_clean,weighted)
		loop_elapsed_time_random = time.time() - loop_start_time
		# print('Solution for random truth is:',random_solution)
		print('Solution score for random truth is:',random_score,'mean is:',random_score_mean,'standard deviation is:',random_score_std)
		print('Running time is:',loop_elapsed_time_random)
		print()
		score_graph_random_list.append(random_score);
		loop_graph_random_time_list.append(loop_elapsed_time_random); solution_graph_random_list.append(random_solution)

		loop_start_time = time.time()
		greedy_solution, greedy_score, greedy_edges, greedy_score_mean, greedy_score_std = mra.graph_algorithm_greedy(category_graph_clean,weighted)
		loop_elapsed_time_greedy = time.time() - loop_start_time
		print('Solution for greedy truth is:',greedy_solution)
		print('Solution Edges for greedy truth is:',greedy_edges)
		print('Solution score for greedy truth is:',greedy_score,'mean is:',greedy_score_mean,'standard deviation is:',greedy_score_std)
		print('Running time is:',loop_elapsed_time_greedy)
		print()
		score_graph_greedy_list.append(greedy_score);
		loop_graph_greedy_time_list.append(loop_elapsed_time_greedy); solution_graph_greedy_list.append(greedy_solution)


		loop_start_time = time.time()
		graph_rank_solution, graph_rank_score, graph_rank_edges = mra.graph_ranking_heuristic(category_graph_clean,weighted)
		loop_elapsed_time_graph_rank = time.time() - loop_start_time
		print('Solution for graph rank truth is:',graph_rank_solution)
		print('Solution Edges for graph rank truth is:',graph_rank_edges)
		# algorithm_list.append('GraphRanking'); score_list.append(graph_rank_score); num_of_nodes_list.append(num_of_nodes)
		print('Solution score for graph ranking is:',graph_rank_score,'mean is:',graph_rank_score,'standard deviation is:',0)
		print('Running time is:',loop_elapsed_time_graph_rank)
		print()
		score_graph_ranking_list.append(graph_rank_score);
		loop_graph_ranking_time_list.append(loop_elapsed_time_graph_rank); solution_graph_ranking_list.append(graph_rank_solution)

		total_elapsed_time = time.time() - total_start_time
		num_of_skills_list.append(num_of_skills);
		time_list.append(total_elapsed_time);
		max_score_list.append(max_score); 
		categories_list.append(customized_categories);

		df = pd.DataFrame()
		df['score graph random'] = score_graph_random_list; df['score graph greedy'] = score_graph_greedy_list; df['score graph ranking'] = score_graph_ranking_list; 
		df['solution graph random'] = solution_graph_random_list; df['solution graph greedy'] = solution_graph_greedy_list; df['solution graph ranking'] = solution_graph_ranking_list;
		df['loop time graph random'] = loop_graph_random_time_list; df['loop time graph greedy'] = loop_graph_greedy_time_list; df['loop time graph ranking'] = loop_graph_ranking_time_list;
		df['max score'] = max_score_list;
		df['skills'] = num_of_skills_list; df['time'] = time_list; # df['probability'] = probability_list;
		df['categories'] = categories_list;


if __name__ == '__main__':
    main(sys.argv[1:])
    print('Exiting main!')
