from __future__ import print_function

import sys
import json 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv

# customized_scientific_categories = {'machine learning':1, 'artificial intelligence':1,'natural language':1, 'computational linguistics':1, 'neural information':1, 
# 									'world wide web':1, 'data mining':1, \
# 									'security and privacy':1, 'computer graphics':1, 'data management systems':1,\
# 									'theory of computation':1, 'computer vision':1 , 'pattern recogntion':1, ''
# 									'human computer interaction':1,'communication hardware, interfaces and storage':1,'control methods':1}

# A Dynamic Programming based Python program for edit 
# distance problem 
def editDistDP(str1, str2): 
	m = len(str1); n = len(str2)
	# Create a table to store results of subproblems 
	dp = [[0 for x in range(n+1)] for x in range(m+1)] 
  
	# Fill d[][] in bottom up manner 
	for i in range(m+1): 
		for j in range(n+1): 
			if i == 0: 
				dp[i][j] = j

			elif j == 0: 
				dp[i][j] = i

			elif str1[i-1] == str2[j-1]: 
				dp[i][j] = dp[i-1][j-1] 
			else: 
				dp[i][j] = 1 + min(dp[i][j-1],        # Insert 
									dp[i-1][j],        # Remove 
									dp[i-1][j-1])    # Replace 

	return dp[m][n]

def read_json_to_list(json_file):

	categories_list = []
	json_file = "/Users/smnikolakaki/Dropbox/Research/mutual_respect/IJCAI2019_MutualRespect_local/code/datasets/acm_classification/acm-classification-assignments.json"

	with open(json_file) as json_file:  
		data = json.load(json_file)
		for cat in data['categories']:
			categories_list.append(cat)

	return categories_list

def extend_customized_categories(categories_list):
	'''
	Input: categories with related sub-categories list
	Output: dictionary - key: field keywords related to customized scientific categories, value: corresponding customized scientific category 
	'''

	ext_customized_categories_dict = {}
	for i,cat in enumerate(categories_list):
		overlapping_elements = list(set(cat.keys()) & set(customized_scientific_categories.keys()))
		if overlapping_elements:
			for key,val in cat.iteritems():
				if key not in ext_customized_categories_dict:
					ext_customized_categories_dict[key] = overlapping_elements[0]


	return ext_customized_categories_dict

def histogram(frequencies):   

	frequencies = [x for x in frequencies if x >= 100]
	file_hist = "conf_paper_distr.png"
	print('New number of venues:',len(frequencies))
	yscale = None
	x_label = "Number of papers"
	y_label = "Number of conferences"
	htype = "bar"
	bins = 20
	title = "Distribution of the number of skills required by projects in Guru"    
	# width = 0.8
	xmin = min(frequencies) - 1; xmax = max(frequencies) + 1
	# You typically want your plot to be ~1.33x wider than tall.  
	# Common sizes: (10, 7.5) and (12, 9)  
	plt.figure(figsize=(20, 10))  
	frequency_set = list(set(frequencies))
	frequency_set.sort()
    
	min_value = frequency_set[0]; max_value = frequency_set[len(frequency_set)-1]

	num_of_bins = np.arange(bins+1)-0.5
	print('Min Hist Value:',min_value)
	print('Max Hist Value:',max_value)

	plt.figure()
	if htype == "bar":
		(n, bins, patches) = plt.hist(frequencies, color="#3F5D7D", normed=False,\
			histtype='bar', ec='black')
		# (n, bins, patches) = plt.hist(frequencies, bins=num_of_bins, width = width, color="#3F5D7D", normed=False,\
		# 	histtype='bar', ec='black')
    
	plt.xticks(fontsize = 14)
	plt.yticks(fontsize = 14)
	plt.xlabel(x_label,fontsize=16)
	plt.ylabel(y_label,fontsize=16)
	plt.xlim([xmin, xmax])
	if yscale == "log":
		plt.yscale('log')

	plt.title(title)
	plt.savefig(file_hist, bbox_inches="tight")
        
	max_value_of_all = frequency_set[len(frequency_set)-1]    

def filter_popular_venues(venue_counter_dict,thresh):

	pop_venue_counter_dict = dict((k, v) for k, v in venue_counter_dict.iteritems() if v >= thresh)

	return pop_venue_counter_dict

def write_list_to_csv(file_name,venue_list):

	with open(file_name, 'wb') as f:
		writer = csv.writer(f)
		for val in venue_list:
			writer.writerow([val.lower().encode('utf-8')])

def assign_venue_to_category(venue_list,ext_customized_categories_dict):

	edit_distance_venue_to_category = {}
	assigned_conferences_dict = {}

	category_keywords = ext_customized_categories_dict.keys()

	for i,venue in enumerate(venue_list):
		# print('I:',i,'venue is:',venue)
		venue_str = venue.lower()
		for j,cat_str in enumerate(category_keywords):
			score = editDistDP(venue_str,cat_str)
			if venue_str not in edit_distance_venue_to_category:
				edit_distance_venue_to_category[venue_str] = []
			edit_distance_venue_to_category[venue_str].append((cat_str,score))
	
	print('Edit distance dictionary:',edit_distance_venue_to_category)	

def read_found_venues_assign_csv(file_name_venues_assignment_csv):
	venue_cat_dict = {}
	with open(file_name_venues_assignment_csv, "rb") as f:
		reader = csv.reader(f, delimiter="\t")
		for i, line in enumerate(reader):
			csv_line_split = line[0].strip().split(',')	
			venue_name = csv_line_split[0].strip(); cat_name = csv_line_split[2].strip();
			venue_cat_dict[venue_name] = cat_name 

	return venue_cat_dict

def print_key_value(venue_cat_dict):
	for key,value in venue_cat_dict.iteritems():
		print('Key:',key,'Value:',value)

def read_dict_to_list(file_name):

	# print('Reading txt file:',file_name)

	venue_list = json.load(open(file_name))

	# print('Reading completed.')
	return venue_list

def main(args):
	file_path_venues = args[0]; file_name_venues = args[1]; 
	file_path_categories = args[2];file_name_categories = args[3]

	file_path_name_venues = file_path_venues + '/' + file_name_venues
	file_path_name_categories = file_path_categories + '/' + file_name_categories

	file_name_venues_csv = "venues_category.csv"
	file_name_venues_assignment_csv = "venues_cat_assign_refined.csv"
	file_name_venues_counter_txt = "venue_paper_counts.txt"
	# threshold to find popular conferences
	paper_thresh = 1000

	# read file with venue names
	# venue_counter_dict = read_dict_to_list(file_path_name_venues)
	# print('Number of all venues with at least one paper is:',len(venue_counter_dict))

	# keep a dictionary of popular venues with more than paper_thresh papers
	# pop_venue_counter_dict = filter_popular_venues(venue_counter_dict,paper_thresh)
	# print('Number of all venues with >',paper_thresh,'papers is:',len(pop_venue_counter_dict))

	# write popular venues to a csv file
	# write_list_to_csv(file_name_venues_csv,pop_venue_counter_dict.keys())

	# read csv file with venues manually assigned to categories to a dictionary (key: venue, val: assigned category)
	# venue_cat_dict = read_found_venues_assign_csv(file_name_venues_assignment_csv)

	# read file .txt file with all venues and the corresponding number of papers to a dictionary (key: venue, val: num of papers)
	# venues_counter_dict = read_dict_to_list(file_name_venues_counter_txt)

	# format name of venues in the dictionaries: remove commas and transform to lower case 
	# venues_counter_dict = {i.replace(",","").lower():val for i,val in venues_counter_dict.iteritems()}
	# venues_counter_pop_dict = {i:val for i,val in venues_counter_dict.iteritems() if val>paper_thresh}

	# find the overlap between the popular venues (papers > paper_thresh) and the ones that have been manually assigned to categories
	# overlap_pop_assigned = list(set(venues_counter_pop_dict.keys())&set(venue_cat_dict.keys()))
	# pop_not_assigned = list(set(venues_counter_pop_dict.keys())-set(venue_cat_dict.keys()))
	# assigned_not_pop = list(set(venue_cat_dict.keys())-set(venues_counter_pop_dict.keys()))

	# print venues that are manually assigned but do not have > paper_thresh papers in the dataset
	# for venue in assigned_not_pop:
	# 	try:
	# 		print('Venue:',venue,'paper count:',venues_counter_dict[venue])
	# 	except:
	# 		print('No key:',venue)

	# print venues that have > paper_thresh papers but are not manually assigned
	# print('Popular but not assigned conferences:')
	# for i in pop_not_assigned:
	# 	try:
	# 		print(i.encode('utf-8'),'count:',venues_counter_dict[i])
	# 	except:
	# 		continue



if __name__ == '__main__':
    main(sys.argv[1:])
    print('Exiting main!')

