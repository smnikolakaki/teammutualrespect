# encoding: utf-8

from __future__ import print_function
import csv
import pandas as pd
import sys
import mutual_respect_algorithms as mra
import time
import random 

def main(args):

	not_included_dict = {'Player':1, 'Year':1, 'Pos':1, 'Age':1, 'Tm': 1, 'Unnamed: 0':1, 'blanl':1, 'blank2':1, 'GS Games Started':1, 'MP Minutes Played':1,\
						'ORB%Offensive Rebound Percentage':1,'DRB% Defensive Rebound Percentage':1,'TOV%Turnover Percentage':1,'TOV Turnovers':1,\
						'PF Personal Fouls':1, 'G':1, 'GS':1, 'MP':1, 'TRB%':1, 'USG%':1, 'WS/48':1, 'OBPM':1, 'DBPM':1, 'BPM':1, 'VORP':1, 'ORB':1,
						'DRB':1,'TRB':1,'AST':1,'STL':1,'BLK':1,'TOV':1,'PF':1
						}

	# '3PAr':3-Point Attempt Rate, '3P%':3-Point Field Goal Percentage, '2P%':2-Point Field Goal Percentage, 
	# 'ORB%':Offensive Rebound Percentage, 'DRB%':Defensive Rebound Percentage 'TRB%':Total Rebound Percentage,
	# 'AST%':Assist Percentage, 'STL%':Steal Percentage,'BLK%':Block Percentage,
	# 'BLK%':Block Percentage, 'FG':Field Goal Percentage, 'FT%':Free Throw Percentage,
	# 'OBPM':Offensive Box Plus/Minus, 'DBPM' Defensive Box Plus/Minus, 'VORP' Value Over Replacement

	included_dict = {'3P':1,'TRB':1,'AST':1,'STL':1,'BLK':1,'FG':1,'2P':1,'FT':1 ,'OBPM':1,'DBPM':1,'VORP':1}


	file_path = 'Seasons_Stats.csv'
	num_of_nodes_list = []; num_of_skills_list = []; algorithm_list = []; score_list = []; solution_list = []
	time_list = []; loop_time_list = []; year_list = []
	weighted = False
	filename = 'nba.csv'

	random.seed(1)
	for year in range(2010,2018,1):
		year_list.append(year); year_list.append(year); year_list.append(year)
		total_start_time = time.time()
		print('year is:',year)
		nba_dataset_ranking = {}

		df = pd.read_csv(file_path)
		df = df.loc[df['Year'] == year]
		df = df.groupby(['Player','Age']).mean().reset_index()
		df = df.loc[df['G'] >= 27]
		df['MPG'] = df['MP']/df['G']
		df = df.loc[df['MPG'] >= 15]
		df = df.fillna(0)
		# df = df.head(60)
		print('For year:',year,'number of players is:',df.size,len(df))
		category_names = list(df.columns.values)
		names = df['Player'].tolist()

		for i,category_name in enumerate(category_names):
			if category_name in included_dict:
				category_list = df[category_name].tolist()
				category_ranking = []
				for j,value in enumerate(category_list):
					tup = (names[j],value)
					category_ranking.append(tup)

				# sort based on value from highest to lower
				category_ranking.sort(key=lambda elem: elem[1], reverse=True)
				nba_dataset_ranking[category_name] = category_ranking

		num_of_nodes = len(names)
		num_of_skills = len(nba_dataset_ranking.keys())

		skill_author_index_dict = mra.create_skill_author_index(nba_dataset_ranking)

		max_score_solution = mra.max_score_algo(nba_dataset_ranking,weighted)

		# print('Number of elements in max score solution is:',len(max_score_solution))
		if max_score_solution != None:
			max_score = mra.compute_ranking_score(max_score_solution,nba_dataset_ranking,skill_author_index_dict)
			print('Max score is:',max_score)
			print('Max score solution:',max_score_solution)
		else:
			print('There is no optimal solution.')

		half_approx_start_time = time.time()
		half_approx_solution_dict = mra.half_approximation_algorithm(nba_dataset_ranking,weighted)
		half_approx_elapsed_time = time.time() - half_approx_start_time
		print('Elapsed time for half approximation:',half_approx_elapsed_time)
		time_list.append(half_approx_elapsed_time)
		half_approx_score = mra.compute_ranking_score(half_approx_solution_dict,nba_dataset_ranking,skill_author_index_dict)
		# print('1 Solution dictionary from half approximation algorithm is:',half_approx_solution_dict)
		print('Score is for half approximation:',half_approx_score)
		print()
		# storing solutions
		num_of_nodes_list.append(num_of_nodes); num_of_skills_list.append(num_of_skills)
		algorithm_list.append('Half Approximation'); score_list.append(half_approx_score)
		solution_list.append(half_approx_solution_dict)

		half_approx_var_start_time = time.time()
		half_approx_var_solution_dict = mra.half_approximation_per_rank_algorithm(nba_dataset_ranking,weighted,skill_author_index_dict)
		half_approx_var_elapsed_time = time.time() - half_approx_var_start_time
		print('Elapsed time for half approximation variation:',half_approx_var_elapsed_time)
		time_list.append(half_approx_var_elapsed_time)
		half_approx_var_score = mra.compute_ranking_score(half_approx_var_solution_dict,nba_dataset_ranking,skill_author_index_dict)
		# print('Solution dictionary from half approximation variation algorithm is:',half_approx_solution_dict)
		print('Score is for half approximation variation:',half_approx_var_score)
		print()
		# storing solutions
		num_of_nodes_list.append(num_of_nodes); num_of_skills_list.append(num_of_skills)
		algorithm_list.append('Half Approximation Variation'); score_list.append(half_approx_var_score)
		solution_list.append(half_approx_var_solution_dict)


		

		greedy_start_time = time.time()
		greedy_solution_dict = mra.greedy_algorithm_random_skills_stop(nba_dataset_ranking,weighted,skill_author_index_dict)
		greedy_elapsed_time = time.time() - greedy_start_time
		time_list.append(greedy_elapsed_time)
		greedy_score = mra.compute_ranking_score(greedy_solution_dict,nba_dataset_ranking,skill_author_index_dict)
		print('Solution dictionary is for greedy algorithm:',greedy_solution_dict)
		print('Score is for greedy algorithm:',greedy_score)
		print()
		num_of_nodes_list.append(num_of_nodes); num_of_skills_list.append(num_of_skills)
		algorithm_list.append('Greedy'); score_list.append(greedy_score)
		solution_list.append(greedy_solution_dict)
	
		total_elapsed_time = time.time() - total_start_time
		loop_time_list.append(total_elapsed_time); loop_time_list.append(total_elapsed_time); loop_time_list.append(total_elapsed_time);

		df = pd.DataFrame()
		df['year'] = year_list
		df['algorithm'] = algorithm_list; 
		df['score'] = score_list; 
		df['nodes'] = num_of_nodes_list;
		df['skills'] = num_of_skills_list; df['time'] = loop_time_list;
		df['solution'] = solution_list; df['loop_time'] = time_list;

		df.to_csv(filename, encoding='utf-8', index = False)




if __name__ == '__main__':
    main(sys.argv[1:])
    print('Exiting main!')