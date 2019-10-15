"""
    Scraping ACM Classification categories from official site to json objects  
    ------------------------------------------
    
    :author: Sofia Nikolakaki, <smnikol@bu.edu>
    :programming-language: Python 2.7
    
    Site to be scraped: https://dl.acm.org/ccs/ccs_flat.cfm
    Output is a .json file where there are main categories, sub-categories and keywords
"""

# Imports 
from __future__ import print_function
from BeautifulSoup import BeautifulSoup
from selenium import webdriver

import sys
import os
import logging
import argparse
import requests
import json
import re

## Selenium

# driver.get(url)

## Beautiful Soup
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2272.76 Safari/537.36"

parser = argparse.ArgumentParser(description='Scrape ACM Classification System hierarchy')
parser.add_argument('-v','--verbose',help="Set log level to debug", action="store_true")

args = parser.parse_args()

# Log settings #
log = logging.getLogger(__name__)
log.setLevel(logging.ERROR)
if args.verbose:
    log.setLevel(logging.DEBUG)
loghandler = logging.StreamHandler(sys.stderr)
loghandler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
log.addHandler(loghandler)


# Web  #
base_url = "https://dl.acm.org/ccs/ccs_flat.cfm"
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2272.76 Safari/537.36"

def store_list_json(json_list,json_file):

    with open(json_file , 'w+') as json_file:
        json.dump(json_list, json_file)

def read_json_list(json_file):

    with open(json_file) as json_file:  
        data = json.load(json_file)
        for p in data['categories']:
            print('Name: ' + p['main-category-id'])
            print('')

def scrape_acm_categories(json_objects_file,json_assignments_file):
    '''
    Input: None
    Output: None
    This function scrapes the categories/sub-categories/keywords of each sub-category from the official acm classification website
    '''
    id_to_category_name_dict = {}; id_category_hierarch_dict = {}; id_category_hierarch_list = {}
    id_category_hierarch_assign_list = {}
    
    id_category_hierarch_list['categories'] = []
    id_category_hierarch_assign_list['categories'] = []

    log.info("URL TO REQUEST: %s \n" % base_url)
    headers = { 'User-Agent' : user_agent }
    response = requests.get(base_url, headers=headers)
    html = response.text.encode('utf-8')
    soup = BeautifulSoup(html)

    # extracting main categories with corresponding ids
    main_categories = soup.findAll("a", attrs={"class": "boxedlinkh", "title":"Assign This CCS Concept"})
    print('Number of main categories:',len(main_categories))
    for i,cat in enumerate(main_categories):
        num_id = cat["href"].split('"')[1][2:]; main_cat_name = cat.text
        id_to_category_name_dict[num_id] = main_cat_name

    # extracting sub-categories with corresponding ids
    sub_categories = soup.findAll("a", attrs={"class": "boxedlink", "title":"Assign This CCS Concept"})
    print('Number of sub-categories:',len(sub_categories))
    for i,cat in enumerate(sub_categories):
        num_id_string = cat["href"].split('"')[1];
        num_ids = num_id_string.split(".")[1:]
        sub_cat_name = cat.text
        id_to_category_name_dict[num_ids[-1]] = sub_cat_name
        id_category_hierarch_dict = {}; id_category_hierarch_assign_dict = {}

        if len(num_ids) == 2:
            main_cat = num_ids[0]; sub_cat = num_ids[1]; 
            id_category_hierarch_dict['main-category-id'] = main_cat; id_category_hierarch_dict['sub-category-id'] = sub_cat;
            id_category_hierarch_dict['main-category-name'] = id_to_category_name_dict[main_cat].lower(); id_category_hierarch_dict['sub-category-name'] = id_to_category_name_dict[sub_cat].lower();

            id_category_hierarch_assign_dict[id_to_category_name_dict[main_cat].lower()] = 1; id_category_hierarch_assign_dict[id_to_category_name_dict[sub_cat].lower()] = 2;


        if len(num_ids) == 3:
            main_cat = num_ids[0]; sub_cat = num_ids[1]; sub_sub_cat = num_ids[2]
            id_category_hierarch_dict['main-category-id'] = main_cat; id_category_hierarch_dict['sub-category-id'] = sub_cat; id_category_hierarch_dict['sub-sub-category-id'] = sub_sub_cat
            id_category_hierarch_dict['main-category-name'] = id_to_category_name_dict[main_cat].lower(); id_category_hierarch_dict['sub-category-name'] = id_to_category_name_dict[sub_cat].lower(); \
            id_category_hierarch_dict['sub-sub-category-name'] = id_to_category_name_dict[sub_sub_cat].lower()

            id_category_hierarch_assign_dict[id_to_category_name_dict[main_cat].lower()] = 1; id_category_hierarch_assign_dict[id_to_category_name_dict[sub_cat].lower()] = 2;
            id_category_hierarch_assign_dict[id_to_category_name_dict[sub_sub_cat].lower()] = 3;

        if len(num_ids) == 4:
            main_cat = num_ids[0]; sub_cat = num_ids[1]; sub_sub_cat = num_ids[2]; sub_sub_sub_cat = num_ids[3]
            id_category_hierarch_dict['main-category-id'] = main_cat; id_category_hierarch_dict['sub-category-id'] = sub_cat; id_category_hierarch_dict['sub-sub-category-id'] = sub_sub_cat; \
            id_category_hierarch_dict['sub-sub-sub-category-id'] = sub_sub_sub_cat
            id_category_hierarch_dict['main-category-name'] = id_to_category_name_dict[main_cat].lower(); id_category_hierarch_dict['sub-category-name'] = id_to_category_name_dict[sub_cat].lower(); \
            id_category_hierarch_dict['sub-sub-category-name'] = id_to_category_name_dict[sub_sub_cat].lower(); id_category_hierarch_dict['sub-sub-sub-category-name'] = id_to_category_name_dict[sub_sub_sub_cat].lower()

            id_category_hierarch_assign_dict[id_to_category_name_dict[main_cat].lower()] = 1; id_category_hierarch_assign_dict[id_to_category_name_dict[sub_cat].lower()] = 2;
            id_category_hierarch_assign_dict[id_to_category_name_dict[sub_sub_cat].lower()] = 3; id_category_hierarch_assign_dict[id_to_category_name_dict[sub_sub_sub_cat].lower()] = 4;

        if len(num_ids) == 5:
            main_cat = num_ids[0]; sub_cat = num_ids[1]; sub_sub_cat = num_ids[2]; sub_sub_sub_cat = num_ids[3]; sub_sub_sub_sub_cat = num_ids[4]
            id_category_hierarch_dict['main-category-id'] = main_cat; id_category_hierarch_dict['sub-category-id'] = sub_cat; id_category_hierarch_dict['sub-sub-category-id'] = sub_sub_cat; \
            id_category_hierarch_dict['sub-sub-sub-category-id'] = sub_sub_sub_cat; id_category_hierarch_dict['sub-sub-sub-sub-category-id'] = sub_sub_sub_sub_cat
            id_category_hierarch_dict['main-category-name'] = id_to_category_name_dict[main_cat].lower(); id_category_hierarch_dict['sub-category-name'] = id_to_category_name_dict[sub_cat].lower(); \
            id_category_hierarch_dict['sub-sub-category-name'] = id_to_category_name_dict[sub_sub_cat].lower(); id_category_hierarch_dict['sub-sub-sub-category-name'] = id_to_category_name_dict[sub_sub_sub_cat].lower()
            id_category_hierarch_dict['sub-sub-sub-sub-category-name'] = id_to_category_name_dict[sub_sub_sub_sub_cat].lower()

            id_category_hierarch_assign_dict[id_to_category_name_dict[main_cat].lower()] = 1; id_category_hierarch_assign_dict[id_to_category_name_dict[sub_cat].lower()] = 2;
            id_category_hierarch_assign_dict[id_to_category_name_dict[sub_sub_cat].lower()] = 3; id_category_hierarch_assign_dict[id_to_category_name_dict[sub_sub_sub_cat].lower()] = 4;
            id_category_hierarch_assign_dict[id_to_category_name_dict[sub_sub_sub_sub_cat].lower()] = 5;

        if len(num_ids) == 6:
            main_cat = num_ids[0]; sub_cat = num_ids[1]; sub_sub_cat = num_ids[2]; sub_sub_sub_cat = num_ids[3]; sub_sub_sub_sub_cat = num_ids[4]; sub_sub_sub_sub_sub_cat = num_ids[4]
            id_category_hierarch_dict['main-category-id'] = main_cat; id_category_hierarch_dict['sub-category-id'] = sub_cat; id_category_hierarch_dict['sub-sub-category-id'] = sub_sub_cat; \
            id_category_hierarch_dict['sub-sub-sub-category-id'] = sub_sub_sub_cat; id_category_hierarch_dict['sub-sub-sub-sub-category-id'] = sub_sub_sub_sub_cat
            id_category_hierarch_dict['sub-sub-sub-sub-sub-category-id'] = sub_sub_sub_sub_sub_cat
            id_category_hierarch_dict['main-category-name'] = id_to_category_name_dict[main_cat].lower(); id_category_hierarch_dict['sub-category-name'] = id_to_category_name_dict[sub_cat].lower(); \
            id_category_hierarch_dict['sub-sub-category-name'] = id_to_category_name_dict[sub_sub_cat].lower(); id_category_hierarch_dict['sub-sub-sub-category-name'] = id_to_category_name_dict[sub_sub_sub_cat].lower()
            id_category_hierarch_dict['sub-sub-sub-sub-category-name'] = id_to_category_name_dict[sub_sub_sub_sub_cat].lower()
            id_category_hierarch_dict['sub-sub-sub-sub-sub-category-name'] = id_to_category_name_dict[sub_sub_sub_sub_sub_cat].lower()

            id_category_hierarch_assign_dict[id_to_category_name_dict[main_cat].lower()] = 1; id_category_hierarch_assign_dict[id_to_category_name_dict[sub_cat].lower()] = 2;
            id_category_hierarch_assign_dict[id_to_category_name_dict[sub_sub_cat].lower()] = 3; id_category_hierarch_assign_dict[id_to_category_name_dict[sub_sub_sub_cat].lower()] = 4;
            id_category_hierarch_assign_dict[id_to_category_name_dict[sub_sub_sub_sub_cat].lower()] = 5; id_category_hierarch_assign_dict[id_to_category_name_dict[sub_sub_sub_sub_sub_cat].lower()] = 6;

        id_category_hierarch_list['categories'].append(id_category_hierarch_dict)
        id_category_hierarch_assign_list['categories'].append(id_category_hierarch_assign_dict)

    store_list_json(id_category_hierarch_list,json_objects_file)
    store_list_json(id_category_hierarch_assign_list,json_assignments_file)
            
if __name__ == "__main__":

    json_objects_file = "/Users/smnikolakaki/Dropbox/Research/mutual_respect/IJCAI2019_MutualRespect_local/code/datasets/acm_classification/acm-classification.json"
    json_assignments_file = "/Users/smnikolakaki/Dropbox/Research/mutual_respect/IJCAI2019_MutualRespect_local/code/datasets/acm_classification/acm-classification-assignments.json"
    scrape_acm_categories(json_objects_file,json_assignments_file)
    
    




