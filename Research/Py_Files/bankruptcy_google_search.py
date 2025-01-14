from googlesearch import search
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
from typing import List, Tuple

def google_bankruptcy_search(query: str, num_results: int = 10) -> List[List[str]]:
    """
    Perform a Google search for bankruptcy-related news and return structured results.
    
    Args:
        query (str): Search query string
        num_results (int): Number of results to return (default: 10)
    
    Returns:
        List[List[str]]: List of results, where each result is a list containing
                         [title, summary, date, url]
    """
    results = []
    
    try:
        # Perform the Google search
        search_results = search(query, 
                              num_results=num_results,
                              lang="en",
                              advanced=True)
        
        return search_results
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
       