
import requests
from bs4 import BeautifulSoup


class KickSoup:
    """
    Retrieve kickstarter data from a specified project url.
    Requires 'requests' and 'bs4.BeautifulSoup'
    """

    def __init__(self, url):

        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'html.parser')

        self.page_title = soup.title.text

        # self.project_name = soup.h2.text
        # self.project_name = soup.find('h2').text
        for project in soup.find_all('h2'):
            if 'project-name' in project.attrs['class']:
                self.project_name = project.text

        # self.blurb = soup.h2.next_element.next_element.next_element.text
        self.blurb = soup.p.find_next('p').text

        # self.goal = soup.find(attrs={'class':'money'}).text
        # self.goal = soup.find('span', class_='money').text        
        for money in soup.find_all('span', class_='money'):
            if "pledged" in money.previous:
                self.goal = money.text