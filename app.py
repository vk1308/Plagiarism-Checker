import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup as bs

# Directory of the files
DIR = "files"

# URL and Headers
url = input("Enter the URL (Wikipedia Only): ")

headers = {
    "Host": "en.wikipedia.org",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:99.0) Gecko/20100101 Firefox/99.0"
}

# Request the url with headers and convert to text
r = requests.get(url, headers=headers).text

# Creates a bs object to perform parsing
soup = bs(r, 'lxml')
TITLE = soup.find('h1').get_text()

# Dictionary to store parsed string
parsed_dic = {}

# Parsing
para = soup.find('p', class_=None)

for d in para.find_all('sup'):
    d.decompose()
parsed_dic["Introduction"] = para.get_text()

for tag in soup.find_all('h2'):
    sib = tag.find_next_sibling('p')
    if sib is None:
        continue
    p = ""
    while(sib is not None and sib.find_previous_sibling('h2').find('span').get_text() == tag.find('span').get_text()):
        for d in sib.find_all('sup'):
            d.decompose()

        p += sib.get_text()
        sib = sib.find_next_sibling('p')
    parsed_dic[tag.find('span').get_text()] = p


# Creates seperate txt files for every heading in Wikipedia
for key, value in parsed_dic.items():
    with open(DIR + os.sep + key + '.txt', 'w', encoding="utf-8") as f:
        f.write(value)

# Creates a list of files and its data
student_files = [doc for doc in os.listdir(DIR) if doc.endswith('.txt')]

student_notes = [open(DIR + os.sep + _file, encoding='utf-8').read()
                 for _file in student_files]


# Creates vectors of the data of each file
def vectorize(Text): return TfidfVectorizer().fit_transform(Text).toarray()
def similarity(doc1, doc2): return cosine_similarity([doc1, doc2])


# Compares every files vector with each other
vectors = vectorize(student_notes)
s_vectors = list(zip(student_files, vectors))
plagiarism_results = set()


# Function to compare the vectors
def check_plagiarism():
    global s_vectors
    for student_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))
        del new_vectors[current_index]
        for student_b, text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            student_pair = sorted((student_a, student_b))
            score = (student_pair[0], student_pair[1], sim_score)
            plagiarism_results.add(score)
    return plagiarism_results


# Print the result
print("Not very Similar:")
for data in check_plagiarism():
    if data[1].split('.')[0] == 'test' and data[2] <= 0.5:
        print(data)
print()
print("Are kind of Similar:")
for data in check_plagiarism():
    if data[1].split('.')[0] == 'test' and data[2] > 0.5 and data[2] <= 0.75:
        print(data)
print()
print("A lot Similar:")
for data in check_plagiarism():
    if data[1].split('.')[0] == 'test' and data[2] > 0.75:
        print(data)
