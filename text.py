from flair.models import SequenceTagger
from flair.data import Sentence
import re

# Load Flair NER model
tagger = SequenceTagger.load("ner")

# Example resume text
text = """
Oumeyma Tibaoui graduated with a Master's degree in Computer Science from MIT. 
She worked as a software engineer at Google from 2020 to 2022.
She is proficient in Python, JavaScript, Angular, and SQL.
"""

# Create sentence and predict entities
sentence = Sentence(text)
tagger.predict(sentence)

# Collect results
education_entities = []
experience_entities = []
dates = []
skills_found = []

# Example skill list
skill_keywords = ["python", "javascript", "angular", "java", "sql", "react", "flutter", "spring", "communication", "teamwork", "problem-solving"]

# Search entities from Flair
for entity in sentence.get_spans('ner'):
    label = entity.get_label("ner").value
    value = entity.text

    if label == "ORG":
        experience_entities.append(value)
    elif label == "MISC":
        education_entities.append(value)
    elif label == "DATE":
        dates.append(value)

# Match skills manually (case-insensitive)
for skill in skill_keywords:
    if re.search(rf"\b{skill}\b", text.lower()):
        skills_found.append(skill.capitalize())

# Remove duplicates
education_entities = list(set(education_entities))
experience_entities = list(set(experience_entities))
skills_found = list(set(skills_found))
dates = list(set(dates))

# Display results
print("\n🎓 Education-related terms:")
print(education_entities)

print("\n💼 Experience-related companies:")
print(experience_entities)

print("\n📅 Dates / Durations:")
print(dates)

print("\n🛠️ Skills detected:")
print(skills_found)
