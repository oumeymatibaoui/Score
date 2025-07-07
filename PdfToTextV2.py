import re
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
from langdetect import detect
import torch
from torch.nn.functional import cosine_similarity

class PdfToText:
    def __init__(self, pdf_path, text_cv, embedding_model='distilbert-base-uncased'):
        self.pdf_path = pdf_path
        self.text_cv = text_cv
        self.embedding_model = TransformerDocumentEmbeddings(embedding_model)
        self.keywords = {
            "experience": r"(experience|work experience|professional experience|job history|employment history|parcours professionnel|historique d'emploi|antécédents professionnels|stage / internship|freelance|missions)",
            "education": r"(education|diploma|degrees|certification|formation|études|étude|educational background|diplomas|certifications|parcours académique|études supérieures)",
            "skills": r"""
                \b(
                    python|java|c\+\+|c#|typescript|javascript|html|css|php|go|dart|scala|swift|r|
                    sql|mysql|mongodb|postgresql|oracle|nosql|
                    react|angular|vue\.js|django|flask|spring|express|node\.js|laravel|
                    docker|kubernetes|jenkins|ansible|terraform|
                    git|github|gitlab|bitbucket|
                    figma|adobe\s?xd|photoshop|illustrator|
                    agile|scrum|kanban|waterfall|jira|trello|notion|
                    linux|windows|mac\s?os|
                    excel|word|powerpoint|microsoft\s?office|
                    rest|graphql|api|soap|
                    flutter|android|ios|react\s?native|
                    pandas|numpy|scikit-learn|tensorflow|keras|pytorch|machine\s?learning|deep\s?learning|data\s?science|nlp|
                    cisco|tcp/ip|dns|vpn|firewall|networking|ccna|ccnp|
                    helpdesk|it\s?support|troubleshooting|maintenance|diagnostic|sav|
                    rh|recrutement|formation|paie|gestion\s?du\s?personnel|grh|
                    communication|gestion\s?de\s?projet|leadership|organisation
                )\b
            """
        }
        self.standard_descriptions = {
            "experience": {
                "en": (
                    "experience\n"
                    "[Company Name], [City] — Internship ([Month–Month Year])\n"
                    "Worked on backend development tasks using [Technology/Stack], such as building APIs or managing data."
                ),
                "fr": (
                    "Période : [Mois–Mois Année]  |  Entreprise : [Nom de l'entreprise], [Ville]\n"
                    "Poste , Stagiaire au sein du service [nom du service]\n"
                    "Tâches principales , suivi des activités quotidiennes, préparation de documents, aide à l’organisation du travail.\n"
                    "Responsabilités , respecter les délais, transmettre les informations, participé aux réunions internes.\n"
                    "Résultats , meilleure compréhension du fonctionnement d’un service et développement du sens des responsabilités."
                )
            },
            "skills": {
                "en": (
                    "skills\n"
                    "Python, Java, JavaScript, and frameworks like Angular, with experience in databases such as MySQL and Firebase, "
                    "Experience with preventive maintenance, diagnostic tools, electrical and mechanical repairs, and safety protocols, "
                    "Familiar with HR software (e.g., SAGE, SAP HR), employee management, payroll basics, recruitment procedures, and labor law knowledge."
                ),
                "fr": (
                    "Python, Java, JavaScript, et frameworks comme Angular, avec expérience dans les bases de données telles que MySQL et Firebase, "
                    "outils de diagnostic, réparations électriques et mécaniques, et protocoles de sécurité, "
                    "Familier avec les logiciels RH (par exemple, SAGE, SAP RH), gestion des employés, bases de la paie, procédures de recrutement, et connaissances en droit du travail."
                )
            },
            "education": {
                "en": (
                    "education\n"
                    "Graduated in [Month Year] with a [Degree Type] in [Field of Study] from [University Name], [City]. "
                    "Earned a [Degree Type] in [Field of Study] in [Month Year] from [University Name] in [City], [Country]."
                ),
                "fr": (
                    "Diplômé en [Mois Année] avec un [Type de diplôme] en [Domaine d'études] de [Nom de l'université], [Ville]. "
                    "Obtenu un [Type de diplôme] en [Domaine d'études] en [Mois Année] de [Nom de l'université] à [Ville], [Pays]. "
                    "Mémoire de fin d'études portant sur [Sujet du mémoire], encadré par [Nom du professeur]. "
                    "Participation active à des projets universitaires et à des séminaires en lien avec [Thème ou domaine]."
                )
            }
        }

    def clean_text(self, text):
        """Clean text by removing special characters, keeping only alphanumeric characters, spaces, and newlines."""
        text = re.sub(r'[^\w\s\n]', '', text)
        return text

    def detect_language(self, text):
        """Detect the language of the input text, returning 'en' or 'fr'."""
        try:
            sample = text[:1000].lower()
            lang = detect(sample)
            return "fr" if lang == "fr" else "en" if lang == "en" else f"other({lang})"
        except Exception:
            return "unknown"

    def extract_sections_lines(self, text_cv, keyword_regex, num_lines=3):
        """Extract lines following a keyword match in the CV text."""
        lines = text_cv.split('\n')
        extracted = []
        for i, line in enumerate(lines):
            if re.search(keyword_regex, line, re.IGNORECASE):
                section_lines = lines[i:i + num_lines + 1]  # Include the matched line
                extracted.append((line.strip(), section_lines))
        return extracted

    def embed(self, sentence_text):
        """Embed a sentence using the Transformer model and return the embedding tensor."""
        sentence = Sentence(sentence_text)
        self.embedding_model.embed(sentence)
        return sentence.embedding

    def score_similarity(self, base_sent, target_sent):
        """Calculate cosine similarity between two sentence embeddings."""
        return cosine_similarity(base_sent.unsqueeze(0), target_sent.unsqueeze(0), dim=1).item()

    def expand_lines(self, initial_lines, reference_text):
        """Expand initial lines by adding lines that improve similarity to a reference text."""
        base = self.embed(reference_text)
        current_text = ' '.join(initial_lines)
        current_score = self.score_similarity(base, self.embed(current_text))
        expanded_lines = list(initial_lines)
        lines = self.text_cv.split('\n')
        for next_line in lines:
            new_text = current_text + " " + next_line
            new_score = self.score_similarity(base, self.embed(new_text))
            if new_score >= current_score:
                expanded_lines.append(next_line)
                current_text = new_text
                current_score = new_score
            else:
                break
        return expanded_lines

    def extract_skills(self):
        """Extract skills from the CV text using the skills regex pattern."""
        skills_keywords = self.keywords['skills']
        pattern = re.compile(skills_keywords, re.IGNORECASE | re.VERBOSE)
        lines = self.text_cv.split('\n')
        skills = []
        for line in lines:
            matches = pattern.findall(line)
            skills.extend(matches)
        return list(set(skills))  # Remove duplicates