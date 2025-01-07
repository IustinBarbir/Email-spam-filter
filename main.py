import os
import re
import sys
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Functia pentru curatarea emailurilor
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)  # Elimina tag-urile HTML
    text = re.sub(r'[^a-z\s]', ' ', text)  # Pastreaza doar literele si spatiile
    text = re.sub(r'\s+', ' ', text)  # Inlocuieste spatiile multiple cu un singur spatiu
    return text

# Citirea fisierelor dintr-un folder
def load_emails_from_folder(folder_path, label):
    emails = []
    labels = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                email_content = file.readlines()  # Citeste toate liniile din fisier
                if not email_content:
                    print(f"Empty file: {filename}")
                    continue  # Sari peste fișierele goale

                # Prima linie este subiectul, restul este corpul emailului
                subject = email_content[0].strip()
                body = ''.join(email_content[1:]).strip()
                cleaned_email = clean_text(subject + ' ' + body)

                # Adăugăm emailul și eticheta în liste
                emails.append(cleaned_email)
                labels.append(label)
        except Exception as e:
            print(f"Eroare la citirea fișierului {filename}: {e}")

    return emails, labels

# Functia pentru a scrie informatii despre proiect
def write_project_info(output_file):
    project_info = """Project_name: Email Spam Filter
Student_name: Dragos Gavrilut
Alias_Student: Gavrilut_Dragos
Project_version: 1.0"""

    with open(output_file, 'w') as file:
        file.write(project_info)
    print(f"Project information written to {output_file}")

# Functia pentru a analiza folderul
def scan_folder(folder_path, output_file, vectorizer, model):
    results = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                email_content = file.readlines()  # Citeste toate liniile din fisier
                if not email_content:
                    continue  # Sari peste fișierele goale

                # Prima linie este subiectul, restul este corpul emailului
                subject = email_content[0].strip()
                body = ''.join(email_content[1:]).strip()
                cleaned_email = clean_text(subject + ' ' + body)

                # Transformam emailul intr-un vector folosind TF-IDF
                email_tfidf = vectorizer.transform([cleaned_email])

                # Predicție
                prediction = model.predict(email_tfidf)[0]
                label = 'cln' if prediction == 0 else 'inf'

                # Adaugam rezultatul
                results.append(f"{filename}|{label}")
        except Exception as e:
            print(f"Eroare la citirea fișierului {filename}: {e}")

    # Scriem rezultatele in fisier
    with open(output_file, 'w') as file:
        for result in results:
            file.write(result + '\n')
    print(f"Scan results written to {output_file}")

# Functia principala care gestioneaza argumentele
def main():
    parser = argparse.ArgumentParser(description="Email Spam Filter")

    parser.add_argument("-info", type=str, help="Output file for project information")
    parser.add_argument("-scan", type=str, nargs=2, help="Folder to scan and output file")

    args = parser.parse_args()

    if args.info:
        write_project_info(args.info)

    elif args.scan:
        folder_path = args.scan[0]
        output_file = args.scan[1]

        # Incarcam emailurile de antrenament
        clean_folder = 'D:\\Lot1\\Clean'  # Inlocuieste cu calea corecta
        spam_folder = 'D:\\Lot1\\Spam'  # Inlocuieste cu calea corecta

        clean_emails, clean_labels = load_emails_from_folder(clean_folder, 0)  # 0 pentru clean
        spam_emails, spam_labels = load_emails_from_folder(spam_folder, 1)  # 1 pentru spam

        emails = clean_emails + spam_emails
        labels = clean_labels + spam_labels

        # Impartim datele in seturi de antrenament si testare
        X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

        # Transformam emailurile in caracteristici folosind TF-IDF
        vectorizer = TfidfVectorizer(max_features=10000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Antrenam modelul Naive Bayes
        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)

        # Apelam functia pentru scanare
        scan_folder(folder_path, output_file, vectorizer, model)

if __name__ == "__main__":
    main()
