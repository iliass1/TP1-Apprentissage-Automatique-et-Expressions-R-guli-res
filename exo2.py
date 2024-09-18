import re

# Étape 1 : Charger le fichier texte contenant les données
with open('data.txt', 'r') as file:
    texte = file.read()

# Étape 2 : Extraction des informations à l'aide des expressions régulières

# 1. Extraction des adresses email valides
emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', texte)
print("Emails extraits :")
for email in emails:
    print(email)

# 2. Extraction des numéros de téléphone dans plusieurs formats (ex : 10 chiffres, formats internationaux)
# Formats pris en charge : +33 6 12 34 56 78, (555) 123-4567, 123-456-7890, 1234567890
telephones = re.findall(r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}', texte)
print("\nNuméros de téléphone extraits :")
for tel in telephones:
    print(tel)

# 3. Extraction des dates dans plusieurs formats (DD/MM/YYYY, MM-DD-YYYY, etc.)
# Formats pris en charge : 12/08/2021, 08-15-2022, 01/01/23, etc.
dates = re.findall(r'\b\d{2}[/-]\d{2}[/-]\d{2,4}\b', texte)
print("\nDates extraites :")
for date in dates:
    print(date)

# 4. Extraction des URLs avec vérification qu'elles utilisent le protocole HTTPS
urls = re.findall(r'https://[a-zA-Z0-9./?=_-]+', texte)
print("\nURLs extraites (HTTPS uniquement) :")
for url in urls:
    print(url)

