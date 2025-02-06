# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:16:09 2023

@author: wittmann
"""

import glob
import os

def create_stim(sentences, folder, condition):
    
    for idx, sentence in enumerate(sentences, start=1):
        if not os.path.exists(folder):
            os.makedirs(folder)
        print(sentence)
        filename = f"{condition}_{idx}"
        full_path = f"{folder}/{filename}.txt"
        # Write the sentence to the file
        with open(full_path, 'w') as file:
            file.write(sentence)
    
sentences1 = ['Cette horrible femme fume un paquet par jour.',
'Le coureur de tête a fini tout dernier.',
'Ce cuisinier nous a donné une intoxication alimentaire.',
'Le méchant prêtre m’a fait un doigt d’honneur.',
'La maîtresse n’a pas pu résoudre ce calcul.',
'Notre vilain camarade nous a froncé les sourcils.',
'Mon cruel patron a décidé de me licencier.',
'Ce fainéant jeune homme n’a porté aucun carton.',
'Ce musicien maladroit a fait tomber sa guitare.',
'Son enfant a fait une grimace pour me saluer.',
'Cette dame inconsciente a traversé au feu rouge.',
'Ce joueur a perdu beaucoup d’argent à la loterie.',
'Mon ami malade préfère continuer à manger gras.',
'Ce cancre fait plein de bêtises en classe.',
'Le peintre a sali le parquet de peinture.',
'Mon frère a fixé l’étagère à l’envers.']


sentences2 = ['Cette jolie femme court dix kilomètres par jour.',
'Il a battu trois record à cette course.',
'Il a encore gagné un concours de cuisine.',
'Notre gentil superviseur nous a envoyé des fleurs.',
'La maîtresse a résolu un calcul très difficile.',
'Notre gentil camarade nous a fait un sourire.',
'Mon super patron m''a offert une augmentation.',
'Ce jeune homme a porté tous mes cartons.',
'Ce pianiste a gagné un concours de musique.',
'Son enfant me salue quand on se croise.',
'Cette femme a la bonne habitude de mettre de l’argent de côté.',
'Mon mari a gagné beaucoup d’argent à la loterie.',
'Mon ami a décidé de faire un régime.',
'Ce gentil garçon écoute toujours bien en classe.',
'Le peintre a tout fini en une journée.',
'Mon père a construit l’étagère en deux minutes.']
 

sentences3 = [
    "Elle est en bonne santé.",
    "C'est un résultat spectaculaire.",
    "C’est un chef de qualité.",
    "C’est un geste respectueux.",
    "Elle est compétente.",
    "Ce garçon est aimable.",
    "C'est une bonne nouvelle.",
    "Il est très utile.",
    "C’est un grand artiste.",
    "Il est bien éduqué.",
    "Elle est très prudente.",
    "Il est très chanceux.",
    "C’est une bonne idée.",
    "C’est un élève sérieux.",
    "C’est du beau travail.",
    "C’est un génie."]
  
sentences4 = [
    "Elle est en mauvaise santé.",
    "C'est un résultat banal.",
    "C'est un chef incompétent.",
    "C'est un geste irrespectueux.",
    "Elle est incompétente.",
    "Ce garçon est méchant.",
    "C'est une mauvaise nouvelle.",
    "Il est inutile.",
    "C'est un artiste maladroit.",
    "Il est mal éduqué.",
    "Elle est imprudente.",
    "Il est malchanceux.",
    "C'est une mauvaise idée.",
    "C'est un élève puéril.",
    "C'est du vilain travail.",
    "C'est un idiot"
]

os.chdir(r'C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\text')
    
create_stim(sentences1, 'text_stim/contexts', 'CN')
create_stim(sentences2, 'text_stim/contexts', 'CP')
create_stim(sentences3, 'text_stim/statements', 'SP')
create_stim(sentences4, 'text_stim/statements', 'SN')


def get_number_condition(filename):
    return int(filename.split("_")[1].split(".")[0]), filename.split("_")[0]


def combine_and_write(context_files, statement_files, output_directory):
    for context_path in context_files:
        context_number, context_condition = get_number_condition(os.path.basename(context_path))
        for statement_path in statement_files:
            statement_number, statement_condition = get_number_condition(os.path.basename(statement_path))
            if context_number == statement_number:
                with open(context_path, 'r') as context_file, open(statement_path, 'r') as statement_file:
                    context_content = context_file.read().strip()
                    statement_content = statement_file.read().strip()
                
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)

                context_statement_path = os.path.join(output_directory, f"{context_condition}_{statement_condition}_{context_number}.txt")

                with open(context_statement_path, 'w') as output_file:
                    output_file.write(f'{context_content} {statement_content}')


# Assuming you have already obtained the file paths using glob
contexts = glob.glob(os.path.join('text_stim', 'contexts', 'C*.txt'))
statements = glob.glob(os.path.join('text_stim', 'statements', 'S*.txt'))

# Replace with the actual path of your output directory
output_directory = 'text_stim/conversations'

combine_and_write(contexts, statements, output_directory)

