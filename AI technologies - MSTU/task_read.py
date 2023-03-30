import docx
from googletrans import Translator


def task_describe(prnt=True):
    doc = docx.Document('Описание задания.docx')
    if prnt: print('\n'.join([p.text for p in doc.paragraphs]))


def dataset_describe(prnt=True, lang="en"):
    with open("Mechanical DataSet/mechanical-analysis.names.txt", "r") as file:
        text = list()
        for line in file:
            text.append(line.strip()) #.strip("\n")
    text = '\n'.join(text)

    translator = Translator()
    result = translator.translate(text, dest=lang, src='en')

    if prnt: print(result.text)
    
    return result