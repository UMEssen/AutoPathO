
def return_prompt(task: str, doc: str, loc_codes: dict):
    if task == "icd_10":
        prompt = [
            'Bestimme den am besten passenden Topographiecode der ICD-O-3 Klassifikation für die anatomische Lokalisation des Tumors aus dem folgenden Pathologiebefund.',
            '\nWähle den am besten passenden Code aus der folgenden Liste und gib ausschließlich den Code ohne zusätzliche Erklärung oder Beschreibung zurück.'
            '\nFalls keine exakte Übereinstimmung möglich ist, gib den nächstliegenden Code an.'
            '\nListe der möglichen Topographie-Codes: ' + ', '.join(f"{lc['localization']} ({lc['code']})" for lc in loc_codes),
            '\nPathologiebefund: ' + doc,
            "\nAntwort: "
        ]
    elif task == "icd_10_wo_locs":
        prompt = [
            'Bestimme den am besten passenden Topographiecode der ICD-O-3 Klassifikation für die anatomische Lokalisation des Tumors aus dem folgenden Pathologiebefund.',
            '\nGib ausschließlich den Code ohne zusätzliche Erklärung oder Beschreibung zurück.'
            '\nPathologiebefund: ' + doc,
            "\nAntwort: "
        ]
    elif task == "icd_o":
        prompt = [
            'Bestimme den am besten passenden Morphologiecode der ICD-O-3 Klassifikation aus dem folgenden Pathologiebefund.',
            '\nGib ausschließlich den fünfstelligen Morphologie-Code zurück ohne zusätzliche Erklärung oder Beschreibung zurück.'
            '\nHinweise zur Morphologie-Kodierung: Der Morphologie-Code beschreibt den Zelltyp der Neubildung und ihr biologisches Verhalten. Morphologische Schlüsselnummern bestehen aus fünfstelligen Kodenummern des Nummernkreises 8000/0 bis 9989/3. Die ersten vier Stellen kennzeichnen den histologischen Typ des Tumors. Die fünfte Stelle hinter dem Schrägstrich (/) gibt das biologische Verhalten an: 0 für benigne (gutartig), 1 für unklare Dignität (borderline/unsicher), 2 für in situ (nicht invasiv, prämaligne), 3 für in maligne (bösartig, invasiv), 6 für sekundär/metastasierend und 9 für maligne (unspezifisch).'
            '\nPathologiebefund: ' + doc,
            "\nAntwort: "
        ]
    else:
        # Invalid task, throw an error
        raise ValueError("Invalid task specified. Choose from 'icd_10', 'icd_10_wo_locs', or 'icd_o'.")    
    return prompt