import random
import Bio.Entrez as bp
import Bio.Medline as bm


def pull_pubmed_abstracts(search_terms, n_abstracts, email_address):
    bp.email = email_address  # NCBI likes to have your email (so keep requests to a normal amount)
    pubmed_id = search_and_sample(n=int(n_abstracts*2),  # multiply by 1.5 to ensure there is enough valid abstracts
                                  search_terms=search_terms)
    return extractor(include=pubmed_id, max_n=n_abstracts)


def search_and_sample(n, search_terms):
    # Conducting search and pulling out the random sample of *n* articles
    handle = bp.esearch(db='pubmed', term=search_terms, retmax=4000000)
    record = bp.read(handle)
    handle.close()
    id_list = record['IdList']
    if len(id_list) < n:
        return id_list
    else:  # random sample of pubmed IDs if there is enough
        return random.sample(id_list, n)


def extractor(include, max_n):
    # Empty list for valid abstracts
    valid_abstracts = []

    # Fetching the results from PubMed
    handle = bp.efetch(db="pubmed", id=include, rettype="medline", retmode="text")
    records = list(bm.parse(handle))

    # Extracting the meta-data we want
    index = 0
    while (index < len(include)-1) and (len(valid_abstracts) < max_n):
        r = records[index]
        abstract = r.get("AB", "?")
        if abstract == "?":
            pass  # if there is NO abstract, then ignore
        else:  # if valid abstract add to list
            valid_abstracts.append(abstract.lower())
        index += 1  # adding one to the index

    return valid_abstracts
