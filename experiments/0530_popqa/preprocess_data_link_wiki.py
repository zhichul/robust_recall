from pathlib import Path
from datasets import load_dataset
import sys
import tqdm

if '../../lib/doc_cot' not in sys.path:
    sys.path.append('../../lib/doc_cot')
from doc_cot.corpus import wiki

lookup = wiki.WikiLookup()
datatype='train'
for datatype in tqdm.tqdm(['train', 'dev', 'test']):
    for split in tqdm.tqdm(range(1,10)):
        for freq in tqdm.tqdm(['0_to_1000', '1000_to_10000', '10000_to_100000', '100000_to_inf']):
            path = f"data/raw/splits/fifty_fifty/split{split}/{freq}/{datatype}.parquet"
            opath = f"data/raw/splits/fifty_fifty/split{split}/{freq}/{datatype}.wikilinked.parquet"
            decond_dataset = load_dataset('parquet', data_files=path)['train']
            decond_dataset = decond_dataset.map(lambda x: {'s_docs': lookup.get_doc(x['s_wiki_title']), 'o_docs': lookup.get_doc(x['o_wiki_title'])})
            decond_dataset.to_parquet(opath)
            cnt = 0
            for i in tqdm.tqdm(range(len(decond_dataset))):
                cnt += int(len(decond_dataset[i]['s_docs']) == 0)
            print(path, 'unlinked:', cnt, cnt / len(decond_dataset))