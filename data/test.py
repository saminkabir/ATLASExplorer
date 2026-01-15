import pandas as pd
df = pd.read_csv("processed/processed_polyvector_results.csv")
with open("processed/processed_polyvector_results.csv") as f:
    content = f.read()
    pairs=[
        ('gooaq-distilroberta-768-normalized','gooaq'),
        ('laion-clip-512-normalized','laion'),
        ('yahoo-minilm-384-normalized','yahoo'),
        ( 'codesearchnet-jina-768-cosine','codesearchnet'),
        ('word2vex','word2vec'),
        ('space1V','space'),
        ('imagenet','imageNet'),
        ('celeba-resnet-2048-cosine','celeba'),
        ('ccnews-nomic-768-normalized','ccnews'),
        ('landmark-dino-768-cosine','landmark'),
        ('simplewiki-openai-3072-normalized','simplewiki'),
        ('yi-128-ip','yi'),
        ( 'yandex-200-cosine', 'yandex'),
        ( 'llama-128-ip', 'llama'),
        ('arxiv-nomic-768-normalized','arxiv'),
        ('agnews-mxbai-1024-euclidean','agnews'),
        ('text-to-image','text'),
        ('coco-nomic-768-normalized','coco'),
        ('astro1m','astro'),
        ('atro','astro')
        
    ]
    for pair in pairs:
        content=content.replace(pair[0],pair[1])
    with open("processed/processed_polyvector_results_filtered.csv", "w") as f:
        f.write(content)
        
        
        
        
df = pd.read_csv("processed/processed_polyvector_results_filtered.csv")
df = df[~df["dataset"].str.contains("hundred", case=False, na=False)]
df = df[~df["dataset"].str.contains("twenty", case=False, na=False)]
df = df[~df["dataset"].str.contains("imagenet-clip-512-normalized", case=False, na=False)]
df = df[~df["dataset"].str.contains("landmark-nomic-768-normalized", case=False, na=False)]
df = df[~df["dataset"].str.contains("siftsmall", case=False, na=False)]
datasets,model = set(df['dataset'].tolist()),set(df['model'].tolist())
print(sorted(list(datasets)))
print(len(list(datasets)))
print(sorted(list(model)))
print(len(list(model)))

df.to_csv("processed/processed_polyvector_results_filtered.csv", index=False)

mp=dict()
lst=[]
models=set()
for index, row in df.iterrows():
    model=row['model']
    dataset=row['dataset']
    models.add(model)
    if not dataset in mp:
        mp[dataset]=set()
    mp[dataset].add(model)
    
for key in mp.keys():
    print(key,len(mp[key]))
    lst.append((-len(mp[key]),key))
    
lst.sort()
print(lst[34])
print(lst[39])
used_datasets=[]
for i in range(40):
    used_datasets.append(lst[i][1])
print(used_datasets,list(models))


model_dict=dict()

for index, row in df.iterrows():
    model=row['model']
    dataset=row['dataset']
    if dataset in used_datasets:
        if not model in model_dict:
            model_dict[model]=set()
        model_dict[model].add(dataset)
    
lst=[]    
for key in model_dict:
    lst.append((-len(model_dict[key]),key))

lst.sort()
print(lst)



