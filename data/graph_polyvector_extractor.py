import pandas as pd
import os

dataset_infos=[('enron', 94987, 200, 200, 1369, 20), ('lendb', 1000000, 100, 100, 256, 100), ('deep', 1000000, 200, 200, 256, 20), ('geofon', 275174, 100, 100, 128, 100), ('tiny5m', 5000000, 1000, 1000, 384, 100), ('crawl', 1989995, 10000, 10000, 300, 100), ('netflix', 17770, 1000, 1000, 300, 100), ('glove', 1192514, 200, 200, 100, 20), ('sun', 79106, 200, 200, 512, 20), ('audio', 53387, 200, 200, 192, 20), ('cifar', 50000, 200, 200, 512, 20), ('instancegm', 1000000, 100, 100, 128, 100), ('random', 1000000, 200, 200, 100, 20), ('OBST2024', 1000000, 300, 300, 256, 100), ('word2vec', 1000000, 1000, 1000, 300, 100), ('millionSong', 992272, 200, 200, 420, 20), ('Meier2019JGR', 1000000, 300, 300, 256, 100), ('nuswide', 268643, 200, 200, 500, 20), ('ethz', 36643, 100, 100, 256, 100), ('Music', 1000000, 100, 100, 100, 100), ('ISC_EHB_DepthPhases', 1000000, 300, 300, 256, 100), ('MNIST', 69000, 200, 200, 784, 20), ('space1V', 1000000, 100, 100, 100, 100), ('vcseis', 160178, 100, 100, 256, 100), ('OBS', 1000000, 300, 300, 256, 100), ('sald1m', 1000000, 100, 100, 128, 100), ('nytimes', 290000, 100, 100, 256, 100), ('text-to-image', 1000000, 100, 100, 200, 100), ('movielens', 10677, 1000, 1000, 150, 100), ('yahoomusic', 136736, 100, 100, 300, 100), ('seismic1m', 1000000, 100, 100, 256, 100), ('siftsmall', 10000, 100, 100, 128, 100), ('NEIC', 1000000, 300, 300, 256, 100), ('Iquique', 578853, 300, 300, 256, 100), ('PNW', 1000000, 300, 300, 256, 100), ('bigann', 1000000, 100, 100, 128, 100), ('lastfm', 292385, 100, 100, 65, 100), ('txed', 519589, 100, 100, 256, 100), ('stead', 1000000, 100, 100, 256, 100), ('gist', 1000000, 1000, 1000, 960, 100), ('trevi', 99900, 200, 200, 4096, 20), ('notre', 332668, 200, 200, 128, 20), ('uqv', 1000000, 10000, 10000, 256, 100), ('sift', 1000000, 10000, 10000, 128, 100), ('imageNet', 2340373, 200, 200, 150, 20), ('ukbench', 1097907, 200, 200, 128, 20), ('astro1m', 1000000, 100, 100, 256, 100), ('Yelp', 77079, 100, 100, 50, 100)]

def getqn(dataset):
    dataset=dataset.replace("'",'')
    if(dataset[-1]=='2'):
        dataset=dataset[:-1]
    dataset=dataset.replace('-1m','').replace('1m','').replace('1V','').replace('Cifar','cifar').replace('gooaq-distilroberta-768-normalized','gooaq').replace('MNSIT','MNIST').replace('MNJST','MNIST').replace('agnews-mxbai-1024-euclidean','agnews').replace('ccnews-nomic-768-normalized','ccnews')
    for dataset_info in dataset_infos:
        if dataset_info[0]==dataset:
            return dataset,dataset_info[2]
    return dataset,200 
    

def get_processed_data_graph():    
    df = pd.read_csv("raw/Polyvector results - standalone graph based model's result.csv")
    datas=[]
    for index, row in df.iterrows():
        model=row['model']
        dataset=row['dataset']
        search_time=row['search_time without refinemnt']
        construction_time=row['construction time']
        recall=row['recall without refinemet']
        dataset,qn=getqn(dataset)
        datas.append([dataset,model,qn/search_time,construction_time,recall,1])
    return datas

def get_processed_data_quantization():    
    df = pd.read_csv("raw/Polyvector results - standalone quantization results.csv")
    datas=[]
    for index, row in df.iterrows():
        model=row['model']
        dataset=row['dataset']
        search_time=row['searchx1']
        construction_time=row['construction-time']
        recall=row['recallx1']
        search_time2=row['searchx2']
        recall2=row['recallx2']
        dataset,qn=getqn(dataset)
        datas.append([dataset,model,qn/search_time,construction_time,recall,1])
        datas.append([dataset,model,qn/search_time2,construction_time,recall2,2])
    return datas




def get_processed_data_tree():    
    df = pd.read_csv("raw/Polyvector results - tree_results.csv")
    datas=[]
    for index, row in df.iterrows():
        model=row['model']
        dataset=row['dataset']
        if model=='annoy':
            search_time=float(row['search_time'])
            construction_time=row['construction_time (parallel)']
            recall=row['recall']
            dataset,qn=getqn(dataset)
            datas.append([dataset,model,qn/search_time,construction_time,recall,1])
        if model=='scann':
            construction_time=row['construction_time (parallel)']
            search_time2=float(row['search_timex_refined (parallel)'])
            recall2=row['refined_recall']
            dataset,qn=getqn(dataset)
            datas.append([dataset,model,qn/search_time2,construction_time,recall2,2])
    return datas


def get_processed_data_hash():    
    df = pd.read_csv("raw/Polyvector results - hash_results.csv")
    datas=[]
    for index, row in df.iterrows():
        model=row['Method']
        if model=='PM-LSH' or model=='DB-LSH':
            dataset=row['Dataset']
            search_time=row['QPS (sequential)']
            construction_time=row['Construction Time (parallel)']
            recall=row['Refined Recall']
            dataset,qn=getqn(dataset)
            datas.append([dataset,model,search_time,construction_time,recall,1])
    return datas


def get_processed_data_vaq():    
    df = pd.read_csv("raw/Polyvector results - all_possible_standalone_vaq.csv")
    datas=[]
    for index, row in df.iterrows():
        model='VAQ'
        dataset=row['dataset']
        search_time=row['qps']
        construction_time=row['construction_time']
        recall=row['recall']
        refinementx=row['refinementx']
        dataset,qn=getqn(dataset)
        datas.append([dataset,model,search_time,construction_time,recall,refinementx])
    return datas

def get_processed_data_diskann(folderName,model):
    datas=[]
    for file in os.listdir('raw/'+folderName):
        with open('raw/'+folderName+'/'+file, "r", encoding="utf-8") as f:
            content = f.read()
            count = content.count('recall@')
            if count>=1:
                recall=content.split("'recall@': ")[1].split('}')[0]
                dataset=content.split("'dataset': '")[1].split("'")[0]
                search_time=float(content.split("'search_time': ")[1].split(",")[0])
                construction_time=content.split("'construction_time': ")[1].split(",")[0]
                dataset,qn=getqn(dataset)
                datas.append([dataset,model,qn/search_time,construction_time,recall,1])
            if count>=2:
                recall=content.split("'recall@': ")[2].split('}')[0]
                dataset=content.split("'dataset': '")[2].split("'")[0]
                search_time=float(content.split("'search_time': ")[2].split(",")[0])
                construction_time=content.split("'construction_time': ")[2].split(",")[0]
                dataset,qn=getqn(dataset)
                datas.append([dataset,model,qn/search_time,construction_time,recall,1])
    return datas

def get_processed_data_glass(folderName,model):
    datas=[]
    for file in os.listdir('raw/'+folderName):
        with open('raw/'+folderName+'/'+file, "r", encoding="utf-8") as f:
            content = f.read()
            count = content.count('recall@')
            if count>=1:
                recall=content.split("'recall@': ")[1].split(',')[0]
                dataset=content.split("'dataset_name': '")[1].split("'")[0]
                search_time=float(content.split("'search-time': ")[1].split(",")[0])
                construction_time=content.split("'training-time': ")[1].split(",")[0]
                dataset,qn=getqn(dataset)
                datas.append([dataset,model,qn/search_time,construction_time,recall,1])
    return datas

def get_processed_data_mrpt(folderName,model):
    datas=[]
    for file in os.listdir('raw/'+folderName):
        with open('raw/'+folderName+'/'+file, "r", encoding="utf-8") as f:
            content = f.read()
            count = content.count('recall')
            if count>=1:
                if "'I'" in content:
                    recall=float(content.split("'recall': ")[1].split(',')[0])
                else:
                    recall=float(content.split("'recall': ")[1].split('}')[0])
                dataset=content.split("'dataset': '")[1].split("'")[0]
                search_time=float(content.split("'search_time': ")[1].split(",")[0])
                construction_time=content.split("'construction_time': ")[1].split(",")[0]
                dataset,qn=getqn(dataset)
                datas.append([dataset,model,qn/search_time,construction_time,recall,1])
    return datas




def get_processed_data_flann(folderName):
    datas=[]
    for file in os.listdir('raw/'+folderName):
        with open('raw/'+folderName+'/'+file, "r", encoding="utf-8") as f:
            content = f.read()
            count = content.count("'recall'")
            if count>=1 and 'PQFS' in file:
                recall=content.split("'recall': ")[1].split(',')[0]
                dataset=content.split("'dataset': '")[1].split("'")[0]
                search_time=float(content.split("'query_time': ")[1].split(",")[0])
                construction_time=content.split("User time (seconds): ")[1].split("\n")[0]
                dataset,qn=getqn(dataset)
                datas.append([dataset,'FLANN',qn/search_time,construction_time,recall,1])
            if count>=1 and 'KDTREE' in file:
                recall=content.split("'recall': ")[1].split(',')[0]
                dataset=content.split("'dataset': '")[1].split(",")[0]
                search_time=float(content.split("'query_time': ")[1].split(",")[0])
                construction_time=content.split("User time (seconds): ")[1].split("\n")[0]
                dataset,qn=getqn(dataset)
                datas.append([dataset,'KDTREE',qn/search_time,construction_time,recall,1])
    return datas
    

def get_processed_data_nndescent(folderName,model):
    datas=[]
    for file in os.listdir('raw/'+folderName):
        with open('raw/'+folderName+'/'+file, "r", encoding="utf-8") as f:
            content = f.read()
            count = content.count('recall@')
            if count>=1:
                recall=content.split("'recall@': ")[1].split(',')[0]
                dataset=content.split("'dataset': '")[1].split("'")[0]
                search_time=float(content.split("'query time': ")[1].split("}")[0])
                construction_time=content.split("'training-time': ")[1].split(",")[0]
                dataset,qn=getqn(dataset)
                datas.append([dataset,model,qn/search_time,construction_time,recall,1])
    return datas


def get_dataset_name(fileName):
    for i in range(20,-1,-1):
        pt=str(i)+'.txt'
        if pt in fileName:
            return fileName.replace(pt,'')
    return fileName.replace('.txt','')

def get_processed_data_dpg(folderName):
    datas=[]
    for file in os.listdir('raw/'+folderName):
        with open('raw/'+folderName+'/'+file, "r", encoding="utf-8") as f:
            construction_time=None
            search_time=None
            recall=None
            dataset=get_dataset_name(file)
            dataset,qn=getqn(dataset)
            
            for line in f:
                if 'Build cost: ' in line:
                    construction_time=line.split('Build cost: ')[1].replace('\n','')
                if 'search time: ' in line:
                    search_time=float(line.split('search time: ')[1].replace('\n',''))
                if 'NN recall: ' in line:
                    recall=line.split('NN recall: ')[1].replace('\n','')
                    datas.append([dataset,'DPG',qn/search_time,construction_time,recall,1])
    return datas    

def get_processed_data_efanna(folderName):
    datas=[]
    construction_time=None
    for file in os.listdir('raw/'+folderName):
        with open('raw/'+folderName+'/'+file, "r", encoding="utf-8") as f:
            content = f.read()
            construction_time=content.split('Time cost: ')[1].replace('\n','')
        with open('raw/'+folderName+'/'+file, "r", encoding="utf-8") as f:
            search_time=None
            recall=None
            dataset=get_dataset_name(file)
            dataset,qn=getqn(dataset)
            
            for line in f:
                if 'search time: ' in line:
                    search_time=float(line.split('search time: ')[1].replace('\n',''))
                if 'NN recall: ' in line:
                    search_time=float(line.split('NN recall: ')[1].replace('\n',''))
                    datas.append([dataset,'EFANNA',qn/search_time,construction_time,recall,1])
    return datas      

def get_processed_data_hcnng(folderName):
    datas=[]
    for file in os.listdir('raw/'+folderName):
        with open('raw/'+folderName+'/'+file, "r", encoding="utf-8") as f:
            construction_time=None
            search_time=None
            recall=None
            dataset=get_dataset_name(file)
            dataset,qn=getqn(dataset)
            
            for line in f:
                if '__INIT FINISH__: ' in line:
                    construction_time=line.split('__INIT FINISH__: ')[1].replace('\n','')
                if 'search time: ' in line:
                    search_time=float(line.split('search time: ')[1].replace('\n',''))
                if 'NN recall: ' in line:
                    recall=line.split('NN recall: ')[1].replace('\n','')
                    datas.append([dataset,'HCNNG',qn/search_time,construction_time,recall,1])
    return datas           

import ssg_extractor


def get_ssg_data():
    datass=ssg_extractor.extract('/home/saminyeaser/OSU study/Research-Implementation/models/SSG/logs')
    datas=[]
    for data in datass:
        dataset,qn=getqn(data[0])
        datas.append([dataset,'NSSG',qn/data[4],data[5],data[2],1])
    os.chdir('/home/saminyeaser/demo_app_code/data')
    return datas

def get_all_faiss_results(model):
    os.chdir('/home/saminyeaser/OSU study/Research-Implementation/models/VAQ22/data/faiss-library')
    lines=[]
    for folder in os.listdir():
        if (not '.' in folder) and ('log' in folder):
            os.chdir(folder)
            for file in os.listdir():
                if '.txt' in file:
                     with open(file, "r", encoding="utf-8") as f:
                         for line in f:
                             if model in line:
                                 lines.append(line.replace('\n',''))
                    
            os.chdir('..')
    os.chdir('/home/saminyeaser/demo_app_code/data')
    return lines


def get_all():
    all_datas=[]
    all_datas.extend(get_processed_data_graph())
    all_datas.extend(get_processed_data_quantization())
    all_datas.extend(get_processed_data_hash())
    all_datas.extend(get_processed_data_tree())
    all_datas.extend(get_processed_data_vaq())
    all_datas.extend(get_processed_data_diskann('logDiskann','DISKANN'))
    all_datas.extend(get_processed_data_glass('logHNSWGlass','GLASS-HNSW'))
    all_datas.extend(get_processed_data_glass('logNSGGlass','GLASS-NSG'))
    all_datas.extend(get_processed_data_mrpt('mrptlogs','MRPT'))
    all_datas.extend(get_processed_data_flann('logFLANN'))
    all_datas.extend(get_processed_data_nndescent('logNNdescent','NNdescent'))
    all_datas.extend(get_processed_data_dpg('new_dpg_logs'))
    all_datas.extend(get_processed_data_efanna('original_efanna_logs'))
    all_datas.extend(get_processed_data_hcnng('new_efanna_logs'))
    all_datas.extend(get_processed_data_hcnng('hcnng_logs'))
    all_datas.extend(get_ssg_data())
    return all_datas

all_datas=get_all()


lines=get_all_faiss_results("OPQ,IMI2x1")

lines.extend(get_all_faiss_results("OPQ,IMI2x2"))

lines=get_all_faiss_results("PQ,IMI2x1")

lines.extend(get_all_faiss_results("PQ,IMI2x2"))

for line in lines:
    dataset=line.split("'dataset_name': '")[1].split("'")[0]
    search_time=float(line.split("'search-time': ")[1].split(",")[0])
    construction_time=float(line.split("'construction-time': ")[1].split(",")[0])
    recall=float(line.split("'recall@': ")[1].split(",")[0])
    model=line.split("'model_name': '")[1].split("'")[0]
    dataset,qn=getqn(dataset)
    all_datas.append([dataset,model,qn/search_time,construction_time,recall,1])
    
lines=get_all_faiss_results("PQFS")

for line in lines:
    dataset=line.split("'dataset_name': '")[1].split("'")[0]
    search_time=float(line.split("'search-time': ")[1].split(",")[0])
    construction_time=float(line.split("'training-time': ")[1].split(",")[0])
    recall=float(line.split("'recall@': ")[1].split(",")[0])
    model=line.split("'model_name': '")[1].split("'")[0]
    dataset,qn=getqn(dataset)
    all_datas.append([dataset,model,qn/search_time,construction_time,recall,1])
    
lines=get_all_faiss_results(": 'PQ'")
# print(lines)

for line in lines:
    print(line)
    print('\n')
    if "'recall@': " in line and "'search-time': " in line:
        dataset=line.split("'dataset_name': '")[1].split("'")[0]
        search_time=float(line.split("'search-time': ")[1].split(",")[0])
        construction_time=float(line.split("'training-time': ")[1].split(",")[0])
        recall=float(line.split("'recall@': ")[1].split(",")[0])
        model=line.split("'model_name': '")[1].split("'")[0]
        dataset,qn=getqn(dataset)
        if (not 'znorm' in dataset) and (not 'hundred' in dataset) and (not 'twenty' in dataset):
            all_datas.append([dataset,model,qn/search_time,construction_time,recall,1])
    
    
lines=get_all_faiss_results(": 'OPQ'")
# print(lines)

for line in lines:
    # print(line)
    if "'recall@': " in line and "'search-time': " in line:
        dataset=line.split("'dataset_name': '")[1].split("'")[0]
        search_time=float(line.split("'search-time': ")[1].split(",")[0])
        construction_time=float(line.split("'training-time': ")[1].split(",")[0])
        recall=float(line.split("'recall@': ")[1].split(",")[0])
        model=line.split("'model_name': '")[1].split("'")[0]
        dataset,qn=getqn(dataset)
        if (not 'znorm' in dataset) and (not 'hundred' in dataset) and (not 'twenty' in dataset):
            all_datas.append([dataset,model,qn/search_time,construction_time,recall,1])
        

    
lines=get_all_faiss_results(": 'IVFPQ'")
# print(lines)

for line in lines:
    # print(line)
    if "'recall@': " in line and "'search-time': " in line:
        dataset=line.split("'dataset_name': '")[1].split("'")[0]
        search_time=float(line.split("'search-time': ")[1].split(",")[0])
        construction_time=float(line.split("'training-time': ")[1].split(",")[0])
        recall=float(line.split("'recall@': ")[1].split(",")[0])
        model=line.split("'model_name': '")[1].split("'")[0]
        dataset,qn=getqn(dataset)
        if (not 'znorm' in dataset) and (not 'hundred' in dataset) and (not 'twenty' in dataset):
            all_datas.append([dataset,model,qn/search_time,construction_time,recall,1])
        
        


    
lines=get_all_faiss_results("vamana")

for line in lines:
    dataset=line.split("'dataset_name': '")[1].split("'")[0]
    search_time=float(line.split("'search-time': ")[1].split(",")[0])
    construction_time=float(line.split("'construction-time': ")[1].split(",")[0])
    recall=float(line.split("'recall@': ")[1].split(",")[0])
    model=line.split("'model_name': '")[1].split("'")[0]
    dataset,qn=getqn(dataset)
    all_datas.append([dataset,model,qn/search_time,construction_time,recall,1])
    
lines=get_all_faiss_results("ITQ-LSH")

for line in lines:
    dataset=line.split("'dataset_name': '")[1].split("'")[0]
    search_time=float(line.split("'search-time': ")[1].split(",")[0])
    construction_time=float(line.split("'training-time': ")[1].split(",")[0])
    recall=float(line.split("'recall@': ")[1].split(",")[0])
    model=line.split("'model_name': '")[1].split("'")[0]
    dataset,qn=getqn(dataset)
    all_datas.append([dataset,model,qn/search_time,construction_time,recall,1])
    
    
def cleanString(s):
    return s.replace(' ','').replace('\t','').replace('\n','')

def get_qualsh_data(folder):
    os.chdir(folder)
    datas=[]
    for file in os.listdir():
        construction_time=None
        with open(file, "r", encoding="utf-8") as f:
            Lines = f.readlines()
            last_index=0
            qps=None
            recall=None
            ds=None
            for i in range(0,len(Lines)):
                ln=Lines[i]
                if ln.startswith('Indexing Time = '):
                    lll=ln.split('Indexing Time = ')[1].split(' Seconds')[0]
                    construction_time=lll
                elif ln.startswith('k-NN Search by QALSH:'):
                    nxt=Lines[i+2]
                    nxt_splitted=nxt.split('\t')
                    qps=1/(float(cleanString(nxt_splitted[4]))/1000)
                    recall=float(cleanString(nxt_splitted[6]))/100
                    datas.append([ds.replace('\n',''),'QALSH',qps,construction_time,recall,1])
                    
                elif ln.startswith('prefix = '):
                    lll=ln.split('prefix = /data/kabir/similarity-search/dataset')[1].replace('/','')
                    ds=lll
                    if 'gist' in ln:
                        ds='gist'
                        
    os.chdir('/home/saminyeaser/demo_app_code/data')
    return datas
        
    
    
all_datas.extend(get_qualsh_data('/home/saminyeaser/logsq'))
    
    
    

columns = [
    "dataset",
    "model",
    "qps",
    "construction_time",
    "recall",
    "refinex"
]



models=set()
datasets=set()

for data in all_datas:
    models.add(data[1])
    datasets.add(data[0])

print(len(models))
print(models)
print(len(datasets))

for data in all_datas:
    if len(data[0])>40:
        print(data)
        break
    
print(sorted(list(datasets)))






lines=get_all_faiss_results("'num_leaves_to_search_'")
# print(lines[20])
print(lines)
print(len(lines))

ds=set()
for ln in lines:
    if 'x2' in ln:
        print(ln)
        dss=ln.split("'dataset_name': '")[1].split("'")[0]
        print(dss)
        ds.add(dss)
print(len(ds))

lines=get_all_faiss_results("'annoy'")
print(len(lines))

# print(lines)

for line in lines:
    print(line)
    dataset=line.split("'dataset_name': '")[1].split("'")[0]
    search_time=float(line.split("'search-time': ")[1].split(",")[0])
    construction_time=float(line.split("'construction-time': ")[1].split(",")[0])
    if "'recall@': " in line:
        recall=float(line.split("'recall@': ")[1].split(",")[0])
    else:
        recall=float(line.split("'recall': ")[1].split(",")[0])
    model=line.split("'model_name': '")[1].split("'")[0]
    dataset,qn=getqn(dataset)
    all_datas.append([dataset,model,qn/search_time,construction_time,recall,1])


lines=get_all_faiss_results("'NSG'")
print(len(lines))
# print(lines)

for line in lines:
    print(line)
    splits=line.split("'recall@': ")
    for i in range(1,len(splits)):
        
        s=splits[i]
        print(s)
        recall=float(s.split(',')[0])
        dataset=s.split("'dataset_name': '")[1].split("'")[0]
        construction_time=float(s.split("'training-time': ")[1].split(",")[0])
        search_time=float(s.split("'search-time': ")[1].split(",")[0])
        model='NSG'
        dataset,qn=getqn(dataset)
        all_datas.append([dataset,model,qn/search_time,construction_time,recall,1])
        


lines=get_all_faiss_results("'HNSW'")
print(len(lines))


for line in lines:
    # print(line)
    dataset=line.split("'dataset_name': '")[1].split("'")[0]
    search_time=float(line.split("'search-time': ")[1].split(",")[0])
    construction_time=float(line.split("'training-time': ")[1].split(",")[0])
    if "'recall@': " in line:
        recall=float(line.split("'recall@': ")[1].split(",")[0])
    else:
        recall=float(line.split("'recall': ")[1].split(",")[0])
    model=line.split("'model_name': '")[1].split("'")[0]
    dataset,qn=getqn(dataset)
    all_datas.append([dataset,model,qn/search_time,construction_time,recall,1])


def get_all_faiss_results_v1(model):
    os.chdir('/home/saminyeaser/OSU study/Research-Implementation/models/VAQ22/data/faiss-library')
    lines=[]
    for folder in os.listdir():
        if (not '.' in folder) and ('log' in folder):
            os.chdir(folder)
            for file in os.listdir():
                if '.txt' in file:
                     with open(file, "r", encoding="utf-8") as f:
                         for line in f:
                             if model in line:
                                 lines.append((line.replace('\n',''),file))
                    
            os.chdir('..')
    os.chdir('/home/saminyeaser/demo_app_code/data')
    return lines

lines=get_all_faiss_results_v1("'FlatNav'")
print(len(lines))
# print(lines)



for line_p in lines:
    # print(line_p)
    line=line_p[0]
    dataset=line_p[1].split('-')[0]
    if "'search-time': " in line:
        search_time=float(line.split("'search-time': ")[1].split(",")[0])
    if "'search_time': " in line:
        search_time=float(line.split("'search_time': ")[1].split(",")[0])
    construction_time=float(line.split("'construction_time': ")[1].split(",")[0])
    if "'recall@': " in line:
        recall=float(line.split("'recall@': ")[1].split(",")[0])
    else:
        recall=float(line.split("'recall': ")[1].split(",")[0])
    model='FLATNAV'
    dataset,qn=getqn(dataset)
    all_datas.append([dataset,model,qn/search_time,construction_time,recall,1])
    
    
    
lines=get_all_faiss_results("'RabitQ'")
# print(lines)
# print(len(lines))


for line in lines:
    # print(line)
    dataset=line.split("'dataset_name': '")[1].split("'")[0]
    search_time=float(line.split("'search-time': ")[1].split(",")[0])
    construction_time=float(line.split("'training-time': ")[1].split(",")[0])
    if "'recall@': " in line:
        recall=float(line.split("'recall@': ")[1].split(",")[0])
    else:
        recall=float(line.split("'recall': ")[1].split(",")[0])
    model=line.split("'model_name': '")[1].split("'")[0]
    dataset,qn=getqn(dataset)
    all_datas.append([dataset,model,qn/search_time,construction_time,recall,1])
    
lines=get_all_faiss_results("'lorann'")
# print(lines)
print(len(lines))


for line in lines:
    # print(line)
    dataset=line.split("'dataset': '")[1].split("'")[0]
    search_time=float(line.split("'search_time': ")[1].split(",")[0])
    construction_time=float(line.split("'construction_time': ")[1].split(",")[0])
    recall=float(line.split("'recall': ")[1].split(",")[0])
    model=line.split("'model': '")[1].split("'")[0]
    dataset,qn=getqn(dataset)
    all_datas.append([dataset,model,qn/search_time,construction_time,recall,1])
    search_time=float(line.split("'search_timex2': ")[1].split(",")[0])
    if not 'Command being timed' in line:
        recall=float(line.split("'recallx2': ")[1].replace('\t','').split("}")[0])
    else:
        recall=float(line.split("'recallx2': ")[1].split("\t")[0].replace('}',''))
    all_datas.append([dataset,model,qn/search_time,construction_time,recall,1])






def get_vaq_data():
    os.chdir('/home/saminyeaser/demo_app_code/data/raw/VAQ')
    datas=[]
    for folder in os.listdir():
        os.chdir(folder)
        os.chdir('finalTests2')
        for file in os.listdir():
            recalls=[]
            search_time=[]
            training_time=None
            construction_time=None
            dataset=None
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    if '== Training time: ' in line:
                        training_time=float(line.split('== Training time: ')[1].split(' s')[0])
                    if '== Encoding time: ' in line:
                        construction_time=float(line.split('== Encoding time: ')[1].split(' s')[0])
                    if 'Querying time: ' in line:
                        search_time.append(float(line.split('Querying time: ')[1].split(' s')[0]))
                    if 'recall@R (Probably correct): ' in line:
                        recalls.append(float(line.split('recall@R (Probably correct): ')[1].split(' ')[0]))
                    if 'dataset = ../../../dataset/' in line:
                        dataset=line.split('dataset = ../../../dataset/')[1].split('/')[0]
                if dataset!=None:
                    dataset,qn=getqn(dataset)
                    for i in range(len(recalls)):
                        print('reach here')
                        print([dataset,model,qn/search_time[i],training_time+construction_time,recalls[i],1])
                        datas.append([dataset,'VAQ',qn/search_time[i],training_time+construction_time,recalls[i],1])
            
        os.chdir('..')
        os.chdir('..')
    os.chdir('/home/saminyeaser/demo_app_code/data')
    return datas

all_datas.extend(get_vaq_data())


def get_pmlsh_data():
    os.chdir('raw/pmlsh_logs')
    for file in os.listdir():
        with open(file, "r", encoding="utf-8") as f:
            content=f.read()
            if ("FINISH BUILDING WITH TIME: " in content):
                dataset=content.split('Using PM-LSH for ')[1].split(' ...')[0]
                construction_time=float(content.split('FINISH BUILDING WITH TIME: ')[1].split(' s')[0])
                search_time=1/(float(content.split('AVG QUERY TIME:    ')[1].split('ms')[0])/1000)
                recall=content.split('AVG RECALL:')[1].split('\n')[0].replace(' ','')
                all_datas.append([dataset,'PM-LSH',search_time,construction_time,recall,1])
                print([dataset,'PM-LSH',search_time,construction_time,recall,1])
            
        
    os.chdir('..')
    os.chdir('..')
get_pmlsh_data()

df_original = pd.DataFrame(all_datas, columns=columns)

df_original.to_csv("processed/processed_polyvector_results_original.csv", index=False)

df_preporcessed = pd.read_csv("processed/preporcessed.csv")



df_out = pd.concat([df_original, df_preporcessed], ignore_index=True)


df_out.to_csv("processed/processed_polyvector_results.csv", index=False)





df = pd.read_csv("processed/processed_polyvector_results.csv")
datasets,model = set(df['dataset'].tolist()),set(df['model'].tolist())
print(sorted(list(datasets)))
print(len(list(datasets)))
print(sorted(list(model)))
print(len(list(model)))
# print(df_out['model'].tolist())
# pq_excludes=set()

# for data in all_datas:
#     if data[1]=='PQ':
#         pq_excludes.add(data[0])
        
# print(pq_excludes)
# print(len(pq_excludes))



# opq_excludes=set()

# for data in all_datas:
#     if data[1]=='OPQ':
#         opq_excludes.add(data[0])
        
# print(opq_excludes)
# print(len(opq_excludes))



# pqfs_excludes=set()

# for data in all_datas:
#     if data[1]=='PQFS':
#         pqfs_excludes.add(data[0])
        
# print(pqfs_excludes)
# print(len(pqfs_excludes))




# efanna_excludes=set()

# for data in all_datas:
#     if data[1]=='vamana':
#         efanna_excludes.add(data[0])
        
# print(efanna_excludes)
# print(len(efanna_excludes))

# efanna_excludes=set()

# for data in all_datas:
#     if data[1]=='vamana_LVQ':
#         efanna_excludes.add(data[0])
        
# print(efanna_excludes)
# print(len(efanna_excludes))

