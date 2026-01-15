import os

mp={}
new_mp={}
perf=[]

def addConstructionFlag(perf):
    converted=[]
    for i in range(0,len(perf)):
        c_f=1
        s_f=1
        for j in range(i+1,len(perf)):
            if perf[i][0]!=perf[j][0]:
                break
            if perf[j][5]<perf[i][5]:
                c_f=0
            if perf[j][4]<perf[i][4]:
                s_f=0
        converted.append((perf[i][0], perf[i][1], perf[i][2], perf[i][3], perf[i][4], perf[i][5], perf[i][6], s_f, c_f))
    return converted


def getDatasetInfo(fileName):
    split=fileName[:len(fileName)-2]
    sub_case=fileName.replace(split,'').replace('-','')
    construction_time=0
    case=''
    if split[len(split)-1]>='0' and split[len(split)-1]<='9':
        case=split[len(split)-1]
        if split[len(split)-2]>='0' and split[len(split)-2]<='9':
            case=split[len(split)-2]+case
        split=split[0:len(split)-len(case)]
    else:
        case='100'
    
    
    
    new_mp[fileName]=mp[fileName]
    new_mp[fileName]['dataset']=split
    new_mp[fileName]['case']=case
    new_mp[fileName]['sub-case']=sub_case
    {'search-time': 0.0480334, 'recall': 0.71, 'map': 0.61657, 'efanna-constructime': 297.301, 'ssg-constructime': 417.365, 'construction-time': 714.6659999999999, 'dataset': 'sald1m', 'case': '7', 'sub-case': '3'}
    perf.append((split, 'NSSG', float(new_mp[fileName]['recall']), new_mp[fileName]['map'], new_mp[fileName]['search-time'], new_mp[fileName]['construction-time'], new_mp[fileName]['case']+'-'+new_mp[fileName]['sub-case'], 1, 1))

def setValue(case, key, data):
    if case not in mp.keys():
        mp[case]={}
    mp[case][key]=data
    
def setValueInList(key,data):
    if not key in mp.keys():
        mp[key]=[data]
    mp[key].append(data)


def extractSingleFile(fileName):
    if fileName.startswith('efannlog'):
        key=fileName.replace('efannlog','').replace('.txt','')
        data=float(open(fileName,'r').read().split('Time cost: ')[1].replace('\n',''))
        setValue(key, 'efanna-constructime', data)
    if fileName.startswith('ssgbuildlog'):
        key=fileName.replace('ssgbuildlog','').replace('.txt','')
        data=float(open(fileName,'r').read().split('Build Time: ')[1].replace('\n',''))
        setValue(key, 'ssg-constructime', data)
    if fileName.startswith('ssgsearchlog'):
        key=fileName.replace('ssgsearchlog','').replace('.txt','')
        for line in open(fileName,'r').readlines():
            if 'Search Time: ' in line:
                data=float(line.split('Search Time: ')[1].replace('\n',''))
                setValue(key, 'search-time',data)
            if 'Recall: ' in line:
                data=float(line.split('Recall: ')[1].replace('\n',''))
                setValue(key, 'recall',data)
            if 'MAP: ' in line:
                data=float(line.split('MAP: ')[1].replace('\n',''))
                setValue(key, 'map',data)
        

def extract(loc):
    os.chdir(loc)
    fileNames=os.listdir()
    for fileName in fileNames:
        extractSingleFile(fileName)
    for key in mp.keys():
        if key[len(key)-2]=='-':
            con_key=key[:len(key)-2]
            for s_con_key in mp[con_key]:
                mp[key][s_con_key]=mp[con_key][s_con_key]
            mp[key]['construction-time']=mp[key]['ssg-constructime']+mp[key]['efanna-constructime']
    # print(mp)
    for key in mp.keys():
        if key[len(key)-2]=='-':
            getDatasetInfo(key)
    con=addConstructionFlag(perf)
    return sorted(con)
    
