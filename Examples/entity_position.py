result = []
for x in all_problems_in_text:
    if x:
        index = x.split('|')
        tmp = [i.split() for i in index]
        result.append(tmp)
    else: 
        result.append(x)
        
        
