#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Duplicate features with respect to a given feature value

use case: duplicate features to separate prediction of attachment and labelling
of inter- and intra-sentence rhetorical relations. 

depending on the value of given feature, all binary features of one or the other will be set to 0/False
continuous values are unchanged. non binary categorial features will be unchanged. 

!Warning!: this makes sense only for learning methods that encode the model as
a weight vector

eg, usage: 
duplicateFeatures inputdata.csv SAME_SENTENCE outputdata.csv

author: P. Muller / Melodi group / IRIT-Univ Toulouse

TODO: data cloning in orange is ... painful. maybe this can be done as a pure csv to csv transformation
"""

import sys, csv

MODE="csv"#"orange"
DELIMITER=","

try: 
    import Orange
except:
    MODE="csv"

data_file = sys.argv[1]
split_feature=sys.argv[2]
if len(sys.argv)>3:
    out_file = sys.argv[3]
else:
    out_file = None

if MODE=="csv":
    data = [x for x in csv.reader(open(data_file),delimiter=DELIMITER)]
    header = data[0]
    data = data[1:]
    domain = dict(zip(header,map(set,zip(*data))))    
    
else:
    data = Orange.data.Table()
    domain = data.domain
    
try: 
        a = domain[split_feature]
except:
        print >> sys.stderr, "ERROR: feature %s is not present in data... aborting"%split_feature
        sys.exit(0)




def orig2new(label,split_feature):
    """duplicate a feature name wrt a given feature"""
    return "%s_%s"%(label,split_feature)

def new2orig(label,split_feature):
    """get back the original name of a duplicate feature"""
    return label.split("_"+split_feature)[0]

# orange
#def is_binary(feature):
#    return repr(x.var_type)=="Discrete" and len(x.values)==2
#def classVar(domain):
#    return domain.classVar
#def features(domain):
#    return domain.attributes()

# pure csv
def is_binary(feature,data={}):
    return feature.startswith("D#") and data.get(feature,[True])[0] in [True,False]

def classVar(domain):
    return filter(lambda x: x.startswith("c#"), domain)[0]

def features(domain):
    return domain.keys()
# features to duplicate are binay features
#

relevant_features = filter(lambda x: is_binary(x) and x!=classVar(domain) and x!=split_feature,features(domain))

# labels for the duplicate features
new_features = []


for feature in relevant_features:
    dupl_name = orig2new(feature,split_feature)
    #dupl = Orange.feature.Discrete(dupl_name, values=["True","False"])
    new_features.append(dupl_name)



#new_domain = Orange.data.Domain(new_features+data.domain,data.domain.classVar)
#original_features=list(data.domain.attributes)
#new_domain = Orange.data.Domain(new_features+original_features,data.domain.classVar)
#new_data =  Orange.data.Table(new_domain, data)

# add meta / data.domain.get_metas() + id = Orange.feature.Descriptor.new_meta_id()
# + data.domain.add_meta(id, x)

header.extend(new_features)
index = dict([(x,y) for (y,x) in enumerate(header)])

for (i,instance) in enumerate(data): 
    instance.extend([None]*len(new_features))    

    true_4_split = bool(instance[index[split_feature]])

 
    for feature in new_features:
            if true_4_split:
               #instance is true for the split feature, so erase the original feature values
                # and put them in the new features
                instance[index[feature]] = instance[index[new2orig(feature,split_feature)]]
                instance[index[new2orig(feature,split_feature)]] = "False"
            else:
                #instance is false for the split feature, keep the original value, set the new to 0/False
                #instance[index[new2orig(feature,split_feature)]] unchanged
                instance[index[feature]] = "False"
    for feature in relevant_features:
            if true_4_split:
                instance[index[feature]]="False"
            else:
                pass
                #instance[index[feature]] unchanged


if MODE=="orange":
    data.save(out_file)
else:
    if out_file:
        out_file = open(out_file,"w")
    else:
        out_file = sys.stdout
    print >> out_file, DELIMITER.join(header)
    print >> out_file, "\n".join([DELIMITER.join(x) for x in data])
