"""simple relation scheme converter for classification problems (simpler than add_features)

input is csv of instances (with orange feature names convention)
argument must specify output scheme among: []

output to standard output

"""

import sys

import Orange

# modifies self
def collapse_values(self, featurename, mapping, new_values):
        """use a set of super-classes to replace a more profligate set of classes
        new_classes is a dict of set
        """
        for val in new_values: 
            self.domain[featurename].add_value(val)
        for one in self:
            # if unknown label, keep as is (not ideal)
            one[featurename] = mapping.get(one[featurename].value, one[featurename])

def set2mapping(sets):
    """ invert a dictionary of key:set to make a mapping from elements of sets to key
    eg   {"c1":set(1,2,3),"c2":set(4,5,6)} -> {1:"c1",2:"c1", ...}
    """
    result = {}
    for key in sets:
        for element in sets[key]:
            result[element] = key
    return result


class_schemes = {
    # two-class classification of relations / subord-coord
    "subord_coord":{
        "subord":set(["elaboration", "e-elab", "attribution", "comment", "flashback", "explanation", "alternation"]),
        "coord": set(["continuation", "parallel", "contrast", "temploc", "frame", "narration", "conditional", "result", "goal", "background"]),
        "NONE":set(["null", "unknown", "NONE"])
        },
    # four class +/- closer to PDTB
    "pdtb": {
        "contingency": set(["explanation", "conditional", "result", "goal", "background"]),
        "temporal": set(["temploc", "narration", "flashback"]),
        "comparison":set(["parallel", "contrast"]),
        "expansion":set(["frame", "elaboration", "e-elab", "attribution", "continuation", "comment", "alternation"]),
        "error":set(["null", "unknown", "NONE"])
        },
    # our own hierarchy
    "minsdrt":{
        "structural": set(["parallel", "contrast", "alternation", "conditional"]),
        "sequence": set(["result", "narration", "continuation"]),
        "expansion": set(["frame", "elaboration", "e-elab", "attribution", "comment", "explanation"]),
        "temporal": set(["temploc", "goal", "flashback", "background"]),
        "error":set(["null", "unknown", "NONE"])
        }
}
# coord -> contingency+structural, subord -> expansion + temporal

data = Orange.data.Table(sys.argv[1])
target = sys.argv[2]
if len(sys.argv)>3:
    output = sys.argv[3]
else:
    output = "out.csv"
if target not in class_schemes.keys(): 
    print >> sys.stderr, "unknown relation schema"
    sys.exit(0)
mapping = set2mapping(class_schemes[target])
new_values = class_schemes[target].keys()

collapse_values(data,"CLASS",mapping,new_values)
data.save(output)

#data = data.filter_ref({"CLASS":["UNRELATED"]}, negate = 1)

