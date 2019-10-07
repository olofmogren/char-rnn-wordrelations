#!/usr/bin/python3

import rdflib
g = rdflib.Graph()
#result = g.parse('trees.ttl') 
#result = g.parse('trees.ttl', format='ttl')
result = g.parse('data/sv_dbnary_ontolex_20170601.ttl', format='turtle')
print(len(g))
for stmt in g:
  print(stmt)

